"""
generate_fake_data.py
Populates the local DuckDB database with non-live alert data for local testing,
from two possible sources:

  random   Realistic-looking SYNTHETIC data, generated locally. No internet needed.
  kaggle   Real historical data loaded from the Kaggle "Air-raid sirens in Ukraine"
           dataset (cashncarry/airraid-sirens-in-ukraine). Supports either CSV that
           dataset ships, auto-detected, and matches oblast names in either Ukrainian
           or English (always storing the canonical Ukrainian name, since that's what
           the dashboard's choropleth map joins on):
             oblasts_only.csv  -- region, started_at, finished_at, naive
             full_data.csv     -- oblast, raion, hromada, level, started_at, finished_at
           https://www.kaggle.com/datasets/cashncarry/airraid-sirens-in-ukraine

Neither source calls the real alerts.in.ua API -- no ALERTS_API_TOKEN required.

Run:
    uv run python generate_fake_data.py random                        # 30 days, ~15 alerts/day
    uv run python generate_fake_data.py random --days 60 --avg-per-day 20
    uv run python generate_fake_data.py random --clear                # wipe old synthetic rows first
    uv run python generate_fake_data.py random --seed 42               # reproducible output

    uv run python generate_fake_data.py kaggle                         # loads data/oblasts_only.csv
    uv run python generate_fake_data.py kaggle --csv data/full_data.csv  # hromada-level detail
    uv run python generate_fake_data.py kaggle --clear                 # wipe old Kaggle rows first

Each source tags its rows with a distinct `notes` value and a distinct id range, so they're
always identifiable and can be cleared independently of each other and of real ingested data:
    random  -> notes = 'SYNTHETIC TEST DATA',      ids in [RANDOM_ID_BASE, RANDOM_ID_BASE + 100_000_000)
    kaggle  -> notes = 'KAGGLE_HISTORICAL_DATA',    ids in [KAGGLE_ID_BASE, KAGGLE_ID_BASE + 100_000_000)
"""

from __future__ import annotations

import argparse
import random
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from ingest import Alert, OBLAST_UIDS, get_connection, save_alerts

# ---------------------------------------------------------------------------
# Shared id-range bookkeeping (keeps "random" and "kaggle" rows from colliding,
# and from colliding with real API ids, which are nowhere near this range).
# ---------------------------------------------------------------------------
KAGGLE_ID_BASE = 800_000_000
RANDOM_ID_BASE = 900_000_000
ID_RANGE_SIZE = 100_000_000  # each source gets a 100M-wide id band

KAGGLE_DATA_NOTE = "KAGGLE_HISTORICAL_DATA"
FAKE_DATA_NOTE = "SYNTHETIC TEST DATA"


def _clear_id_range(id_base: int, label: str) -> None:
    """Delete rows whose id falls in [id_base, id_base + ID_RANGE_SIZE) -- one source's worth."""
    con = get_connection()
    upper = id_base + ID_RANGE_SIZE
    count = con.execute(
        "SELECT count(*) FROM alerts WHERE id >= ? AND id < ?", [id_base, upper]
    ).fetchone()[0]
    con.execute("DELETE FROM alerts WHERE id >= ? AND id < ?", [id_base, upper])
    print(f"Cleared {count} previously loaded {label} record(s).")


# ---------------------------------------------------------------------------
# Oblast name matching, language-agnostic.
#
# Different sources name oblasts differently -- the live API and OBLAST_UIDS use Ukrainian
# Cyrillic ("Вінницька область"), while this Kaggle dataset uses English transliteration
# ("Vinnytska oblast"). Every row gets matched here and then stored with the SAME canonical
# Ukrainian name OBLAST_UIDS already uses, regardless of which language the source used --
# this matters because the choropleth map joins on that exact Ukrainian name against the
# GeoJSON, so an English name slipping into location_oblast would silently break the map.
# ---------------------------------------------------------------------------

# English adjectival root for each oblast uid (without the "oblast"/"region" suffix).
ENGLISH_OBLAST_ROOTS = {
    3: "Khmelnytska", 4: "Vinnytska", 5: "Rivnenska", 8: "Volynska",
    9: "Dnipropetrovska", 10: "Zhytomyrska", 11: "Zakarpatska", 12: "Zaporizka",
    13: "Ivano-Frankivska", 14: "Kyivska", 15: "Kirovohradska", 16: "Luhanska",
    17: "Mykolaivska", 18: "Odeska", 19: "Poltavska", 20: "Sumska",
    21: "Ternopilska", 22: "Kharkivska", 23: "Khersonska", 24: "Cherkaska",
    25: "Chernihivska", 26: "Chernivetska", 27: "Lvivska", 28: "Donetska",
}
# Crimea/Kyiv-city/Sevastopol-city don't follow the "<Root> oblast" pattern.
SPECIAL_NAME_TO_UID = {
    "autonomous republic of crimea": 29, "crimea": 29, "ark": 29,
    "sevastopol": 30, "sevastopol city": 30, "m. sevastopol": 30,
    "kyiv city": 31, "m. kyiv": 31,
}


def _normalize_oblast_name(name: str) -> str:
    """Lowercase, drop 'oblast'/'область'/'region' suffix words, collapse whitespace."""
    text = str(name).strip().lower()
    text = re.sub(r"\b(oblast|область|region)\b\.?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_oblast_lookup() -> dict[str, int]:
    lookup: dict[str, int] = {}
    for uid, ukr_name in OBLAST_UIDS.items():
        lookup[_normalize_oblast_name(ukr_name)] = uid
    for uid, root in ENGLISH_OBLAST_ROOTS.items():
        lookup[_normalize_oblast_name(root)] = uid
        lookup[_normalize_oblast_name(f"{root} oblast")] = uid
    for variant, uid in SPECIAL_NAME_TO_UID.items():
        lookup[_normalize_oblast_name(variant)] = uid
    return lookup


OBLAST_NAME_LOOKUP = _build_oblast_lookup()


def match_oblast(name: str) -> tuple[int, str] | tuple[None, None]:
    """Resolve any known name variant (Ukrainian or English) to (uid, canonical_ukrainian_name)."""
    uid = OBLAST_NAME_LOOKUP.get(_normalize_oblast_name(name))
    if uid is None:
        return None, None
    return uid, OBLAST_UIDS[uid]


# ---------------------------------------------------------------------------
# "random" source -- synthetic data generator (unchanged from before)
# ---------------------------------------------------------------------------

ALERT_TYPES_WEIGHTED = [
    ("air_raid", 0.80),
    ("artillery_shelling", 0.10),
    ("urban_fights", 0.05),
    ("chemical", 0.02),
    ("nuclear", 0.03),
]

# Rough weighting so generated data isn't uniform across the country -- loosely
# mimics how real alert frequency skews toward front-line / frequently-targeted oblasts.
OBLAST_WEIGHTS = {
    "Харківська область": 9, "Сумська область": 8, "Дніпропетровська область": 7,
    "Запорізька область": 7, "Херсонська область": 6, "Донецька область": 9,
    "Миколаївська область": 6, "Одеська область": 5, "Київська область": 4,
    "Полтавська область": 3, "Чернігівська область": 4, "Луганська область": 5,
}
DEFAULT_WEIGHT = 1


def _weighted_oblast() -> tuple[int, str]:
    uids = list(OBLAST_UIDS.items())
    weights = [OBLAST_WEIGHTS.get(name, DEFAULT_WEIGHT) for _, name in uids]
    return random.choices(uids, weights=weights, k=1)[0]


def _weighted_alert_type() -> str:
    types, weights = zip(*ALERT_TYPES_WEIGHTED)
    return random.choices(types, weights=weights, k=1)[0]


def _hour_weights() -> list[float]:
    """Higher weight overnight/early morning, lower at midday -- loosely realistic."""
    weights = []
    for h in range(24):
        if 0 <= h < 6:
            weights.append(2.5)
        elif 6 <= h < 9 or 18 <= h < 23:
            weights.append(1.5)
        else:
            weights.append(1.0)
    return weights


def _random_duration_minutes() -> int:
    """Most alerts are short-to-medium; a few run long -- mimics real-world spread."""
    roll = random.random()
    if roll < 0.5:
        return random.randint(10, 45)      # short
    elif roll < 0.85:
        return random.randint(45, 180)     # medium
    return random.randint(180, 480)        # long, up to 8h


def generate_fake_alerts(days: int, avg_per_day: float, leave_active: int) -> list[Alert]:
    now = datetime.now(timezone.utc)
    hour_weights = _hour_weights()
    alerts: list[Alert] = []
    next_id = RANDOM_ID_BASE

    for day_offset in range(days, 0, -1):
        day_start = now - timedelta(days=day_offset)
        n_events = max(0, round(random.gauss(avg_per_day, avg_per_day * 0.3)))

        for _ in range(n_events):
            uid, name = _weighted_oblast()
            hour = random.choices(range(24), weights=hour_weights, k=1)[0]
            minute = random.randint(0, 59)
            started_at = day_start.replace(hour=hour, minute=minute, second=0, microsecond=0)
            duration = _random_duration_minutes()
            finished_at = started_at + timedelta(minutes=duration)

            alerts.append(
                Alert(
                    id=next_id,
                    location_title=name,
                    location_type="oblast",
                    location_uid=str(uid),
                    location_oblast=name,
                    location_oblast_uid=str(uid),
                    location_raion=None,
                    alert_type=_weighted_alert_type(),
                    started_at=started_at,
                    finished_at=finished_at,
                    updated_at=finished_at,
                    notes=FAKE_DATA_NOTE,
                    calculated=False,
                )
            )
            next_id += 1

    # Leave a handful of the most recent alerts "active" so the live panel has something to show.
    alerts.sort(key=lambda a: a.started_at)
    for a in (alerts[-leave_active:] if leave_active else []):
        a.finished_at = None
        a.updated_at = None

    return alerts


# ---------------------------------------------------------------------------
# "kaggle" source -- real historical data, supports two CSV shapes:
#   oblasts_only.csv:  region, started_at, finished_at, naive
#   full_data.csv:     oblast, raion, hromada, level, started_at, finished_at
# The loader auto-detects which one it's looking at based on whether a
# raion/hromada/level column is present.
# ---------------------------------------------------------------------------

REGION_COL_CANDIDATES = ["region", "oblast", "location_oblast", "name"]
START_COL_CANDIDATES = ["started_at", "start", "start_time", "started"]
END_COL_CANDIDATES = ["finished_at", "end", "end_time", "finished"]
NAIVE_COL_CANDIDATES = ["naive"]  # oblasts_only.csv: True = no end-of-alert message was ever received

FULL_DATA_OBLAST_COL_CANDIDATES = ["oblast"]    # always the parent oblast
FULL_DATA_RAION_COL_CANDIDATES = ["raion"]      # district -- maps straight onto location_raion
FULL_DATA_HROMADA_COL_CANDIDATES = ["hromada"]  # populated when level == 'hromada'
FULL_DATA_LEVEL_COL_CANDIDATES = ["level"]      # 'oblast' / 'hromada' / possibly others


def _find_column(columns, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _parse_bool_series(series: pd.Series) -> pd.Series:
    """Coerce a column that may be real bools or text ('True'/'False'/'1'/'0') into bool."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def _clean_str(value) -> str | None:
    """Treat NaN/'nan'/'none'/'' as 'no value' rather than literal text."""
    text = str(value).strip() if value is not None else ""
    return text if text and text.lower() not in ("nan", "none") else None


def _load_oblasts_only(df: pd.DataFrame, source_label: str) -> list[Alert]:
    """Parse the oblasts_only.csv shape: region, started_at, finished_at, naive."""
    region_col = _find_column(df.columns, REGION_COL_CANDIDATES)
    start_col = _find_column(df.columns, START_COL_CANDIDATES)
    end_col = _find_column(df.columns, END_COL_CANDIDATES)

    missing = [
        label for label, col in
        [("region", region_col), ("started_at", start_col), ("finished_at", end_col)]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not find expected column(s) {missing} in {source_label}.\n"
            f"Columns actually found: {list(df.columns)}\n"
            f"Update REGION_COL_CANDIDATES / START_COL_CANDIDATES / END_COL_CANDIDATES "
            f"at the top of generate_fake_data.py to match, then re-run."
        )

    started = pd.to_datetime(df[start_col], utc=True, errors="coerce")
    finished = pd.to_datetime(df[end_col], utc=True, errors="coerce")
    regions = df[region_col].astype(str).str.strip()

    naive_col = _find_column(df.columns, NAIVE_COL_CANDIDATES)
    if naive_col:
        naive_flags = _parse_bool_series(df[naive_col])
    else:
        naive_flags = pd.Series([False] * len(df))
        print("No 'naive' column found -- all rows will be saved with calculated=False.")

    alerts: list[Alert] = []
    next_id = KAGGLE_ID_BASE
    skipped_unknown_region = 0
    skipped_bad_start = 0

    for region, start_ts, end_ts, is_naive in zip(regions, started, finished, naive_flags):
        uid, canonical_name = match_oblast(region)
        if uid is None:
            skipped_unknown_region += 1
            continue
        if pd.isna(start_ts):
            skipped_bad_start += 1
            continue
        # Keep finished_at even when naive=True (it's the source's own start+30min estimate) --
        # nulling it out would make these 2022/2023 rows look "currently active" in the dashboard,
        # since that's how this schema represents an alert with no known end. `calculated=True`
        # is how we flag "this end time is an estimate, not a confirmed end-of-alert message",
        # matching the same field's meaning when it comes from the real alerts.in.ua API.
        finished_at = None if pd.isna(end_ts) else end_ts.to_pydatetime()

        alerts.append(
            Alert(
                id=next_id,
                location_title=canonical_name,
                location_type="oblast",
                location_uid=str(uid),
                location_oblast=canonical_name,
                location_oblast_uid=str(uid),
                location_raion=None,
                alert_type="air_raid",  # this dataset is specifically air-raid sirens
                started_at=start_ts.to_pydatetime(),
                finished_at=finished_at,
                updated_at=finished_at,
                notes=KAGGLE_DATA_NOTE,
                calculated=bool(is_naive),
            )
        )
        next_id += 1
        if next_id - KAGGLE_ID_BASE >= ID_RANGE_SIZE:
            raise RuntimeError(
                "Kaggle CSV has more rows than fit in the reserved id range. "
                "Increase ID_RANGE_SIZE in this script."
            )

    if skipped_unknown_region:
        print(
            f"Skipped {skipped_unknown_region} row(s): region name didn't match any known oblast "
            f"(Ukrainian or English) -- check the names against OBLAST_UIDS / ENGLISH_OBLAST_ROOTS."
        )
    if skipped_bad_start:
        print(f"Skipped {skipped_bad_start} row(s): unparseable start timestamp.")

    return alerts


def _load_full_data(df: pd.DataFrame, source_label: str) -> list[Alert]:
    """
    Parse the full_data.csv shape: oblast, raion, hromada, level, started_at, finished_at.

    'oblast' is always the parent oblast. 'level' says whether this row is oblast-wide or
    finer (hromada-level, etc). 'raion' maps directly onto location_raion (it already means
    the same thing there as it does in the live API). 'hromada', when present, is the most
    specific name available and becomes location_title -- this is what lets the dashboard
    show hromada-level detail (e.g. in the Currently Active panel) instead of only oblast names.
    """
    oblast_col = _find_column(df.columns, FULL_DATA_OBLAST_COL_CANDIDATES)
    raion_col = _find_column(df.columns, FULL_DATA_RAION_COL_CANDIDATES)
    hromada_col = _find_column(df.columns, FULL_DATA_HROMADA_COL_CANDIDATES)
    level_col = _find_column(df.columns, FULL_DATA_LEVEL_COL_CANDIDATES)
    start_col = _find_column(df.columns, START_COL_CANDIDATES)
    end_col = _find_column(df.columns, END_COL_CANDIDATES)

    missing = [
        label for label, col in
        [("oblast", oblast_col), ("started_at", start_col), ("finished_at", end_col)]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not find expected column(s) {missing} in {source_label}.\n"
            f"Columns actually found: {list(df.columns)}\n"
            f"Update the FULL_DATA_*_COL_CANDIDATES lists at the top of generate_fake_data.py "
            f"to match, then re-run."
        )

    oblasts_raw = df[oblast_col].astype(str).str.strip()
    raions = df[raion_col] if raion_col else pd.Series([None] * len(df))
    hromadas = df[hromada_col] if hromada_col else pd.Series([None] * len(df))
    levels = (
        df[level_col].astype(str).str.strip().str.lower() if level_col else pd.Series(["oblast"] * len(df))
    )
    started = pd.to_datetime(df[start_col], utc=True, errors="coerce")
    finished = pd.to_datetime(df[end_col], utc=True, errors="coerce")

    alerts: list[Alert] = []
    next_id = KAGGLE_ID_BASE
    skipped_unknown_oblast = 0
    skipped_bad_start = 0

    for oblast_raw, raion, hromada, level, start_ts, end_ts in zip(
        oblasts_raw, raions, hromadas, levels, started, finished
    ):
        uid, canonical_oblast = match_oblast(oblast_raw)
        if uid is None:
            skipped_unknown_oblast += 1
            continue
        if pd.isna(start_ts):
            skipped_bad_start += 1
            continue

        raion_clean = _clean_str(raion)
        hromada_clean = _clean_str(hromada)
        title = hromada_clean or raion_clean or canonical_oblast
        # Synthetic, locally-built identifier (not a real alerts.in.ua location uid) -- just
        # something stable and unique per distinct location in this dataset.
        sublocation = hromada_clean or raion_clean
        location_uid = f"{uid}-{sublocation}" if sublocation else str(uid)
        finished_at = None if pd.isna(end_ts) else end_ts.to_pydatetime()

        alerts.append(
            Alert(
                id=next_id,
                location_title=title,
                location_type=level or "oblast",
                location_uid=location_uid,
                location_oblast=canonical_oblast,
                location_oblast_uid=str(uid),
                location_raion=raion_clean,
                alert_type="air_raid",
                started_at=start_ts.to_pydatetime(),
                finished_at=finished_at,
                updated_at=finished_at,
                notes=KAGGLE_DATA_NOTE,
                calculated=False,  # full_data.csv doesn't carry a naive/estimated flag
            )
        )
        next_id += 1
        if next_id - KAGGLE_ID_BASE >= ID_RANGE_SIZE:
            raise RuntimeError(
                "Kaggle CSV has more rows than fit in the reserved id range. "
                "Increase ID_RANGE_SIZE in this script."
            )

    if skipped_unknown_oblast:
        print(
            f"Skipped {skipped_unknown_oblast} row(s): oblast name didn't match any known oblast "
            f"(Ukrainian or English) -- check the names against OBLAST_UIDS / ENGLISH_OBLAST_ROOTS."
        )
    if skipped_bad_start:
        print(f"Skipped {skipped_bad_start} row(s): unparseable start timestamp.")

    return alerts


def load_kaggle_alerts(csv_path: str) -> list[Alert]:
    """Load either CSV shape, auto-detected by the presence of a raion/hromada/level column."""
    df = pd.read_csv(csv_path)
    is_full_data = any(
        _find_column(df.columns, candidates) is not None
        for candidates in (FULL_DATA_RAION_COL_CANDIDATES, FULL_DATA_HROMADA_COL_CANDIDATES, FULL_DATA_LEVEL_COL_CANDIDATES)
    )
    if is_full_data:
        print(f"Detected full_data-style CSV (raion/hromada/level columns present) in {csv_path}")
        return _load_full_data(df, csv_path)
    print(f"Detected oblasts_only-style CSV in {csv_path}")
    return _load_oblasts_only(df, csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load non-live air alert data for local testing")
    sub = parser.add_subparsers(dest="source", required=True)

    random_p = sub.add_parser("random", help="Generate synthetic data locally (no files needed)")
    random_p.add_argument("--days", type=int, default=30, help="How many days of history to generate")
    random_p.add_argument("--avg-per-day", type=float, default=15, help="Average alerts per day")
    random_p.add_argument("--active", type=int, default=3, help="How many alerts to leave 'currently active'")
    random_p.add_argument("--clear", action="store_true", help="Delete previously generated synthetic rows first")
    random_p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible output")

    kaggle_p = sub.add_parser("kaggle", help="Load real historical data from the Kaggle CSV")
    kaggle_p.add_argument(
        "--csv", default="data/oblasts_only.csv",
        help="Path to oblasts_only.csv or full_data.csv (shape is auto-detected; default: data/oblasts_only.csv)",
    )
    kaggle_p.add_argument("--clear", action="store_true", help="Delete previously loaded Kaggle rows first")

    args = parser.parse_args()

    if args.source == "random":
        if args.seed is not None:
            random.seed(args.seed)
        if args.clear:
            _clear_id_range(RANDOM_ID_BASE, "synthetic")

        alerts = generate_fake_alerts(args.days, args.avg_per_day, args.active)
        con = get_connection()
        n = save_alerts(con, alerts)
        print(f"Generated and saved {n} synthetic alert record(s) across {args.days} days.")
        print(f"Tagged with notes='{FAKE_DATA_NOTE}' -- remove anytime with: random --clear")

    elif args.source == "kaggle":
        if args.clear:
            _clear_id_range(KAGGLE_ID_BASE, "Kaggle")

        alerts = load_kaggle_alerts(args.csv)
        if not alerts:
            print("No usable rows loaded -- nothing to save. See messages above for why rows were skipped.")
            return

        con = get_connection()
        n = save_alerts(con, alerts)
        print(f"Loaded and saved {n} historical alert record(s) from {args.csv}.")
        print(f"Tagged with notes='{KAGGLE_DATA_NOTE}' -- remove anytime with: kaggle --clear")


if __name__ == "__main__":
    main()