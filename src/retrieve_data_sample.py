"""
retrieve_data_sample.py
Loads the real historical air-alert dataset (oblast + raion + hromada level) from the
Kaggle "Air-raid sirens in Ukraine" export (cashncarry/airraid-sirens-in-ukraine,
full_data.csv) into the local DuckDB database.

This is the data source used for the submitted demo: it requires no ALERTS_API_TOKEN and
no network access, and -- unlike synthetic data -- it reflects real alert activity, which
is what a time-series dashboard like this one should actually be evaluated against.

Run:
    uv run python src/retrieve_data_sample.py                      # loads data/full_data.csv
    uv run python src/retrieve_data_sample.py --csv path/to.csv    # load a different export
    uv run python src/retrieve_data_sample.py --clear              # wipe previously loaded rows first

Source:
    https://www.kaggle.com/datasets/cashncarry/airraid-sirens-in-ukraine

Known data-quality caveat: a handful of hromada names in this export are not unique across
oblasts (e.g. "Kostiantynivska terytorialna hromada" appears under four different oblasts
with identical timestamps). That looks like a name-collision artifact in the source data
rather than four simultaneous real alerts -- worth deduping on
(started_at, finished_at, location_title) before building anything (e.g. a region filter)
that assumes hromada names are unique.
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from ingest import Alert, OBLAST_UIDS, get_connection, save_alerts

# ---------------------------------------------------------------------------
# Reserved id range for this source, so its rows never collide with real
# alerts.in.ua ids (used by ingest.py) and can be cleared independently.
# ---------------------------------------------------------------------------
DATA_ID_BASE = 800_000_000
ID_RANGE_SIZE = 100_000_000

HISTORICAL_DATA_NOTE = "KAGGLE_HISTORICAL_DATA"


def _clear_previous_load() -> None:
    """Delete rows previously loaded by this script (id in the reserved range)."""
    con = get_connection()
    upper = DATA_ID_BASE + ID_RANGE_SIZE
    count = con.execute(
        "SELECT count(*) FROM alerts WHERE id >= ? AND id < ?", [DATA_ID_BASE, upper]
    ).fetchone()[0]
    con.execute("DELETE FROM alerts WHERE id >= ? AND id < ?", [DATA_ID_BASE, upper])
    print(f"Cleared {count} previously loaded historical record(s).")


# ---------------------------------------------------------------------------
# Oblast name matching, language-agnostic.
#
# The Kaggle dataset uses English transliteration ("Vinnytska oblast"), while the live API
# and OBLAST_UIDS use Ukrainian ("Вінницька область"). Every row gets matched here and
# stored under the SAME canonical Ukrainian name OBLAST_UIDS already uses, regardless of
# which language the source used -- this matters because the choropleth map joins on that
# exact Ukrainian name against the GeoJSON, so an English name slipping into
# location_oblast would silently break the map.
# ---------------------------------------------------------------------------

ENGLISH_OBLAST_ROOTS = {
    3: "Khmelnytska", 4: "Vinnytska", 5: "Rivnenska", 8: "Volynska",
    9: "Dnipropetrovska", 10: "Zhytomyrska", 11: "Zakarpatska", 12: "Zaporizka",
    13: "Ivano-Frankivska", 14: "Kyivska", 15: "Kirovohradska", 16: "Luhanska",
    17: "Mykolaivska", 18: "Odeska", 19: "Poltavska", 20: "Sumska",
    21: "Ternopilska", 22: "Kharkivska", 23: "Khersonska", 24: "Cherkaska",
    25: "Chernihivska", 26: "Chernivetska", 27: "Lvivska", 28: "Donetska",
}
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
# CSV parsing: oblast, raion, hromada, level, started_at, finished_at
# ---------------------------------------------------------------------------

OBLAST_COL_CANDIDATES = ["oblast", "region", "location_oblast"]
RAION_COL_CANDIDATES = ["raion"]
HROMADA_COL_CANDIDATES = ["hromada"]
LEVEL_COL_CANDIDATES = ["level"]
START_COL_CANDIDATES = ["started_at", "start", "start_time", "started"]
END_COL_CANDIDATES = ["finished_at", "end", "end_time", "finished"]


def _find_column(columns, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _clean_str(value) -> str | None:
    """Treat NaN/'nan'/'none'/'' as 'no value' rather than literal text."""
    text = str(value).strip() if value is not None else ""
    return text if text and text.lower() not in ("nan", "none") else None


def load_historical_alerts(csv_path: str) -> list[Alert]:
    """
    Parse the Kaggle full_data.csv export into a list of Alert records.

    'oblast' is always the parent oblast. 'level' says whether a row is oblast-wide or
    finer (hromada-level, etc). 'raion' maps directly onto location_raion. 'hromada', when
    present, is the most specific name available and becomes location_title -- this is what
    lets the dashboard show hromada-level detail (e.g. in the Currently Active panel)
    instead of only oblast names.
    """
    df = pd.read_csv(csv_path)

    oblast_col = _find_column(df.columns, OBLAST_COL_CANDIDATES)
    raion_col = _find_column(df.columns, RAION_COL_CANDIDATES)
    hromada_col = _find_column(df.columns, HROMADA_COL_CANDIDATES)
    level_col = _find_column(df.columns, LEVEL_COL_CANDIDATES)
    start_col = _find_column(df.columns, START_COL_CANDIDATES)
    end_col = _find_column(df.columns, END_COL_CANDIDATES)

    missing = [
        label for label, col in
        [("oblast", oblast_col), ("started_at", start_col), ("finished_at", end_col)]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not find expected column(s) {missing} in {csv_path}.\n"
            f"Columns actually found: {list(df.columns)}\n"
            f"Update the *_COL_CANDIDATES lists at the top of retrieve_data_sample.py "
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
    next_id = DATA_ID_BASE
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
                alert_type="air_raid",  # this dataset is specifically air-raid sirens
                started_at=start_ts.to_pydatetime(),
                finished_at=finished_at,
                updated_at=finished_at,
                notes=HISTORICAL_DATA_NOTE,
                calculated=False,  # this export doesn't carry a naive/estimated-end flag
            )
        )
        next_id += 1
        if next_id - DATA_ID_BASE >= ID_RANGE_SIZE:
            raise RuntimeError(
                "CSV has more rows than fit in the reserved id range. "
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the real historical air-alert data sample into DuckDB"
    )
    parser.add_argument(
        "--csv", default="data/full_data.csv",
        help="Path to the historical data CSV (default: data/full_data.csv)",
    )
    parser.add_argument("--clear", action="store_true", help="Delete previously loaded rows first")
    args = parser.parse_args()

    if args.clear:
        _clear_previous_load()

    alerts = load_historical_alerts(args.csv)
    if not alerts:
        print("No usable rows loaded -- nothing to save. See messages above for why rows were skipped.")
        return

    con = get_connection()
    n = save_alerts(con, alerts)
    print(f"Loaded and saved {n} historical alert record(s) from {args.csv}.")
    print(f"Tagged with notes='{HISTORICAL_DATA_NOTE}' -- remove anytime with: --clear")


if __name__ == "__main__":
    main()