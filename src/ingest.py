"""
ingest.py
Fetches air raid alert data from the alerts.in.ua API and stores it in DuckDB.

Usage:
    python ingest.py live                   # continuously poll active alerts (default every 30s)
    python ingest.py live --interval 60      # custom poll interval
    python ingest.py backfill                # one-time historical backfill, all oblasts, last month
    python ingest.py backfill --uid 22       # backfill a single oblast by UID (22 = Kharkiv oblast)

Setup:
    pip install httpx duckdb pydantic python-dotenv tenacity loguru
    Get an API token at https://alerts.in.ua/api-request and put it in a .env file:
        ALERTS_API_TOKEN=your_token_here

API docs: https://devs.alerts.in.ua/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import duckdb
import httpx
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

API_BASE = "https://api.alerts.in.ua/v1"
API_TOKEN = os.getenv("ALERTS_API_TOKEN")
DB_PATH = os.getenv("ALERTS_DB_PATH", "alerts.duckdb")

# active.json soft limit is ~8-10 req/min from one IP -- 30s polling is well under that.
POLL_INTERVAL_SECONDS = 30
# /regions/{uid}/alerts/{period}.json has its own hard limit of 2 req/min.
HISTORY_REQUEST_DELAY_SECONDS = 35

# Oblast UIDs as documented by the API, used for the backfill command.
OBLAST_UIDS = {
    3: "Хмельницька область", 4: "Вінницька область", 5: "Рівненська область",
    8: "Волинська область", 9: "Дніпропетровська область", 10: "Житомирська область",
    11: "Закарпатська область", 12: "Запорізька область", 13: "Івано-Франківська область",
    14: "Київська область", 15: "Кіровоградська область", 16: "Луганська область",
    17: "Миколаївська область", 18: "Одеська область", 19: "Полтавська область",
    20: "Сумська область", 21: "Тернопільська область", 22: "Харківська область",
    23: "Херсонська область", 24: "Черкаська область", 25: "Чернігівська область",
    26: "Чернівецька область", 27: "Львівська область", 28: "Донецька область",
    29: "Автономна Республіка Крим", 30: "м. Севастополь", 31: "м. Київ",
}


class Alert(BaseModel):
    """Validated representation of a single alert record from the API."""

    model_config = ConfigDict(extra="ignore")  # tolerate fields the API adds later

    id: int
    location_title: str
    location_type: str
    location_uid: str
    location_oblast: Optional[str] = None
    location_oblast_uid: Optional[str] = None
    location_raion: Optional[str] = None
    alert_type: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    notes: Optional[str] = ""
    calculated: bool = False

    @field_validator("location_uid", "location_oblast_uid", mode="before")
    @classmethod
    def _stringify_uid(cls, v):
        return str(v) if v is not None else v


def get_connection() -> duckdb.DuckDBPyConnection:
    """Open (or create) the DuckDB file and ensure the schema exists."""
    con = duckdb.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id                  BIGINT PRIMARY KEY,
            location_uid        VARCHAR,
            location_title      VARCHAR,
            location_type       VARCHAR,
            location_oblast     VARCHAR,
            location_oblast_uid VARCHAR,
            location_raion      VARCHAR,
            alert_type          VARCHAR,
            started_at          TIMESTAMP,
            finished_at         TIMESTAMP,
            updated_at          TIMESTAMP,
            notes               VARCHAR,
            calculated          BOOLEAN,
            ingested_at         TIMESTAMP DEFAULT now()
        )
        """
    )
    return con


def save_alerts(con: duckdb.DuckDBPyConnection, alerts: list[Alert]) -> int:
    """Upsert a batch of alerts (insert new, update finished_at/notes on existing ids)."""
    if not alerts:
        return 0
    rows = [
        (
            a.id, a.location_uid, a.location_title, a.location_type,
            a.location_oblast, a.location_oblast_uid, a.location_raion,
            a.alert_type, a.started_at, a.finished_at, a.updated_at,
            a.notes, a.calculated,
        )
        for a in alerts
    ]
    con.executemany(
        """
        INSERT INTO alerts (
            id, location_uid, location_title, location_type,
            location_oblast, location_oblast_uid, location_raion,
            alert_type, started_at, finished_at, updated_at, notes, calculated
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            finished_at = excluded.finished_at,
            updated_at  = excluded.updated_at,
            notes       = excluded.notes,
            calculated  = excluded.calculated
        """,
        rows,
    )
    return len(rows)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    reraise=True,
)
def _get(client: httpx.Client, path: str) -> dict:
    """GET a path from the alerts.in.ua API, retrying on transient/rate-limit failures."""
    if not API_TOKEN:
        raise RuntimeError("ALERTS_API_TOKEN is not set. Add it to your .env file.")
    resp = client.get(f"{API_BASE}{path}", headers={"Authorization": f"Bearer {API_TOKEN}"})
    if resp.status_code == 429:
        logger.warning("Rate limited by API, backing off before retry")
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()


def fetch_active_alerts(client: httpx.Client) -> list[Alert]:
    data = _get(client, "/alerts/active.json")
    return [Alert.model_validate(a) for a in data.get("alerts", [])]


def fetch_region_history(client: httpx.Client, uid: int, period: str = "month_ago") -> list[Alert]:
    data = _get(client, f"/regions/{uid}/alerts/{period}.json")
    return [Alert.model_validate(a) for a in data.get("alerts", [])]


def run_live(poll_interval: int = POLL_INTERVAL_SECONDS) -> None:
    """Continuously poll the active-alerts endpoint and persist results. Ctrl+C to stop."""
    con = get_connection()
    logger.info(f"Starting live ingestion -> {DB_PATH} (every {poll_interval}s)")
    with httpx.Client(timeout=15) as client:
        while True:
            try:
                alerts = fetch_active_alerts(client)
                n = save_alerts(con, alerts)
                logger.info(f"Synced {n} active alert record(s)")
            except Exception as exc:
                logger.error(f"Ingestion cycle failed: {exc}")
            time.sleep(poll_interval)


def run_backfill(period: str = "month_ago", uid: Optional[int] = None) -> None:
    """
    One-time historical backfill via /regions/{uid}/alerts/{period}.json.
    Currently 'month_ago' is the only period the API documents.
    Paces requests to respect that endpoint's 2 req/min hard limit.
    """
    con = get_connection()
    targets = {uid: OBLAST_UIDS[uid]} if uid else OBLAST_UIDS
    logger.info(f"Backfilling {len(targets)} oblast(s) for period '{period}'")
    with httpx.Client(timeout=20) as client:
        for i, (region_uid, name) in enumerate(targets.items()):
            try:
                alerts = fetch_region_history(client, region_uid, period)
                n = save_alerts(con, alerts)
                logger.info(f"[{name}] saved {n} historical record(s)")
            except Exception as exc:
                logger.error(f"[{name}] backfill failed: {exc}")
            if i < len(targets) - 1:
                time.sleep(HISTORY_REQUEST_DELAY_SECONDS)
    logger.info("Backfill complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest air alert data from alerts.in.ua")
    sub = parser.add_subparsers(dest="command", required=True)

    live_p = sub.add_parser("live", help="Continuously poll active alerts")
    live_p.add_argument("--interval", type=int, default=POLL_INTERVAL_SECONDS)

    backfill_p = sub.add_parser("backfill", help="One-time historical backfill")
    backfill_p.add_argument("--period", default="month_ago")
    backfill_p.add_argument("--uid", type=int, default=None, help="Backfill a single oblast UID only")

    args = parser.parse_args()

    try:
        if args.command == "live":
            run_live(args.interval)
        elif args.command == "backfill":
            run_backfill(args.period, args.uid)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()