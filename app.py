"""
app.py
A simple Dash application visualizing air alert data collected by ingest.py.

Run:
    uv run python app.py
    -> http://localhost:8050

Reads (read-only) from the same DuckDB file ingest.py writes to.
For production (e.g. gunicorn), the WSGI callable is exposed as `server` below:
    gunicorn app:server
"""

from __future__ import annotations

import os

import dash
import dash_bootstrap_components as dbc
import duckdb
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("ALERTS_DB_PATH", "alerts.duckdb")
REFRESH_INTERVAL_MS = 30_000  # how often the charts/active panel refresh


def ensure_db_exists() -> None:
    """
    Create the DuckDB file and schema if they don't exist yet.

    duckdb.connect(path, read_only=True) errors out if the file is missing --
    it only ever opens, never creates. ingest.py normally creates it first,
    but the app shouldn't hard-depend on run order (fresh clone, reset demo
    data, etc.), so we make sure it exists before we ever try a read-only query.
    """
    if os.path.exists(DB_PATH):
        return
    con = duckdb.connect(DB_PATH)  # write mode here is fine -- file doesn't exist yet
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
    con.close()


def query_df(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a read-only query against the DuckDB file and return a DataFrame."""
    ensure_db_exists()
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def get_daily_counts() -> pd.DataFrame:
    """Number of alerts started per day, per oblast."""
    return query_df(
        """
        SELECT
            date_trunc('day', started_at) AS day,
            coalesce(location_oblast, location_title) AS oblast,
            count(*) AS alert_count
        FROM alerts
        GROUP BY 1, 2
        ORDER BY 1
        """
    )


def get_oblast_totals() -> pd.DataFrame:
    """Total alert count per oblast across all data currently stored."""
    return query_df(
        """
        SELECT
            coalesce(location_oblast, location_title) AS oblast,
            count(*) AS alert_count
        FROM alerts
        GROUP BY 1
        ORDER BY 2 DESC
        """
    )


def get_active_alerts() -> pd.DataFrame:
    """Alerts currently in progress (finished_at is still null)."""
    return query_df(
        """
        SELECT location_title, alert_type, started_at
        FROM alerts
        WHERE finished_at IS NULL
        ORDER BY started_at DESC
        """
    )


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Ukraine Air Alerts"
server = app.server  # exposed for production WSGI servers (gunicorn app:server)

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Interval(id="refresh-tick", interval=REFRESH_INTERVAL_MS, n_intervals=0),
        dbc.Row(dbc.Col(html.H2("Air Alerts in Ukraine — Time Series", className="my-3"))),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Currently Active", className="card-title"),
                            html.Div(id="active-alerts-panel"),
                        ]
                    ),
                    color="dark",
                    inverse=True,
                ),
                width=12,
            ),
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="daily-trend-chart"), width=8),
                dbc.Col(dcc.Graph(id="oblast-totals-chart"), width=4),
            ]
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("active-alerts-panel", "children"),
    Input("refresh-tick", "n_intervals"),
)
def update_active_alerts(_n: int):
    df = get_active_alerts()
    if df.empty:
        return dbc.Alert("No active alerts right now.", color="success")

    items = [
        html.Li(f"{row.location_title} — {row.alert_type} (since {row.started_at:%H:%M, %d %b})")
        for row in df.itertuples()
    ]
    return [
        dbc.Alert(f"{len(df)} region(s) currently under alert", color="danger"),
        html.Ul(items),
    ]


@app.callback(
    Output("daily-trend-chart", "figure"),
    Input("refresh-tick", "n_intervals"),
)
def update_daily_trend(_n: int):
    df = get_daily_counts()
    if df.empty:
        return px.line(title="No data yet — run ingest.py first")

    fig = px.line(
        df,
        x="day",
        y="alert_count",
        color="oblast",
        title="Daily Alerts by Oblast",
        labels={"day": "Date", "alert_count": "Alerts", "oblast": "Oblast"},
    )
    fig.update_layout(template="plotly_dark", legend=dict(orientation="h", y=-0.3))
    return fig


@app.callback(
    Output("oblast-totals-chart", "figure"),
    Input("refresh-tick", "n_intervals"),
)
def update_oblast_totals(_n: int):
    df = get_oblast_totals()
    if df.empty:
        return px.bar(title="No data yet")

    fig = px.bar(
        df.head(15),
        x="alert_count",
        y="oblast",
        orientation="h",
        title="Total Alerts by Oblast (Top 15)",
        labels={"alert_count": "Total Alerts", "oblast": ""},
    )
    fig.update_layout(template="plotly_dark", yaxis=dict(categoryorder="total ascending"))
    return fig


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)