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

import json
import os

import dash
import dash_bootstrap_components as dbc
import duckdb
import httpx
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("ALERTS_DB_PATH", "alerts.duckdb")
REFRESH_INTERVAL_MS = 30_000  # how often the charts/active panel refresh

# Duration buckets for the "alert duration" filter on the Daily Alerts chart.
# Bounds are in minutes; max=None means "this bucket and longer".
DURATION_BUCKETS = [
    {"value": "ongoing", "label": "Still ongoing", "ongoing": True},
    {"value": "lt30", "label": "< 30 min", "min": 0, "max": 30},
    {"value": "30to60", "label": "30–60 min", "min": 30, "max": 60},
    {"value": "1to3h", "label": "1–3 hrs", "min": 60, "max": 180},
    {"value": "3to6h", "label": "3–6 hrs", "min": 180, "max": 360},
    {"value": "6hplus", "label": "6+ hrs", "min": 360, "max": None},
]
DURATION_BUCKET_BY_VALUE = {b["value"]: b for b in DURATION_BUCKETS}

# Oblast boundaries for the map, from https://github.com/EugeneBorshch/ukraine_geojson
# Its "name" property (e.g. "Харківська область") matches location_oblast exactly,
# so no name-mapping table is needed -- downloaded once and cached to disk.
UKRAINE_GEOJSON_URL = (
    "https://raw.githubusercontent.com/EugeneBorshch/ukraine_geojson/master/UA_FULL_Ukraine.geojson"
)
UKRAINE_GEOJSON_CACHE = os.getenv("UKRAINE_GEOJSON_CACHE", "ukraine_oblasts.geojson")


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


def _duration_filter_clause(selected: list[str] | None) -> tuple[str, list]:
    """
    Build a SQL fragment (with a leading ' AND (...)') plus its bind params for the
    selected duration buckets. Bucket boundaries come only from DURATION_BUCKET_BY_VALUE
    (a fixed constant), never raw user input, so this is safe to splice into SQL.
    """
    if not selected:
        return "", []

    clauses: list[str] = []
    params: list = []
    for value in selected:
        bucket = DURATION_BUCKET_BY_VALUE.get(value)
        if bucket is None:
            continue
        if bucket.get("ongoing"):
            clauses.append("finished_at IS NULL")
            continue
        min_minutes, max_minutes = bucket["min"], bucket["max"]
        if max_minutes is None:
            clauses.append(
                "(finished_at IS NOT NULL AND date_diff('minute', started_at, finished_at) >= ?)"
            )
            params.append(min_minutes)
        else:
            clauses.append(
                "(finished_at IS NOT NULL AND date_diff('minute', started_at, finished_at) >= ? "
                "AND date_diff('minute', started_at, finished_at) < ?)"
            )
            params.extend([min_minutes, max_minutes])

    if not clauses:
        return "", []
    return " AND (" + " OR ".join(clauses) + ")", params


def get_filtered_daily_counts(
    oblasts: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    duration_values: list[str] | None,
) -> pd.DataFrame:
    """Alerts started per day, per oblast -- filtered by oblast, date range, and duration bucket."""
    where = ["1=1"]
    params: list = []

    if oblasts:
        placeholders = ", ".join(["?"] * len(oblasts))
        where.append(f"coalesce(location_oblast, location_title) IN ({placeholders})")
        params.extend(oblasts)

    if start_date:
        where.append("CAST(started_at AS DATE) >= CAST(? AS DATE)")
        params.append(start_date)
    if end_date:
        where.append("CAST(started_at AS DATE) <= CAST(? AS DATE)")
        params.append(end_date)

    duration_clause, duration_params = _duration_filter_clause(duration_values)

    sql = f"""
        SELECT
            date_trunc('day', started_at) AS day,
            coalesce(location_oblast, location_title) AS oblast,
            count(*) AS alert_count
        FROM alerts
        WHERE {' AND '.join(where)}{duration_clause}
        GROUP BY 1, 2
        ORDER BY 1
    """
    return query_df(sql, params + duration_params)


def load_ukraine_geojson() -> dict | None:
    """
    Load oblast boundaries for the map, caching to disk after the first download so the
    app doesn't need network access on every restart. Returns None (rather than raising)
    if the download fails, so a network hiccup degrades the map gracefully instead of
    crashing the whole app.
    """
    if os.path.exists(UKRAINE_GEOJSON_CACHE):
        try:
            with open(UKRAINE_GEOJSON_CACHE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # cache file is corrupt/unreadable -- fall through and re-download

    try:
        resp = httpx.get(UKRAINE_GEOJSON_URL, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
        with open(UKRAINE_GEOJSON_CACHE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    except Exception as exc:
        print(f"[ukraine-map] Could not load oblast boundaries: {exc}")
        return None


def get_available_months() -> list[str]:
    """Distinct 'YYYY-MM' months present in the data, oldest first -- drives the map's slider."""
    df = query_df("SELECT DISTINCT strftime(started_at, '%Y-%m') AS ym FROM alerts ORDER BY 1")
    return df["ym"].tolist() if not df.empty else []


def get_monthly_oblast_counts(year_month: str | None) -> pd.DataFrame:
    """Total alerts per oblast for the given 'YYYY-MM' month (all data if year_month is None)."""
    if year_month:
        where, params = "strftime(started_at, '%Y-%m') = ?", [year_month]
    else:
        where, params = "1=1", []

    sql = f"""
        SELECT
            coalesce(location_oblast, location_title) AS oblast,
            count(*) AS alert_count
        FROM alerts
        WHERE {where}
        GROUP BY 1
    """
    return query_df(sql, params)


def get_oblast_options() -> list[str]:
    """Distinct oblast names currently in the data, for the oblast filter dropdown."""
    df = query_df(
        "SELECT DISTINCT coalesce(location_oblast, location_title) AS oblast FROM alerts ORDER BY 1"
    )
    return df["oblast"].tolist() if not df.empty else []


def get_date_bounds() -> tuple[str | None, str | None]:
    """Earliest/latest alert dates in the data, for the date-range picker's bounds."""
    df = query_df(
        "SELECT min(CAST(started_at AS DATE)) AS min_d, max(CAST(started_at AS DATE)) AS max_d FROM alerts"
    )
    if df.empty or pd.isna(df.loc[0, "min_d"]):
        return None, None
    return str(df.loc[0, "min_d"]), str(df.loc[0, "max_d"])


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


def get_oblast_total_hours() -> pd.DataFrame:
    """
    Total cumulative TIME spent under alert per oblast, in hours -- only counts completed
    alerts (a still-active one has no duration yet). This is a different ranking than
    get_oblast_totals(): an oblast can have fewer, longer alerts and still top this chart.
    """
    return query_df(
        """
        SELECT
            coalesce(location_oblast, location_title) AS oblast,
            sum(date_diff('minute', started_at, finished_at)) / 60.0 AS total_hours
        FROM alerts
        WHERE finished_at IS NOT NULL
          AND finished_at > started_at
        GROUP BY 1
        ORDER BY 2 DESC
        """
    )


def get_active_alerts() -> pd.DataFrame:
    """Alerts currently in progress (finished_at is still null)."""
    return query_df(
        """
        SELECT location_title, location_oblast, location_raion, location_type, alert_type, started_at
        FROM alerts
        WHERE finished_at IS NULL
        ORDER BY started_at DESC
        """
    )


def get_national_daily_counts() -> pd.DataFrame:
    """Total alerts (all oblasts combined) started per day -- the input for the trend chart."""
    return query_df(
        """
        SELECT
            date_trunc('day', started_at) AS day,
            count(*) AS alert_count
        FROM alerts
        GROUP BY 1
        ORDER BY 1
        """
    )


def get_alert_durations() -> pd.DataFrame:
    """
    Duration (in minutes) of every completed alert, with its type.
    Only finished alerts are included -- a still-active alert has no duration yet.
    """
    return query_df(
        """
        SELECT
            alert_type,
            date_diff('minute', started_at, finished_at) AS duration_minutes
        FROM alerts
        WHERE finished_at IS NOT NULL
          AND finished_at > started_at
        """
    )


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Ukraine Air Alerts"
server = app.server  # exposed for production WSGI servers (gunicorn app:server)

# Filter option values, fetched once at startup. Good enough for a hackathon scope --
# if the set of oblasts or the date range changes meaningfully while the app is running,
# restart the process to pick up the new values.
OBLAST_OPTIONS = get_oblast_options()
MIN_DATE, MAX_DATE = get_date_bounds()
AVAILABLE_MONTHS = get_available_months()
UKRAINE_GEOJSON = load_ukraine_geojson()

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Interval(id="refresh-tick", interval=REFRESH_INTERVAL_MS, n_intervals=0),
        dbc.Row(dbc.Col(html.H2("Air Alerts in Ukraine — Time Series", className="my-3"))),
        dbc.Row(
            dbc.Col(
                [
                    html.H5("Alerts by Oblast — Monthly Map", className="mb-2"),
                    dcc.Graph(id="ukraine-map", style={"height": "520px"}),
                    html.Div(
                        dcc.Slider(
                            id="month-slider",
                            min=0,
                            max=max(len(AVAILABLE_MONTHS) - 1, 0),
                            step=1,
                            value=max(len(AVAILABLE_MONTHS) - 1, 0),
                            marks={i: m for i, m in enumerate(AVAILABLE_MONTHS)},
                        ),
                        className="px-4 pt-2 pb-1",
                    ),
                ],
                width=12,
            ),
            className="mb-4",
        ),
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
                dbc.Col(dcc.Graph(id="total-count-chart"), width=6),
                dbc.Col(dcc.Graph(id="total-duration-chart"), width=6),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Oblast", className="text-light small mb-1"),
                        dcc.Dropdown(
                            id="oblast-filter",
                            options=[{"label": o, "value": o} for o in OBLAST_OPTIONS],
                            multi=True,
                            placeholder="All oblasts",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("Date range", className="text-light small mb-1"),
                        dcc.DatePickerRange(
                            id="date-filter",
                            min_date_allowed=MIN_DATE,
                            max_date_allowed=MAX_DATE,
                            start_date=MIN_DATE,
                            end_date=MAX_DATE,
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("Alert duration", className="text-light small mb-1"),
                        dcc.Dropdown(
                            id="duration-filter",
                            options=[{"label": b["label"], "value": b["value"]} for b in DURATION_BUCKETS],
                            multi=True,
                            placeholder="All durations",
                        ),
                    ],
                    width=4,
                ),
            ],
            className="mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="daily-trend-chart"), width=8),
                dbc.Col(dcc.Graph(id="oblast-totals-chart"), width=4),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="escalation-trend-chart"), width=8),
                dbc.Col(html.Div(id="trend-badge"), width=4, className="d-flex align-items-center justify-content-center"),
            ],
            className="mt-3",
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id="duration-histogram"), width=12),
            className="mt-3",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("ukraine-map", "figure"),
    Input("month-slider", "value"),
)
def update_ukraine_map(month_index: int):
    if UKRAINE_GEOJSON is None:
        return px.bar(title="Map unavailable — failed to download oblast boundaries (check network)")
    if not AVAILABLE_MONTHS:
        return px.choropleth(title="No data yet — run ingest.py first")

    year_month = AVAILABLE_MONTHS[month_index] if month_index is not None else AVAILABLE_MONTHS[-1]
    df = get_monthly_oblast_counts(year_month)
    if df.empty:
        return px.choropleth(title=f"No alerts recorded for {year_month}")

    fig = px.choropleth(
        df,
        geojson=UKRAINE_GEOJSON,
        locations="oblast",
        featureidkey="properties.name",
        color="alert_count",
        color_continuous_scale="OrRd",
        labels={"alert_count": "Alerts"},
        title=f"Alerts by Oblast — {year_month}",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _active_alert_item(row) -> html.Li:
    """
    Render one active-alert row, showing hromada/raion-level detail when present
    (location_raion is only populated for finer-than-oblast rows -- e.g. from a
    full_data.csv Kaggle load) instead of just the oblast name.
    """
    if row.location_raion:
        where = f"{row.location_title} ({row.location_oblast})" if row.location_oblast else row.location_title
    else:
        where = row.location_title

    granularity = (row.location_type or "oblast").strip().lower() or "oblast"
    badge = dbc.Badge(granularity, color="secondary", className="ms-2")

    return html.Li([f"{where} — {row.alert_type} (since {row.started_at:%H:%M, %d %b})", badge])


@app.callback(
    Output("active-alerts-panel", "children"),
    Input("refresh-tick", "n_intervals"),
)
def update_active_alerts(_n: int):
    df = get_active_alerts()
    if df.empty:
        return dbc.Alert("No active alerts right now.", color="success")

    items = [_active_alert_item(row) for row in df.itertuples()]
    return [
        dbc.Alert(f"{len(df)} location(s) currently under alert", color="danger"),
        html.Ul(items),
    ]


@app.callback(
    Output("total-count-chart", "figure"),
    Input("refresh-tick", "n_intervals"),
)
def update_total_count_chart(_n: int):
    df = get_oblast_totals()
    if df.empty:
        return px.bar(title="No data yet")

    fig = px.bar(
        df.head(15),
        x="alert_count",
        y="oblast",
        orientation="h",
        title="Total Alerts by Oblast — Quantity",
        labels={"alert_count": "Number of Alerts", "oblast": ""},
    )
    fig.update_layout(template="plotly_dark", yaxis=dict(categoryorder="total ascending"))
    return fig


@app.callback(
    Output("total-duration-chart", "figure"),
    Input("refresh-tick", "n_intervals"),
)
def update_total_duration_chart(_n: int):
    df = get_oblast_total_hours()
    if df.empty:
        return px.bar(title="No completed alerts yet")

    fig = px.bar(
        df.head(15),
        x="total_hours",
        y="oblast",
        orientation="h",
        title="Total Alerts by Oblast — Cumulative Time",
        labels={"total_hours": "Total Hours Under Alert", "oblast": ""},
    )
    fig.update_layout(template="plotly_dark", yaxis=dict(categoryorder="total ascending"))
    return fig


@app.callback(
    Output("daily-trend-chart", "figure"),
    Input("refresh-tick", "n_intervals"),
    Input("oblast-filter", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("duration-filter", "value"),
)
def update_daily_trend(_n: int, oblasts, start_date, end_date, duration_values):
    df = get_filtered_daily_counts(oblasts, start_date, end_date, duration_values)
    if df.empty:
        return px.line(title="No alerts match the current filters")

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


@app.callback(
    Output("escalation-trend-chart", "figure"),
    Output("trend-badge", "children"),
    Input("refresh-tick", "n_intervals"),
)
def update_escalation_trend(_n: int):
    df = get_national_daily_counts()
    if df.empty or len(df) < 2:
        return px.line(title="No data yet — run ingest.py first"), ""

    df = df.sort_values("day").reset_index(drop=True)
    df["rolling_avg"] = df["alert_count"].rolling(window=7, min_periods=1).mean()

    fig = px.bar(df, x="day", y="alert_count", labels={"day": "Date", "alert_count": "Alerts"})
    fig.update_traces(marker_color="rgba(108, 142, 191, 0.35)", name="Daily alerts", showlegend=True)
    fig.add_scatter(
        x=df["day"],
        y=df["rolling_avg"],
        mode="lines",
        name="7-day rolling average",
        line=dict(width=3),
    )
    fig.update_layout(
        template="plotly_dark",
        title="Escalation / De-escalation Trend (nationwide)",
        legend=dict(orientation="h", y=-0.25),
    )

    # Compare the most recent 7-day window against the 7 days before it to
    # give a simple, glanceable read on whether things are trending up or down.
    last_7 = df["alert_count"].tail(7).mean()
    prev_7 = df["alert_count"].iloc[-14:-7].mean() if len(df) >= 14 else None

    if prev_7 is None or prev_7 == 0:
        badge = dbc.Alert("Not enough history yet for a trend comparison.", color="secondary")
    else:
        pct_change = (last_7 - prev_7) / prev_7 * 100
        if pct_change > 10:
            badge = dbc.Alert(f"⬆ Escalating: +{pct_change:.0f}% vs. prior 7 days", color="danger")
        elif pct_change < -10:
            badge = dbc.Alert(f"⬇ De-escalating: {pct_change:.0f}% vs. prior 7 days", color="success")
        else:
            badge = dbc.Alert(f"→ Roughly stable: {pct_change:+.0f}% vs. prior 7 days", color="warning")

    return fig, badge


@app.callback(
    Output("duration-histogram", "figure"),
    Input("refresh-tick", "n_intervals"),
)
def update_duration_histogram(_n: int):
    df = get_alert_durations()
    if df.empty:
        return px.histogram(title="No completed alerts yet — durations appear once alerts end")

    median_minutes = df["duration_minutes"].median()

    fig = px.histogram(
        df,
        x="duration_minutes",
        color="alert_type",
        nbins=40,
        title="Alert Duration Distribution",
        labels={"duration_minutes": "Duration (minutes)", "count": "Number of alerts"},
        opacity=0.85,
    )
    fig.add_vline(
        x=median_minutes,
        line_dash="dash",
        annotation_text=f"median: {median_minutes:.0f} min",
        annotation_position="top",
    )
    fig.update_layout(template="plotly_dark", bargap=0.05, legend=dict(orientation="h", y=-0.25))
    return fig


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)