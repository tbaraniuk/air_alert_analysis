# Air Alerts in Ukraine — Time Series Dashboard

A dashboard that visualizes air-raid alert activity across Ukraine's oblasts over time: a
monthly choropleth map, live "currently active" tracking, total alert counts and cumulative
time-under-alert by oblast, escalation/de-escalation trends, and alert duration distributions.

Built for a hackathon. Data comes from two places:

- **Live data** — polled from the [alerts.in.ua](https://alerts.in.ua) API
- **Historical data sample** — loaded from the Kaggle [Air-raid sirens in Ukraine](https://www.kaggle.com/datasets/cashncarry/airraid-sirens-in-ukraine) dataset (this is what the submitted demo uses, since we didn't receive API access in time)

## Features

- 🗺️ **Monthly map** — choropleth of Ukraine shaded by alert count, with a month slider
- 🔴 **Currently Active panel** — live list of ongoing alerts, including hromada/raion-level detail when available
- 📊 **Total alerts by oblast** — ranked by quantity *and* by cumulative time under alert (two different stories — an oblast can rank high on one and not the other)
- 📈 **Daily alerts by oblast** — filterable by oblast, date range, and alert duration bucket
- 📉 **Escalation / de-escalation trend** — 7-day rolling average with a stable/escalating/de-escalating readout
- ⏱️ **Duration distribution** — histogram of how long alerts last, by alert type

## Tech stack

- **[Dash](https://dash.plotly.com/)** + **Plotly** + **dash-bootstrap-components** for the dashboard
- **[DuckDB](https://duckdb.org/)** as the embedded data store (single file, no server)
- **httpx** + **pydantic** + **tenacity** for the live ingestion pipeline
- **pandas** for data wrangling
- **[uv](https://docs.astral.sh/uv/)** for dependency management

## Project structure

```
.
├── app.py                     # Dash dashboard (run this to view the app)
├── src/
│   ├── ingest.py               # Live ingestion: polls the alerts.in.ua API into DuckDB
│   └── retrieve_data_sample.py # Loads the real historical data sample into DuckDB
├── data/
│   └── full_data.csv            # Kaggle dataset, oblast+raion+hromada-level (place here manually)
├── pyproject.toml             # uv-managed dependencies
├── uv.lock
├── .env.example                # Copy to .env and fill in your API token
└── alerts.duckdb                # Created automatically on first run (gitignored)
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it already.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   ```bash
   # Required only if you want live data (see "Running" below)
   ALERTS_API_TOKEN=your_token_here   # get one at https://alerts.in.ua/api-request

   # Optional — defaults to ./alerts.duckdb
   ALERTS_DB_PATH=alerts.duckdb
   ```

## Running

You need data in the database before the dashboard shows anything interesting. Pick one or
both of the options below — they're safe to combine, since each source is tagged and
stored in its own id range (see [Data sources](#data-sources--id-ranges) below).

### Option A — Live + recent history (real, requires an API token)

```bash
# One-time backfill of the last ~month, per oblast
uv run python src/ingest.py backfill

# Continuous polling for new/ongoing alerts (run this in its own terminal, leave it running)
uv run python src/ingest.py live
```

### Option B — Historical data sample (real, no API token needed)

This is the data source used for the submitted demo. Download `full_data.csv` from
[the Kaggle dataset page](https://www.kaggle.com/datasets/cashncarry/airraid-sirens-in-ukraine)
into the `data/` folder, then:

```bash
uv run python src/retrieve_data_sample.py
```

See [Kaggle dataset structure](#kaggle-dataset-structure) below for what the CSV looks like.

### Launch the dashboard

```bash
uv run python app.py
```

Open **http://localhost:8050**.

## Kaggle dataset structure

The loader (`src/retrieve_data_sample.py`) expects `full_data.csv` (oblast + raion +
hromada granularity):

| Column | Meaning |
|---|---|
| `oblast` | Parent oblast name (English) |
| `raion` | District name, populated for finer-than-oblast rows |
| `hromada` | Community (hromada) name, populated when `level == "hromada"` |
| `level` | Granularity of this specific row: `"oblast"`, `"hromada"`, etc. |
| `started_at` | Alert start timestamp (UTC) |
| `finished_at` | Alert end timestamp (UTC), if known |

### How this maps into the database

- Oblast names are matched **regardless of language** — the dataset uses English
  transliteration (`"Kharkivska oblast"`), while the live API and the dashboard's map use
  Ukrainian (`"Харківська область"`). Every row is canonicalized to the Ukrainian name
  on load, since that's what the choropleth map joins against.
- `hromada`/`raion` values flow into `location_raion` / `location_title`, which is what
  lets the Currently Active panel show finer-than-oblast detail when that data is loaded.

### Known limitation

A handful of hromada names in this export are not unique across oblasts (e.g.
"Kostiantynivska terytorialna hromada" appears under four different oblasts with identical
timestamps). This looks like a name-collision artifact in the source data rather than four
simultaneous real alerts, and it will slightly inflate per-oblast totals for the affected
oblasts. Dedupe on `(started_at, finished_at, location_title)` before relying on exact
oblast-level counts, or before building any filter that assumes hromada names are unique
to one oblast.

## Data sources & id ranges

Each data source is tagged so it can be identified and cleared independently:

| Source | `notes` value | id range |
|---|---|---|
| Live API (`ingest.py`) | *(none)* | real alerts.in.ua ids |
| Historical data sample (`retrieve_data_sample.py`) | `KAGGLE_HISTORICAL_DATA` | `800,000,000`–`899,999,999` |

Pass `--clear` to `retrieve_data_sample.py` to wipe its rows before reloading.

## Attribution

- Alert data: [alerts.in.ua](https://alerts.in.ua) ([API docs](https://devs.alerts.in.ua/))
- Historical dataset: [cashncarry/airraid-sirens-in-ukraine](https://www.kaggle.com/datasets/cashncarry/airraid-sirens-in-ukraine) on Kaggle
- Oblast boundaries: [EugeneBorshch/ukraine_geojson](https://github.com/EugeneBorshch/ukraine_geojson)