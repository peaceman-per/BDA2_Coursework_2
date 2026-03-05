# NYC TLC Big Data Analysis – Coursework 2 (DSM010)

> **Module**: DSM010 Big Data Analytics 2  
> **Cluster**: Lena (University YARN / HDFS cluster)  
> **Stack**: HDFS · Apache Spark (PySpark) · Spark MLlib · PostgreSQL (JDBC)

---

## Project Overview

This project analyses **NYC Taxi and Limousine Commission (TLC) trip records** from January 2018 to November 2025 across three service types: Yellow taxi, Green taxi, and For-Hire Vehicles (FHV).

### Research Hypotheses

| # | Hypothesis | Gold mart used |
|---|-----------|---------------|
| **H1** | Yellow taxi demand follows a strong hour-of-day and day-of-week cycle; peak demand zones are concentrated in Manhattan. | `gold/zone_hour_demand` |
| **H2** | Monthly trip volumes for all three service types declined sharply during 2020 (COVID-19) and have not fully recovered to 2019 levels by 2025. | `gold/monthly_service_summary` |
| **H3** | Trips originating from airport zones (JFK, LGA, EWR) have a significantly higher share of long-duration trips (>45 min) compared to central Manhattan zones. | `gold/zone_hour_reliability` |
| **H4** | A GBT model trained on lag and calendar features can forecast next-hour pickup demand with lower RMSE than a seasonal-naïve baseline (lag-24). | `gold/model_metrics` |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | NYC TLC Trip Record Data (public domain) |
| Period | 2018-01 – 2025-11 |
| Services | Yellow taxi, Green taxi, FHV |
| Format | Monthly Parquet files on HDFS |
| Approx. size | ~150 GB uncompressed |

### HDFS Layout

```
hdfs://lena/user/wsidn001/bda2/coursework-2/
├── dat/
│   └── {YYYY}/
│       ├── yellow_tripdata_{YYYY}_{MM}.parquet
│       ├── green_tripdata_{YYYY}_{MM}.parquet
│       └── fhv_tripdata_{YYYY}_{MM}.parquet
└── out/nyc_tlc/
    ├── silver/trips/           ← cleaned, unified schema
    ├── subset/trips/           ← ≤10 MB sample for marker
    └── gold/
        ├── zone_hour_demand/
        ├── monthly_service_summary/
        ├── zone_hour_reliability/
        ├── model_features/
        ├── model_predictions_sample/
        ├── model_metrics/
        └── perf_timings/
```

---

## Repository Structure

```
BDA2_Coursework_2/
├── README.md
├── notebooks/
│   └── cw2_nyc_tlc.ipynb       ← Main analysis notebook (run on Lena)
├── src/
│   ├── tlc_config.py           ← Centralised config (paths, constants)
│   ├── tlc_io.py               ← HDFS data loading helpers
│   ├── tlc_transform.py        ← Schema harmonisation (Bronze → Silver)
│   ├── tlc_analytics.py        ← Gold mart aggregation functions
│   ├── tlc_ml.py               ← Spark MLlib GBT pipeline
│   └── pg_export.py            ← PostgreSQL JDBC export
├── scripts/
│   ├── make_subset.py          ← Create ≤10 MB HDFS subset
│   └── capture_submission_evidence.sh
└── evidence/
    └── .gitkeep                ← evidence/ls-l.txt written here at submission
```

---

## Running on Lena JupyterHub

### 1. Clone / upload the repository

Upload the repository contents to your Lena home directory or clone via JupyterHub terminal:

```bash
# In the Lena JupyterHub terminal
git clone <your-repo-url> ~/BDA2_Coursework_2
cd ~/BDA2_Coursework_2
```

### 2. Open the notebook

Open **`notebooks/cw2_nyc_tlc.ipynb`** from the JupyterHub file browser.

Ensure the kernel is set to **PySpark** (or the cluster's Python 3 kernel with PySpark available on `PYTHONPATH`).

### 3. Subset mode (fast / default)

By default `USE_SUBSET = True` in `src/tlc_config.py`.  
The notebook reads only the pre-written subset from HDFS, so all cells complete quickly.

To **create** (or recreate) the subset:

```bash
# From JupyterHub terminal – writes subset to HDFS
python scripts/make_subset.py --month 2019-06 --service yellow --zones 30
```

### 4. Full-data run

Edit `src/tlc_config.py`:

```python
USE_SUBSET = False
```

Then re-run all notebook cells.  Expect runtimes of 20–60 minutes per cell depending on cluster load.

---

## Configuration

All tuneable parameters live in **`src/tlc_config.py`**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_SUBSET` | `True` | Use the small HDFS subset instead of full data |
| `SUBSET_MONTH` | `"2019-06"` | Month to extract for the subset |
| `SUBSET_SERVICE` | `"yellow"` | Service type for the subset |
| `START_YM` | `"2018-01"` | Full-data range start |
| `END_YM` | `"2025-11"` | Full-data range end |
| `TOP_ZONES_N` | `50` | Top-N zones used in ML training |
| `TRAIN_END_YM` | `"2024-12"` | ML train/test cut-off |
| `LONG_TRIP_THRESHOLD_SEC` | `2700` | Threshold for H3 reliability metric (45 min) |
| `SHUFFLE_PARTITIONS` | `400` | Spark shuffle partition count |

---

## PostgreSQL Export

The notebook exports four gold tables to PostgreSQL via Spark JDBC.  
Set the following environment variables **before** launching JupyterHub (or in a `.env` file sourced in the terminal):

```bash
export PG_HOST=<hostname>
export PG_PORT=5432          # default
export PG_DB=<database>
export PG_USER=<username>
export PG_PASSWORD=<password>
```

If any variable is absent the export step is silently skipped and the notebook continues without error.

Tables written:

| Table | Contents |
|-------|----------|
| `cw2_monthly_service_summary` | Monthly trip counts per service |
| `cw2_model_metrics` | MAE / RMSE for baseline and GBT model |
| `cw2_perf_timings` | Cache vs. no-cache benchmark timings |
| `cw2_zone_hour_demand_2025` | 2025 zone-hour demand sample |

---

## Submission Evidence

Run the following immediately before zipping the submission:

```bash
bash scripts/capture_submission_evidence.sh
```

This writes `evidence/ls-l.txt` with a timestamped recursive directory listing.  Include the `evidence/` folder in your submission zip.

---

## Notebook Sections

| Section | Title | Hypothesis |
|---------|-------|-----------|
| §0 | Parameters & Imports | – |
| §1 | Spark Session Setup | – |
| §2 | Data Discovery | – |
| §3 | Schema Harmonisation (Bronze → Silver) | – |
| §4 | Gold Mart: Zone-Hour Demand | H1 |
| §5 | Gold Mart: Monthly Service Summary | H2 |
| §6 | Gold Mart: Zone-Hour Reliability | H3 |
| §7 | EDA & Visualisation | H1–H3 |
| §8 | ML Dataset & Feature Engineering | H4 |
| §9 | ML Pipeline & Evaluation | H4 |
| §10 | PostgreSQL Export & Final Summary | – |
