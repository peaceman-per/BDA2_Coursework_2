# NYC TLC Big Data Analysis - Coursework 2 (DSM010)

**Module**: DSM010 Big Data Analytics 2  
**Cluster**: Lena  
**Stack**: HDFS - Apache Spark (PySpark) - Spark MLlib

---

## Project Overview

This project analyses **NYC Taxi and Limousine Commission (TLC) trip records** from January 2018 to November 2025 across three services: Yellow taxis, Green taxis and For-Hire Vehicles (FHV).

### Research Hypotheses

| # | Hypothesis | Gold mart used |
|---|-----------|---------------|
| **H1** | Yellow taxi demand follows a strong hourly and weekday cycle; peak demand is concentrated in Manhattan. | `gold/zone_hour_demand` |
| **H2** | Monthly trip volumes across all three services show a sustained post-pandemic recovery trend from 2022 to 2025, but yellow taxi volumes remain below pre-COVID peak levels. | `gold/monthly_service_summary` |
| **H3** | Trips originating from airports (JFK, LGA, EWR) have a significantly higher share of long-duration trips (>45 min) compared to central zones. | `gold/zone_hour_reliability` |
| **H4** | A GBT model trained on lag and calendar features can forecast next-hour pickup demand with lower RMSE than a seasonal-naive baseline (lag-24). | `gold/model_metrics` |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | NYC TLC Trip Record Data (public domain) |
| Period | 2022-01 - 2025-11 |
| Services | Yellow taxi, Green taxi, FHV |
| Format | Monthly Parquet files on HDFS |
| Size | ~10 GB uncompressed |

---

### HDFS Layout

```
hdfs://lena/user/wsidn001/bda2/coursework-2/
├── dat/
│   └── {YYYY}/
│       ├── yellow_tripdata_{YYYY}-{MM}.parquet
│       ├── green_tripdata_{YYYY}-{MM}.parquet
│       └── fhv_tripdata_{YYYY}-{MM}.parquet
└── out/nyc_tlc/
    ├── silver/trips/           <- cleaned, unified schema
    ├── subset/trips/           <- =<10 MB sample for marker
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

## Directory Structure

```
BDA2_Coursework_2/
├── README.md
├── notebooks/
│   └── cw2_nyc_tlc.ipynb       - Main analysis notebook (run on Lena)
├── src/
│   ├── tlc_config.py           - Centralised config (paths, constants)
│   ├── tlc_io.py               - HDFS data loading helpers
│   ├── tlc_transform.py        - Schema harmonisation (Bronze -> Silver)
│   ├── tlc_analytics.py        - Gold mart aggregation functions
│   └── tlc_ml.py               - Spark MLlib GBT pipeline
├── scripts/
│   ├── make_subset.py          - Create =<10 MB HDFS subset
│   └── capture_submission_evidence.sh
└── evidence/
    └── .gitkeep                - evidence/ls-l.txt written here at submission
```

## Configuration

All tuneable parameters live in `src/tlc_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_SUBSET` | `True` | Use the small HDFS subset instead of full data |
| `SUBSET_MONTH` | `"2022-06"` | Month to extract for the subset |
| `SUBSET_SERVICE` | `"yellow"` | Service type for the subset |
| `START_YM` | `"2022-01"` | Full-data range start |
| `END_YM` | `"2025-11"` | Full-data range end |
| `TOP_ZONES_N` | `50` | Top-N zones used in ML training |
| `TRAIN_END_YM` | `"2024-12"` | ML train/test cut-off |
| `LONG_TRIP_THRESHOLD_SEC` | `2700` | Threshold for H3 reliability metric (45 min) |
| `SHUFFLE_PARTITIONS` | `400` | Spark shuffle partition count |



---

## Notebook Sections

| Section | Title | Hypothesis |
|---------|-------|-----------|
| 0 | Parameters & Imports | - |
| 1 | Spark Session Setup | - |
| 2 | Data Discovery | - |
| 3 | Schema Harmonisation (Bronze -> Silver) | - |
| 4 | Gold Mart: Zone-Hour Demand | H1 |
| 5 | Gold Mart: Monthly Service Summary | H2 |
| 6 | Gold Mart: Zone-Hour Reliability | H3 |
| 7 | EDA & Visualisation | H1-H3 |
| 8 | Performance Benchmarking | - |
| 9 | ML Dataset & Feature Engineering | H4 |
| 10 | ML Pipeline & Evaluation | H4 |
| 11 | Subset Mode Demonstration | - |
| 12 | Final Summary | - |
