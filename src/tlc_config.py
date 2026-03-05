"""
tlc_config.py – Centralised configuration for NYC TLC Coursework 2.
All HDFS paths, global constants, and default parameters are defined here.
"""

# ── HDFS paths ──────────────────────────────────────────────────────────────
HDFS_INPUT_ROOT   = "hdfs://lena/user/wsidn001/bda2/coursework-2/dat"
HDFS_PROJECT_ROOT = "hdfs://lena/user/wsidn001/bda2/coursework-2/out/nyc_tlc"

SILVER_PATH  = f"{HDFS_PROJECT_ROOT}/silver/trips"
SUBSET_PATH  = f"{HDFS_PROJECT_ROOT}/subset/trips"

GOLD_ZONE_HOUR_DEMAND     = f"{HDFS_PROJECT_ROOT}/gold/zone_hour_demand"
GOLD_MONTHLY_SUMMARY      = f"{HDFS_PROJECT_ROOT}/gold/monthly_service_summary"
GOLD_ZONE_HOUR_RELIABILITY= f"{HDFS_PROJECT_ROOT}/gold/zone_hour_reliability"
GOLD_MODEL_FEATURES       = f"{HDFS_PROJECT_ROOT}/gold/model_features"
GOLD_MODEL_PREDICTIONS    = f"{HDFS_PROJECT_ROOT}/gold/model_predictions_sample"
GOLD_MODEL_METRICS        = f"{HDFS_PROJECT_ROOT}/gold/model_metrics"
GOLD_PERF_TIMINGS         = f"{HDFS_PROJECT_ROOT}/gold/perf_timings"

# ── Dataset parameters ───────────────────────────────────────────────────────
SERVICE_TYPES = ["yellow", "green", "fhv"]
START_YM      = "2018-01"
END_YM        = "2025-11"

# ── Run toggles ──────────────────────────────────────────────────────────────
USE_SUBSET    = True          # Set False for full-data run
SUBSET_MODE   = "one_month"   # "one_month" | "sample_fraction"
SUBSET_MONTH  = "2019-06"     # month used for subset (small-enough month)
SUBSET_SERVICE= "yellow"      # service used for one_month subset

# ── ML parameters ────────────────────────────────────────────────────────────
SEED       = 42
TOP_ZONES_N = 50              # restrict to top N zones for ML training

# ML time-based split
TRAIN_END_YM = "2024-12"
TEST_START_YM= "2025-01"

# ── Reliability threshold (minutes → seconds) ────────────────────────────────
LONG_TRIP_THRESHOLD_SEC = 45 * 60   # 45 minutes

# ── Spark tuning ─────────────────────────────────────────────────────────────
SHUFFLE_PARTITIONS = 400
