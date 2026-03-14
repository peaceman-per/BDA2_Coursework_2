"""
tlc_io.py – Data ingestion and HDFS I/O helpers for NYC TLC Coursework 2.
"""

import re
from contextlib import contextmanager
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from tlc_config import (
    HDFS_INPUT_ROOT,
    SERVICE_TYPES,
    START_YM,
    END_YM,
    SUBSET_PATH,
    SUBSET_MONTH,
    SUBSET_SERVICE,
)

############# Month iterator #############

def month_range(start_ym: str, end_ym: str):
    """Yield (year, month) int tuples from start_ym to end_ym inclusive."""
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


############# Robust file loader #############
# def _try_read(spark: SparkSession, path: str) -> DataFrame | None:
# too new!
# for older versions of python
from typing import Optional

def _try_read(spark: SparkSession, path: str) -> Optional[DataFrame]:
    """Attempt to read a parquet path; return None if it does not exist."""
    try:
        df = spark.read.parquet(path)
        # trigger schema resolution to detect missing path early
        _ = df.schema
        return df
    except Exception:
        return None

    
from pyspark.sql.functions import lit

# function to get union of column set and apply it to dfs
def align_columns(df1,df2):
    for col in df1.columns:
        if col not in df2.columns:
            df2 = df2.withColumn(col, lit(None))
    for col in df2.columns:
        if col not in df1.columns:
            df1 = df1.withColumn(col, lit(None))
    return df1.select(sorted(df1.columns)), df2.select(sorted(df1.columns))

# funct to load files inot a single df
def load_service_raw(
    spark: SparkSession,
    service: str,
    start_ym: str = START_YM,
    end_ym: str = END_YM,
) -> Optional[DataFrame]:
    """
    Load all monthly parquet files for one service_type and return a unioned
    DataFrame with service_type, year, and month columns added.

    Assumes input pattern:
      {HDFS_INPUT_ROOT}/{YYYY}/{service}_tripdata_{YYYY}-{MM}.parquet
    """
    dfs = []
    missing = []

    for y, m in month_range(start_ym, end_ym):
        fname = f"{service}_tripdata_{y:04d}-{m:02d}.parquet"
        p = f"{HDFS_INPUT_ROOT}/{y:04d}/{fname}"
        loaded = False
        df = _try_read(spark, p)
        if df is not None:
            df = (
                df.withColumn("service_type", F.lit(service))
                  .withColumn("src_year",  F.lit(y).cast("int"))
                  .withColumn("src_month", F.lit(m).cast("int"))
            )
            dfs.append(df)
            loaded = True
        if not loaded:
            missing.append(f"{y:04d}-{m:02d}")

    if missing:
        print(f"[{service}] Missing {len(missing)} file(s): {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")

    if not dfs:
        print(f"[{service}] No files found – skipping.")
        return None

    print(f"[{service}] Loaded {len(dfs)} file(s).")
    result = dfs[0]
    for d in dfs[1:]:
        result, d = align_columns(result, d)
        result = result.unionByName(d)
    return result

# old python fix again
from typing import List
from typing import Dict

def load_all_raw(
    spark: SparkSession,
    services: List[str] = SERVICE_TYPES,
    start_ym: str = START_YM,
    end_ym: str = END_YM,
) -> Dict[str, DataFrame]:
    """Return a dict {service_type: raw_df} for each service."""
    raw = {}
    for svc in services:
        df = load_service_raw(spark, svc, start_ym, end_ym)
        if df is not None:
            raw[svc] = df
    return raw


############# Subset loader #############

def load_subset(spark: SparkSession) -> DataFrame:
    """Read the pre-written subset parquet from HDFS."""
    return spark.read.parquet(SUBSET_PATH)


############# Save helper #############

def save_parquet(df: DataFrame, path: str, partition_cols: Optional[List[str]] = None, mode: str = "overwrite"):
    """Write a DataFrame to HDFS parquet with optional partitioning."""
    w = df.write.mode(mode)
    if partition_cols:
        w = w.partitionBy(*partition_cols)
    w.parquet(path)
    print(f"Saved -> {path}")
