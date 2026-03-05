#!/usr/bin/env python3
"""
make_subset.py – Write a <=10 MB subset of NYC TLC data to HDFS.

Usage (from Lena JupyterHub terminal):
    python scripts/make_subset.py [--month 2019-06] [--service yellow] [--zones N]

The subset is written to:
    hdfs://lena/user/wsidn001/bda2/coursework-2/out/nyc_tlc/subset/trips/
"""

import argparse
import sys
import os

# Allow importing src modules without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from tlc_config import (
    HDFS_INPUT_ROOT,
    SUBSET_PATH,
    SUBSET_MONTH,
    SUBSET_SERVICE,
)
from tlc_io import load_service_raw
from tlc_transform import standardize


def make_subset(month: str, service: str, max_zones: int = 30):
    y_str, m_str = month.split("-")
    y, m = int(y_str), int(m_str)

    spark = (
        SparkSession.builder
        .appName("NYC_TLC_MakeSubset")
        .master("yarn")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Loading {service} for {month}…")
    df = load_service_raw(spark, service, start_ym=month, end_ym=month)
    if df is None:
        print("No data found – aborting.")
        spark.stop()
        return

    df_std = standardize(df, service)

    # Restrict to top max_zones pickup zones to keep file small
    top_zones = (
        df_std.groupBy("pu_location_id")
              .count()
              .orderBy(F.desc("count"))
              .limit(max_zones)
              .select("pu_location_id")
    )
    df_small = df_std.join(top_zones, on="pu_location_id", how="inner")

    df_small.write.mode("overwrite").parquet(SUBSET_PATH)
    count = df_small.count()
    print(f"Subset written to {SUBSET_PATH}  ({count:,} rows)")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create subset parquet for marker.")
    parser.add_argument("--month",   default=SUBSET_MONTH,   help="YYYY-MM of source month")
    parser.add_argument("--service", default=SUBSET_SERVICE, help="Service type (yellow/green/fhv)")
    parser.add_argument("--zones",   default=30, type=int,    help="Max number of pickup zones")
    args = parser.parse_args()
    make_subset(args.month, args.service, args.zones)
