"""
tlc_analytics.py - Gold mart creation functions for NYC TLC Coursework 2.

Functions return Spark DataFrames.  Callers are responsible for saving.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from tlc_config import LONG_TRIP_THRESHOLD_SEC


############# GOLD 1: Zone-hour demand #############

def build_zone_hour_demand(df_silver: DataFrame) -> DataFrame:
    """
    Aggregate by (service_type, pu_location_id, ts_hour).

    Returns columns:
      service_type, zone_id, ts_hour, pickups,
      avg_total_amount, avg_trip_distance, p50_duration, p95_duration
    """
    df = df_silver.filter(F.col("valid_time") & F.col("valid_locations"))
    df = df.withColumn(
        "ts_hour",
        F.date_trunc("hour", F.col("pickup_datetime")),
    )
    return (
        df.groupBy("service_type", F.col("pu_location_id").alias("zone_id"), "ts_hour")
          .agg(
              F.count("*").alias("pickups"),
              F.avg("total_amount").alias("avg_total_amount"),
              F.avg("trip_distance").alias("avg_trip_distance"),
              F.expr("percentile_approx(trip_duration_sec, 0.50)").alias("p50_duration"),
              F.expr("percentile_approx(trip_duration_sec, 0.95)").alias("p95_duration"),
          )
          .orderBy("service_type", "zone_id", "ts_hour")
    )


############# GOLD 2: Monthly service summary #############

def build_monthly_service_summary(df_silver: DataFrame) -> DataFrame:
    """
    Monthly aggregate per service_type.

    Returns columns:
      service_type, year, month, trips,
      unique_pu_zones, avg_duration_sec, p95_duration_sec
    """
    df = df_silver.filter(F.col("valid_time"))
    return (
        df.groupBy("service_type", "year", "month")
          .agg(
              F.count("*").alias("trips"),
              F.countDistinct("pu_location_id").alias("unique_pu_zones"),
              F.avg("trip_duration_sec").alias("avg_duration_sec"),
              F.expr("percentile_approx(trip_duration_sec, 0.95)").alias("p95_duration_sec"),
          )
          .orderBy("service_type", "year", "month")
    )


############# GOLD 3: Zone-hour reliability #############

def build_zone_hour_reliability(df_silver: DataFrame) -> DataFrame:
    """
    Share of long trips (> LONG_TRIP_THRESHOLD_SEC) per zone-hour.

    Returns columns:
      service_type, zone_id, hour, share_long_trips, trip_count
    """
    df = df_silver.filter(F.col("valid_time") & F.col("valid_locations"))
    return (
        df.groupBy("service_type", F.col("pu_location_id").alias("zone_id"), "hour")
          .agg(
              F.count("*").alias("trip_count"),
              F.avg(
                  (F.col("trip_duration_sec") > LONG_TRIP_THRESHOLD_SEC).cast("int")
              ).alias("share_long_trips"),
          )
          .orderBy("service_type", "zone_id", "hour")
    )


############# Performance benchmarking helper #############

def benchmark_aggregation(spark, df_zone_hour, label="zone_hour_demand"):
    """
    Benchmark a simple aggregation on df_zone_hour with and without caching.
    """
    import time
    from pyspark.sql import Row

    def _run_agg(df):
        """A representative aggregation on the zone-hour mart."""
        return (
            df.groupBy("service_type", "zone_id")
              .agg(
                  F.sum("pickups").alias("total_pickups"),
                  F.avg("avg_total_amount").alias("mean_amount"),
              )
              .count()
        )

    rows = []

    # run 1: no cache
    df_zone_hour.unpersist()
    t0 = time.time()
    _run_agg(df_zone_hour)
    t1 = time.time()
    rows.append(Row(experiment=f"{label}_no_cache", elapsed_sec=round(t1 - t0, 2)))

    # run 2: with cache
    df_zone_hour.cache()
    df_zone_hour.count()  # materialise
    t2 = time.time()
    _run_agg(df_zone_hour)
    t3 = time.time()
    rows.append(Row(experiment=f"{label}_cached", elapsed_sec=round(t3 - t2, 2)))

    df_zone_hour.unpersist()

    return spark.createDataFrame(rows)
