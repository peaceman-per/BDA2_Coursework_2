"""
tlc_transform.py – Schema harmonisation (Bronze → Silver) for NYC TLC data.

Each service type has different column names across years.  The three
standardise_* functions map each service to a canonical schema so all data
can be unioned into a single Silver DataFrame.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, TimestampType
from typing import Optional

# ── Canonical column list ────────────────────────────────────────────────────
#
# Columns guaranteed to exist after standardisation (nullable where noted):
#   service_type  str
#   pickup_datetime   timestamp
#   dropoff_datetime  timestamp  (nullable for some FHV files)
#   pu_location_id    int
#   do_location_id    int        (nullable for some FHV files)
#   passenger_count   int        (nullable)
#   trip_distance     double     (nullable)
#   fare_amount       double     (nullable)
#   total_amount      double     (nullable)
#   year, month, day, hour, dow  int (derived)
#   trip_duration_sec double     (derived)
#   valid_time        boolean
#   valid_locations   boolean

CANONICAL_COLS = [
    "service_type",
    "pickup_datetime",
    "dropoff_datetime",
    "pu_location_id",
    "do_location_id",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "year", "month", "day", "hour", "dow",
    "trip_duration_sec",
    "valid_time",
    "valid_locations",
]

MAX_DURATION_SEC = 4 * 3600   # 4 hours


def _find_col(df: DataFrame, *candidates: str):
    """Return the first column name that exists (case-insensitive), or None."""
    col_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        actual = col_map.get(cand.lower())
        if actual is not None:
            return actual
    return None


def _coerce_col(df: DataFrame, col_in: Optional[str], col_out: str, dtype) -> DataFrame:
    """Rename + cast a column; produce a null column if col_in is absent."""
    if col_in is not None and col_in in df.columns:
        return df.withColumn(col_out, F.col(col_in).cast(dtype))
    return df.withColumn(col_out, F.lit(None).cast(dtype))


def _add_time_cols(df: DataFrame) -> DataFrame:
    """Add year/month/day/hour/dow and trip_duration_sec."""
    df = (
        df
        .withColumn("year",  F.year("pickup_datetime").cast(IntegerType()))
        .withColumn("month", F.month("pickup_datetime").cast(IntegerType()))
        .withColumn("day",   F.dayofmonth("pickup_datetime").cast(IntegerType()))
        .withColumn("hour",  F.hour("pickup_datetime").cast(IntegerType()))
        .withColumn("dow",   F.dayofweek("pickup_datetime").cast(IntegerType()))
    )
    df = df.withColumn(
        "trip_duration_sec",
        (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")).cast(DoubleType()),
    )
    return df


def _add_quality_flags(df: DataFrame) -> DataFrame:
    """Add valid_time and valid_locations boolean flags."""
    df = df.withColumn(
        "valid_time",
        F.col("pickup_datetime").isNotNull()
        & F.col("trip_duration_sec").isNotNull()
        & (F.col("trip_duration_sec") > 0)
        & (F.col("trip_duration_sec") <= MAX_DURATION_SEC),
    )
    df = df.withColumn(
        "valid_locations",
        F.col("pu_location_id").isNotNull() & F.col("do_location_id").isNotNull(),
    )
    return df


# ── Yellow taxi ──────────────────────────────────────────────────────────────

def standardize_yellow(df: DataFrame) -> DataFrame:
    """Map Yellow taxi raw columns to canonical schema."""
    # Column names changed across years; handle both versions
    pu_col = _find_col(df, "tpep_pickup_datetime", "pickup_datetime")
    do_col = _find_col(df, "tpep_dropoff_datetime", "dropoff_datetime")
    pu_loc = _find_col(df, "PULocationID", "pu_location_id")
    do_loc = _find_col(df, "DOLocationID", "do_location_id")

    if pu_col is not None:
        df = df.withColumn("pickup_datetime", F.col(pu_col).cast(TimestampType()))
    else:
        df = df.withColumn("pickup_datetime", F.lit(None).cast(TimestampType()))

    if do_col is not None:
        df = df.withColumn("dropoff_datetime", F.col(do_col).cast(TimestampType()))
    else:
        df = df.withColumn("dropoff_datetime", F.lit(None).cast(TimestampType()))

    df = _coerce_col(df, pu_loc,           "pu_location_id",  IntegerType())
    df = _coerce_col(df, do_loc,           "do_location_id",  IntegerType())
    df = _coerce_col(df, "passenger_count","passenger_count", IntegerType())
    df = _coerce_col(df, "trip_distance",  "trip_distance",   DoubleType())
    df = _coerce_col(df, "fare_amount",    "fare_amount",     DoubleType())
    df = _coerce_col(df, "total_amount",   "total_amount",    DoubleType())
    df = _add_time_cols(df)
    df = _add_quality_flags(df)
    return df.select(CANONICAL_COLS)


# ── Green taxi ───────────────────────────────────────────────────────────────

def standardize_green(df: DataFrame) -> DataFrame:
    """Map Green taxi raw columns to canonical schema."""
    pu_col = _find_col(df, "lpep_pickup_datetime", "pickup_datetime")
    do_col = _find_col(df, "lpep_dropoff_datetime", "dropoff_datetime")
    pu_loc = _find_col(df, "PULocationID", "pu_location_id")
    do_loc = _find_col(df, "DOLocationID", "do_location_id")

    if pu_col is not None:
        df = df.withColumn("pickup_datetime", F.col(pu_col).cast(TimestampType()))
    else:
        df = df.withColumn("pickup_datetime", F.lit(None).cast(TimestampType()))

    if do_col is not None:
        df = df.withColumn("dropoff_datetime", F.col(do_col).cast(TimestampType()))
    else:
        df = df.withColumn("dropoff_datetime", F.lit(None).cast(TimestampType()))

    df = _coerce_col(df, pu_loc,           "pu_location_id",  IntegerType())
    df = _coerce_col(df, do_loc,           "do_location_id",  IntegerType())
    df = _coerce_col(df, "passenger_count","passenger_count", IntegerType())
    df = _coerce_col(df, "trip_distance",  "trip_distance",   DoubleType())
    df = _coerce_col(df, "fare_amount",    "fare_amount",     DoubleType())
    df = _coerce_col(df, "total_amount",   "total_amount",    DoubleType())
    df = _add_time_cols(df)
    df = _add_quality_flags(df)
    return df.select(CANONICAL_COLS)


# ── FHV (For-Hire Vehicle) ────────────────────────────────────────────────────

def standardize_fhv(df: DataFrame) -> DataFrame:
    """
    Map FHV raw columns to canonical schema.
    FHV files may lack passenger_count, trip_distance, fare_amount,
    total_amount.  These are set to null.
    """
    pu_col = _find_col(df, "pickup_datetime", "Pickup_DateTime", "Pickup_datetime")
    do_col = _find_col(df, "dropoff_datetime", "DropOff_DateTime", "Dropoff_datetime", "dropOff_datetime")
    pu_loc = _find_col(df, "PUlocationID", "PULocationID", "pulocationid", "pu_location_id")
    do_loc = _find_col(df, "DOlocationID", "DOLocationID", "dolocationid", "do_location_id")

    if pu_col is not None:
        df = df.withColumn("pickup_datetime", F.col(pu_col).cast(TimestampType()))
    else:
        df = df.withColumn("pickup_datetime", F.lit(None).cast(TimestampType()))

    if do_col is not None:
        df = df.withColumn("dropoff_datetime", F.col(do_col).cast(TimestampType()))
    else:
        df = df.withColumn("dropoff_datetime", F.lit(None).cast(TimestampType()))

    df = _coerce_col(df, pu_loc, "pu_location_id", IntegerType())
    df = _coerce_col(df, do_loc, "do_location_id", IntegerType())
    # FHV typically lacks these; fill with null
    for col_name, dtype in [("passenger_count", IntegerType()),
                             ("trip_distance",   DoubleType()),
                             ("fare_amount",      DoubleType()),
                             ("total_amount",     DoubleType())]:
        df = df.withColumn(col_name, F.lit(None).cast(dtype))

    df = _add_time_cols(df)
    df = _add_quality_flags(df)
    return df.select(CANONICAL_COLS)


# ── Dispatcher ───────────────────────────────────────────────────────────────

_STANDARDIZERS = {
    "yellow": standardize_yellow,
    "green":  standardize_green,
    "fhv":    standardize_fhv,
}


def standardize(df: DataFrame, service: str) -> DataFrame:
    """Dispatch to the correct standardiser based on service name."""
    fn = _STANDARDIZERS.get(service)
    if fn is None:
        raise ValueError(f"Unknown service type: {service!r}")
    return fn(df)
