"""
pg_export.py – PostgreSQL export via Spark JDBC for NYC TLC Coursework 2.

Reads connection details from environment variables:
  PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD

If any variable is missing the export is silently skipped.
"""

import os
from typing import Optional, Dict
from pyspark.sql import DataFrame, SparkSession


def _pg_url() -> Optional[str]:
    """Build JDBC URL from environment; return None if any var is unset."""
    host = os.environ.get("PG_HOST")
    port = os.environ.get("PG_PORT", "5432")
    db   = os.environ.get("PG_DB")
    if not host or not db:
        return None
    return f"jdbc:postgresql://{host}:{port}/{db}"


def _pg_props() -> Optional[Dict]:
    """Build JDBC properties dict; return None if credentials are unset."""
    user = os.environ.get("PG_USER")
    pwd  = os.environ.get("PG_PASSWORD")
    if not user or not pwd:
        return None
    return {
        "user":     user,
        "password": pwd,
        "driver":   "org.postgresql.Driver",
    }


def export_table(df: DataFrame, table: str, mode: str = "overwrite") -> bool:
    """
    Write df to PostgreSQL table via Spark JDBC.

    Returns True on success, False if credentials are absent.
    """
    url   = _pg_url()
    props = _pg_props()
    if url is None or props is None:
        print(f"Skipping PostgreSQL export for '{table}' (no env vars).")
        return False
    df.write.jdbc(url=url, table=table, mode=mode, properties=props)
    print(f"Exported → PostgreSQL table '{table}' ({mode}).")
    return True


def export_all(
    df_monthly_summary: DataFrame,
    df_model_metrics: DataFrame,
    df_perf_timings: DataFrame,
    df_zone_hour_2025: Optional[DataFrame] = None,
) -> None:
    """
    Export all curated gold tables to PostgreSQL.
    Each export is individually guarded against missing credentials.
    """
    export_table(df_monthly_summary, "cw2_monthly_service_summary")
    export_table(df_model_metrics,   "cw2_model_metrics")
    export_table(df_perf_timings,    "cw2_perf_timings")
    if df_zone_hour_2025 is not None:
        export_table(df_zone_hour_2025, "cw2_zone_hour_demand_2025")
