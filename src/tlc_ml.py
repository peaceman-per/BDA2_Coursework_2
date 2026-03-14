"""
tlc_ml.py - Spark MLlib pipeline (GBTRegressor) for next-hour pickup forecasting.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

from tlc_config import SEED, TOP_ZONES_N, TRAIN_END_YM, TEST_START_YM


############# Feature engineering #############

def build_features(df_zone_hour: DataFrame, top_zones_n: int = TOP_ZONES_N) -> DataFrame:
    """
    Build lag/rolling features for next-hour pickup forecasting.

    Input:  zone_hour_demand mart (service_type, zone_id, ts_hour, pickups, ...)
    Output: model DataFrame with label (next_hour_pickups) + feature columns.
    """
    # Restrict to top N zones by total pickups for tractable training
    top_zones = (
        df_zone_hour
        .groupBy("zone_id")
        .agg(F.sum("pickups").alias("total_pickups"))
        .orderBy(F.desc("total_pickups"))
        .limit(top_zones_n)
        .select("zone_id")
    )
    df = df_zone_hour.join(top_zones, on="zone_id", how="inner")

    # window per zone ordered by ts_hour
    w = Window.partitionBy("zone_id").orderBy("ts_hour")

    df = (
        df
        # Lag features
        .withColumn("lag_1",   F.lag("pickups",  1).over(w))
        .withColumn("lag_24",  F.lag("pickups", 24).over(w))
        .withColumn("lag_168", F.lag("pickups",168).over(w))
        # rolling means via rangeFromSpec (rows-based approximation)
        .withColumn("rolling_mean_24",
                    F.avg("pickups").over(w.rowsBetween(-24, -1)))
        .withColumn("rolling_mean_168",
                    F.avg("pickups").over(w.rowsBetween(-168, -1)))
        # calendar features
        .withColumn("hour_of_day", F.hour("ts_hour"))
        .withColumn("day_of_week", F.dayofweek("ts_hour"))
        .withColumn("month_num",   F.month("ts_hour"))
        # Label: next-hour pickups
        .withColumn("label", F.lead("pickups", 1).over(w))
    )

    # drop rows with any null in key feature columns
    feature_cols = ["lag_1","lag_24","lag_168","rolling_mean_24","rolling_mean_168",
                    "hour_of_day","day_of_week","month_num","label"]
    df = df.dropna(subset=feature_cols)

    # add year/month for time-based split
    df = (
        df
        .withColumn("feat_year",  F.year("ts_hour"))
        .withColumn("feat_month", F.month("ts_hour"))
    )
    return df


############# Time-based train/test split #############

def time_split(df: DataFrame, train_end: str = TRAIN_END_YM, test_start: str = TEST_START_YM):
    """
    Split model_df into train/test by calendar date (no random shuffling).
    train_end and test_start are "YYYY-MM" strings.
    """
    te_y, te_m = map(int, train_end.split("-"))
    ts_y, ts_m = map(int, test_start.split("-"))

    train = df.filter(
        (F.col("feat_year") < te_y)
        | ((F.col("feat_year") == te_y) & (F.col("feat_month") <= te_m))
    )
    test = df.filter(
        (F.col("feat_year") > ts_y)
        | ((F.col("feat_year") == ts_y) & (F.col("feat_month") >= ts_m))
    )
    return train, test


############# Spark ML pipeline #############

NUMERIC_FEATURES = [
    "lag_1","lag_24","lag_168",
    "rolling_mean_24","rolling_mean_168",
    "hour_of_day","day_of_week","month_num",
]


def build_pipeline() -> Pipeline:
    """
    Build Spark ML Pipeline:
      VectorAssembler -> GBTRegressor
    (zone_id encoding omitted for simplicity; zone selection provides implicit grouping)
    """
    assembler = VectorAssembler(inputCols=NUMERIC_FEATURES, outputCol="features")
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="label",
        seed=SEED,
    )
    return Pipeline(stages=[assembler, gbt])


def build_param_grid(pipeline: Pipeline) -> list:
    """Modest cross-validation grid for GBTRegressor."""
    gbt = pipeline.getStages()[-1]
    return (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth,  [3, 5])
        .addGrid(gbt.maxIter,   [20, 50])
        .addGrid(gbt.stepSize,  [0.05, 0.1])
        .build()
    )


def train_with_cv(train_df: DataFrame) -> object:
    """Train a GBTRegressor with 3-fold CrossValidator and return the best model."""
    pipeline  = build_pipeline()
    param_grid = build_param_grid(pipeline)
    evaluator  = RegressionEvaluator(labelCol="label", metricName="rmse")

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        seed=SEED,
    )
    return cv.fit(train_df)


############# Baseline evaluation #############

def evaluate_baseline(test_df: DataFrame):
    """
    Seasonal-naive baseline: predict y_hat = lag_24.
    Returns (mae, rmse).
    """
    evaluator_mae  = RegressionEvaluator(labelCol="label",
                                         predictionCol="lag_24",
                                         metricName="mae")
    evaluator_rmse = RegressionEvaluator(labelCol="label",
                                         predictionCol="lag_24",
                                         metricName="rmse")
    # lag_24 must be cast to double for the evaluator
    df = test_df.withColumn("lag_24", F.col("lag_24").cast("double"))
    return evaluator_mae.evaluate(df), evaluator_rmse.evaluate(df)


############# Model evaluation #############

def evaluate_model(model, test_df: DataFrame):
    """Return (mae, rmse) for the fitted model on test_df."""
    preds = model.transform(test_df)
    evaluator_mae  = RegressionEvaluator(labelCol="label",
                                         predictionCol="prediction",
                                         metricName="mae")
    evaluator_rmse = RegressionEvaluator(labelCol="label",
                                         predictionCol="prediction",
                                         metricName="rmse")
    return evaluator_mae.evaluate(preds), evaluator_rmse.evaluate(preds), preds
