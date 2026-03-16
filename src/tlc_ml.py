"""
tlc_ml.py - Spark MLlib pipeline (GBTRegressor) for next-hour pickup forecasting.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

from tlc_config import SEED, TOP_ZONES_N, TRAIN_END_YM, TEST_START_YM


############# Feature engineering #############

def build_features(df_zone_hour, top_zones_n=TOP_ZONES_N):
    top_zones = (
        df_zone_hour
        .groupBy("zone_id")
        .agg(F.sum("pickups").alias("total_pickups"))
        .orderBy(F.desc("total_pickups"))
        .limit(top_zones_n)
        .select("zone_id")
    )
    df = df_zone_hour.join(top_zones, on="zone_id", how="inner")

    w = Window.partitionBy("zone_id").orderBy("ts_hour")

    df = (
        df
        .withColumn("lag_1", F.lag("pickups",  1).over(w))
        .withColumn("lag_24", F.lag("pickups", 24).over(w))
        .withColumn("lag_168", F.lag("pickups",168).over(w))
        .withColumn("rolling_mean_24",
                    F.avg("pickups").over(w.rowsBetween(-24, -1)))
        .withColumn("rolling_mean_168",
                    F.avg("pickups").over(w.rowsBetween(-168, -1)))
        .withColumn("hour_of_day", F.hour("ts_hour"))
        .withColumn("day_of_week", F.dayofweek("ts_hour"))
        .withColumn("month_num", F.month("ts_hour"))
        .withColumn("label", F.lead("pickups", 1).over(w))
    )

    feature_cols = ["lag_1","lag_24","lag_168","rolling_mean_24","rolling_mean_168",
                    "hour_of_day","day_of_week","month_num","label"]
    df = df.dropna(subset=feature_cols)

    df = (
        df
        .withColumn("feat_year", F.year("ts_hour"))
        .withColumn("feat_month", F.month("ts_hour"))
    )
    return df


############# Time-based train/test split #############

def time_split(df, train_end=TRAIN_END_YM, test_start=TEST_START_YM):
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


def build_pipeline():
    assembler = VectorAssembler(inputCols=NUMERIC_FEATURES, outputCol="features")
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="label",
        seed=SEED,
    )
    return Pipeline(stages=[assembler, gbt])


def build_param_grid(pipeline):
    gbt = pipeline.getStages()[-1]
    return (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth,  [3, 5])
        .addGrid(gbt.maxIter,   [20, 50])
        .build()
    )


def train_with_cv(train_df):
    """Train a GBTRegressor with 3-fold CV using deterministic fold assignment."""
    pipeline = build_pipeline()
    param_grid = build_param_grid(pipeline)
    evaluator = RegressionEvaluator(labelCol="label", metricName="rmse")

    train_df = train_df.withColumn(
        "fold", F.abs(F.hash("zone_id", "ts_hour")) % 3
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3
    )
    return cv.fit(train_df)


############# Baseline evaluation #############

def evaluate_baseline(test_df):
    df = test_df.withColumn("lag_24", F.col("lag_24").cast("double"))
    evaluator_mae  = RegressionEvaluator(labelCol="label",
                                         predictionCol="lag_24",
                                         metricName="mae")
    evaluator_rmse = RegressionEvaluator(labelCol="label",
                                         predictionCol="lag_24",
                                         metricName="rmse")
    return evaluator_mae.evaluate(df), evaluator_rmse.evaluate(df)


############# Model evaluation #############

def evaluate_model(model, test_df):
    preds = model.transform(test_df)
    evaluator_mae  = RegressionEvaluator(labelCol="label",
                                         predictionCol="prediction",
                                         metricName="mae")
    evaluator_rmse = RegressionEvaluator(labelCol="label",
                                         predictionCol="prediction",
                                         metricName="rmse")
    return evaluator_mae.evaluate(preds), evaluator_rmse.evaluate(preds), preds