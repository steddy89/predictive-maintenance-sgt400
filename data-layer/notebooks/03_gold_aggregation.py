# ===========================================================================
# Fabric Notebook: Gold Layer - Aggregated Model-Ready Dataset
# ===========================================================================
# Aggregates Silver data into business-level KPIs, health scores,
# and model-ready datasets for the Gold layer.
#
# Reference:
#   - https://learn.microsoft.com/fabric/data-science/tutorial-data-science-ingest-data
# ===========================================================================

# CELL 1 - Imports
# ---------------------------------------------------------------------------
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, avg, stddev, min as spark_min, max as spark_max, count, sum as spark_sum,
    when, lit, current_timestamp, date_trunc, first, last, round as spark_round,
    percentile_approx, collect_list, struct, to_json
)

spark = SparkSession.builder.appName("SGT400_Gold_Aggregation").getOrCreate()

SENSOR_COLUMNS = [
    "exhaust_gas_temp_c", "compressor_discharge_temp_c", "vibration_mm_s",
    "inlet_pressure_bar", "discharge_pressure_bar", "turbine_load_mw",
    "fuel_flow_kg_s", "rotor_speed_rpm", "lube_oil_temp_c", "ambient_temp_c",
]

# CELL 2 - Read Silver Data
# ---------------------------------------------------------------------------
df_silver = spark.table("silver_sensor_features")

# CELL 3 - Hourly Aggregated KPIs
# ---------------------------------------------------------------------------
def create_hourly_kpis(df):
    """Aggregate sensor data to hourly KPIs per turbine."""
    
    df_hourly = df.withColumn("hour_bucket", date_trunc("hour", col("timestamp")))
    
    agg_exprs = []
    for sensor in SENSOR_COLUMNS:
        agg_exprs.extend([
            avg(col(sensor)).alias(f"{sensor}_avg"),
            spark_min(col(sensor)).alias(f"{sensor}_min"),
            spark_max(col(sensor)).alias(f"{sensor}_max"),
            stddev(col(sensor)).alias(f"{sensor}_std"),
        ])
    
    # Add anomaly counts and derived metrics
    agg_exprs.extend([
        count("*").alias("reading_count"),
        spark_sum(when(col("is_anomaly"), 1).otherwise(0)).alias("anomaly_count"),
        avg(col("anomaly_score")).alias("avg_anomaly_score"),
        avg(col("pressure_ratio")).alias("avg_pressure_ratio"),
        avg(col("heat_rate")).alias("avg_heat_rate"),
        avg(col("efficiency_pct")).alias("avg_efficiency"),
        avg(col("specific_fuel_consumption")).alias("avg_sfc"),
        avg(col("temp_differential")).alias("avg_temp_differential"),
    ])
    
    df_agg = (
        df_hourly
        .groupBy("turbine_id", "hour_bucket")
        .agg(*agg_exprs)
    )
    
    return df_agg

df_hourly = create_hourly_kpis(df_silver)

# CELL 4 - Turbine Health Score Calculation
# ---------------------------------------------------------------------------
def calculate_health_score(df):
    """
    Compute a composite Turbine Health Score (0-100).
    
    Score components:
    - Temperature health (25%): EGT and CDT within normal range
    - Vibration health (25%): Low vibration levels
    - Efficiency health (25%): Operating near design efficiency
    - Stability health (25%): Low variance in readings
    """
    
    # Temperature health: penalize deviation from nominal
    df = df.withColumn(
        "temp_health",
        spark_round(
            when(col("exhaust_gas_temp_c_avg") < 520, 100)
            .when(col("exhaust_gas_temp_c_avg") < 560, 100 - (col("exhaust_gas_temp_c_avg") - 520) * 1.25)
            .when(col("exhaust_gas_temp_c_avg") < 590, 50 - (col("exhaust_gas_temp_c_avg") - 560) * 1.0)
            .otherwise(10),
            1
        )
    )
    
    # Vibration health
    df = df.withColumn(
        "vibration_health",
        spark_round(
            when(col("vibration_mm_s_avg") < 2.5, 100)
            .when(col("vibration_mm_s_avg") < 4.0, 100 - (col("vibration_mm_s_avg") - 2.5) * 20)
            .when(col("vibration_mm_s_avg") < 7.0, 70 - (col("vibration_mm_s_avg") - 4.0) * 15)
            .otherwise(10),
            1
        )
    )
    
    # Efficiency health
    df = df.withColumn(
        "efficiency_health",
        spark_round(
            when(col("avg_efficiency") > 95, 100)
            .when(col("avg_efficiency") > 85, 100 - (95 - col("avg_efficiency")) * 5)
            .when(col("avg_efficiency") > 70, 50 - (85 - col("avg_efficiency")) * 2)
            .otherwise(10),
            1
        )
    )
    
    # Stability health: based on sensor standard deviations
    df = df.withColumn(
        "stability_health",
        spark_round(
            (100 - col("avg_anomaly_score") * 15).cast("double"),
            1
        )
    )
    df = df.withColumn(
        "stability_health",
        when(col("stability_health") < 0, 0).otherwise(col("stability_health"))
    )
    
    # Composite health score
    df = df.withColumn(
        "health_score",
        spark_round(
            col("temp_health") * 0.25
            + col("vibration_health") * 0.25
            + col("efficiency_health") * 0.25
            + col("stability_health") * 0.25,
            1
        )
    )
    
    # Risk classification
    df = df.withColumn(
        "risk_level",
        when(col("health_score") >= 80, "LOW")
        .when(col("health_score") >= 60, "MEDIUM")
        .when(col("health_score") >= 40, "HIGH")
        .otherwise("CRITICAL")
    )
    
    return df

df_health = calculate_health_score(df_hourly)

# CELL 5 - Write Gold: Hourly KPIs with Health Score
# ---------------------------------------------------------------------------
(
    df_health
    .withColumn("processing_timestamp", current_timestamp())
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("turbine_id")
    .option("overwriteSchema", "true")
    .saveAsTable("gold_turbine_hourly_kpis")
)

print(f"Gold hourly KPIs written: {df_health.count():,} records")

# CELL 6 - Latest Turbine Status (Snapshot)
# ---------------------------------------------------------------------------
def create_latest_status(df):
    """Create a snapshot table of the latest status for each turbine."""
    
    window_latest = Window.partitionBy("turbine_id").orderBy(col("hour_bucket").desc())
    
    df_latest = (
        df.withColumn("rn", row_number().over(window_latest))
        .filter(col("rn") == 1)
        .drop("rn")
    )
    
    return df_latest

from pyspark.sql.functions import row_number

df_latest = create_latest_status(df_health)

(
    df_latest
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold_turbine_latest_status")
)

print(f"Latest status snapshot: {df_latest.count()} turbines")

# CELL 7 - Alert History Gold Table
# ---------------------------------------------------------------------------
df_alarms = spark.table("bronze_alarm_logs")

df_alerts_gold = (
    df_alarms
    .withColumn("hour_bucket", date_trunc("hour", col("timestamp")))
    .groupBy("turbine_id", "hour_bucket", "alarm_code", "severity")
    .agg(
        count("*").alias("occurrence_count"),
        avg("sensor_value").alias("avg_sensor_value"),
        first("alarm_description").alias("description"),
        first("threshold").alias("threshold"),
    )
    .withColumn("processing_timestamp", current_timestamp())
)

(
    df_alerts_gold
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("turbine_id")
    .option("overwriteSchema", "true")
    .saveAsTable("gold_alert_history")
)

print(f"Gold alert history: {df_alerts_gold.count():,} aggregated records")

# CELL 8 - ML Training Dataset (Gold)
# ---------------------------------------------------------------------------
def create_ml_training_dataset(df_silver_full):
    """
    Create a curated ML-ready dataset for model training.
    Includes all features, labels, and proper train/test split metadata.
    """
    
    feature_cols = [c for c in df_silver_full.columns if c not in [
        "timestamp", "turbine_id", "ingestion_timestamp", "source_file",
        "data_quality_flag", "year", "month", "day", "processing_timestamp"
    ]]
    
    df_ml = df_silver_full.select(
        "timestamp", "turbine_id", *feature_cols
    )
    
    # Add binary label for anomaly detection
    df_ml = df_ml.withColumn(
        "is_faulty",
        when(col("fault_type") != "normal", 1).otherwise(0)
    )
    
    return df_ml

df_ml = create_ml_training_dataset(df_silver)

(
    df_ml
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("turbine_id")
    .option("overwriteSchema", "true")
    .saveAsTable("gold_ml_training_data")
)

print(f"ML training dataset: {df_ml.count():,} records, {len(df_ml.columns)} features")
print(f"Positive (faulty) ratio: {df_ml.filter(col('is_faulty') == 1).count() / df_ml.count() * 100:.1f}%")

# CELL 9 - Gold Layer Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GOLD LAYER SUMMARY")
print("=" * 70)
for table in ["gold_turbine_hourly_kpis", "gold_turbine_latest_status", 
              "gold_alert_history", "gold_ml_training_data"]:
    df_t = spark.table(table)
    print(f"  {table}: {df_t.count():,} rows, {len(df_t.columns)} columns")
print("=" * 70)
