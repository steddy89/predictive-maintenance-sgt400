# ===========================================================================
# Fabric Notebook: Silver Layer - Cleaning & Feature Engineering
# ===========================================================================
# Transforms Bronze sensor data into cleaned, validated, and feature-enriched
# Silver layer tables for ML model consumption.
#
# Reference:
#   - https://learn.microsoft.com/fabric/data-science/tutorial-data-science-ingest-data
#   - https://learn.microsoft.com/fabric/data-engineering/lakehouse-overview
# ===========================================================================

# CELL 1 - Configuration & Imports
# ---------------------------------------------------------------------------
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, lag, lead, avg, stddev, min as spark_min, max as spark_max,
    when, abs as spark_abs, lit, current_timestamp, count,
    percentile_approx, expr, udf, row_number, monotonically_increasing_id
)
from pyspark.sql.types import DoubleType, StringType, IntegerType
from pyspark.ml.feature import StandardScaler, VectorAssembler
import numpy as np

spark = SparkSession.builder.appName("SGT400_Silver_Transform").getOrCreate()

SENSOR_COLUMNS = [
    "exhaust_gas_temp_c", "compressor_discharge_temp_c", "vibration_mm_s",
    "inlet_pressure_bar", "discharge_pressure_bar", "turbine_load_mw",
    "fuel_flow_kg_s", "rotor_speed_rpm", "lube_oil_temp_c", "ambient_temp_c",
]

DERIVED_COLUMNS = ["pressure_ratio", "heat_rate", "efficiency_pct"]

# Sensor physical limits for validation
SENSOR_LIMITS = {
    "exhaust_gas_temp_c": (100, 700),
    "compressor_discharge_temp_c": (100, 500),
    "vibration_mm_s": (0, 25),
    "inlet_pressure_bar": (0.5, 2.0),
    "discharge_pressure_bar": (5, 25),
    "turbine_load_mw": (0, 15),
    "fuel_flow_kg_s": (0, 10),
    "rotor_speed_rpm": (3000, 12000),
    "lube_oil_temp_c": (10, 100),
    "ambient_temp_c": (-20, 60),
}

# CELL 2 - Read Bronze Data
# ---------------------------------------------------------------------------
df_bronze = spark.table("bronze_sensor_readings")
print(f"Bronze records: {df_bronze.count():,}")

# CELL 3 - Data Cleaning
# ---------------------------------------------------------------------------
def clean_sensor_data(df):
    """
    Clean raw sensor data:
    1. Remove duplicates
    2. Filter out physically impossible values
    3. Handle nulls with forward-fill
    4. Flag anomalous readings
    """
    
    # Remove duplicates
    df_deduped = df.dropDuplicates(["timestamp", "turbine_id"])
    
    # Apply physical range validation
    df_valid = df_deduped
    for sensor, (low, high) in SENSOR_LIMITS.items():
        df_valid = df_valid.withColumn(
            sensor,
            when(
                (col(sensor) >= low) & (col(sensor) <= high),
                col(sensor)
            ).otherwise(None)
        )
    
    # Forward-fill nulls using window function
    window_spec = Window.partitionBy("turbine_id").orderBy("timestamp")
    
    for sensor in SENSOR_COLUMNS:
        df_valid = df_valid.withColumn(
            sensor,
            when(col(sensor).isNotNull(), col(sensor))
            .otherwise(lag(col(sensor), 1).over(window_spec))
        )
    
    # Drop remaining nulls (first record per turbine if null)
    df_valid = df_valid.dropna(subset=SENSOR_COLUMNS)
    
    return df_valid

df_clean = clean_sensor_data(df_bronze)
print(f"Clean records: {df_clean.count():,}")

# CELL 4 - Feature Engineering: Rolling Statistics
# ---------------------------------------------------------------------------
def add_rolling_features(df, windows=[12, 36, 144]):
    """
    Add rolling window features per turbine.
    
    Windows (at 5-min intervals):
    - 12  = 1 hour
    - 36  = 3 hours
    - 144 = 12 hours
    """
    
    for window_size in windows:
        window_spec = (
            Window.partitionBy("turbine_id")
            .orderBy("timestamp")
            .rowsBetween(-window_size, 0)
        )
        
        suffix = f"_{window_size}"
        
        for sensor in SENSOR_COLUMNS:
            # Rolling mean
            df = df.withColumn(
                f"{sensor}_rolling_mean{suffix}",
                avg(col(sensor)).over(window_spec)
            )
            # Rolling std
            df = df.withColumn(
                f"{sensor}_rolling_std{suffix}",
                stddev(col(sensor)).over(window_spec)
            )
            # Rolling min/max
            df = df.withColumn(
                f"{sensor}_rolling_min{suffix}",
                spark_min(col(sensor)).over(window_spec)
            )
            df = df.withColumn(
                f"{sensor}_rolling_max{suffix}",
                spark_max(col(sensor)).over(window_spec)
            )
    
    return df

df_features = add_rolling_features(df_clean, windows=[12, 144])

# CELL 5 - Feature Engineering: Rate of Change
# ---------------------------------------------------------------------------
def add_rate_of_change(df):
    """Calculate rate of change (derivative) for key sensors."""
    
    window_spec = Window.partitionBy("turbine_id").orderBy("timestamp")
    
    for sensor in SENSOR_COLUMNS:
        df = df.withColumn(
            f"{sensor}_rate_of_change",
            col(sensor) - lag(col(sensor), 1).over(window_spec)
        )
        # Acceleration (second derivative)
        df = df.withColumn(
            f"{sensor}_acceleration",
            col(f"{sensor}_rate_of_change") - lag(col(f"{sensor}_rate_of_change"), 1).over(window_spec)
        )
    
    return df

df_features = add_rate_of_change(df_features)

# CELL 6 - Feature Engineering: Cross-Sensor Correlations
# ---------------------------------------------------------------------------
def add_cross_sensor_features(df):
    """Add interaction features between related sensors."""
    
    # Temperature differential
    df = df.withColumn(
        "temp_differential",
        col("exhaust_gas_temp_c") - col("compressor_discharge_temp_c")
    )
    
    # Specific fuel consumption
    df = df.withColumn(
        "specific_fuel_consumption",
        col("fuel_flow_kg_s") / (col("turbine_load_mw") + 0.001)
    )
    
    # Vibration-to-speed ratio (normalized)
    df = df.withColumn(
        "vibration_speed_ratio",
        col("vibration_mm_s") / (col("rotor_speed_rpm") / 1000 + 0.001)
    )
    
    # Compression efficiency indicator
    df = df.withColumn(
        "compression_efficiency",
        col("discharge_pressure_bar") / (col("inlet_pressure_bar") * (col("compressor_discharge_temp_c") / col("ambient_temp_c") + 0.001))
    )
    
    # Temperature-load ratio
    df = df.withColumn(
        "temp_load_ratio",
        col("exhaust_gas_temp_c") / (col("turbine_load_mw") + 0.001)
    )
    
    return df

df_features = add_cross_sensor_features(df_features)

# CELL 7 - Anomaly Flagging (Statistical)
# ---------------------------------------------------------------------------
def add_anomaly_flags(df):
    """
    Flag statistical anomalies using z-score method.
    Points beyond 3 sigma are flagged.
    """
    
    window_spec = (
        Window.partitionBy("turbine_id")
        .orderBy("timestamp")
        .rowsBetween(-288, 0)  # 24-hour lookback at 5-min intervals
    )
    
    anomaly_cols = []
    
    for sensor in SENSOR_COLUMNS:
        mean_col = f"{sensor}_24h_mean"
        std_col = f"{sensor}_24h_std"
        zscore_col = f"{sensor}_zscore"
        anomaly_col = f"{sensor}_anomaly"
        
        df = df.withColumn(mean_col, avg(col(sensor)).over(window_spec))
        df = df.withColumn(std_col, stddev(col(sensor)).over(window_spec))
        df = df.withColumn(
            zscore_col,
            when(col(std_col) > 0, (col(sensor) - col(mean_col)) / col(std_col))
            .otherwise(0.0)
        )
        df = df.withColumn(
            anomaly_col,
            when(spark_abs(col(zscore_col)) > 3.0, 1).otherwise(0)
        )
        anomaly_cols.append(anomaly_col)
        
        # Clean up intermediate columns
        df = df.drop(mean_col, std_col)
    
    # Overall anomaly score (count of flagged sensors)
    df = df.withColumn(
        "anomaly_score",
        sum([col(c) for c in anomaly_cols])
    )
    df = df.withColumn(
        "is_anomaly",
        when(col("anomaly_score") >= 2, True).otherwise(False)
    )
    
    return df

df_silver = add_anomaly_flags(df_features)

# CELL 8 - Write Silver Table
# ---------------------------------------------------------------------------
(
    df_silver
    .withColumn("processing_timestamp", current_timestamp())
    .write
    .format("delta")
    .mode("overwrite")
    .partitionBy("turbine_id")
    .option("overwriteSchema", "true")
    .saveAsTable("silver_sensor_features")
)

print(f"Silver layer written: {df_silver.count():,} records")
print(f"Feature columns: {len(df_silver.columns)}")
print(f"Anomalies detected: {df_silver.filter(col('is_anomaly')).count():,}")

# CELL 9 - Silver Layer Quality Report
# ---------------------------------------------------------------------------
df_report = spark.table("silver_sensor_features")
print("=" * 70)
print("SILVER LAYER QUALITY REPORT")
print("=" * 70)
print(f"Total Records:    {df_report.count():,}")
print(f"Total Features:   {len(df_report.columns)}")
print(f"Turbines:         {df_report.select('turbine_id').distinct().count()}")
print(f"Date Range:       {df_report.agg({'timestamp': 'min'}).collect()[0][0]} to "
      f"{df_report.agg({'timestamp': 'max'}).collect()[0][0]}")
print(f"Anomaly Rate:     {df_report.filter(col('is_anomaly')).count() / df_report.count() * 100:.2f}%")
print("=" * 70)
