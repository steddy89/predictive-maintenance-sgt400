# ===========================================================================
# Fabric Notebook: Bronze Layer Ingestion
# ===========================================================================
# Ingests raw SGT400 turbine sensor data into the Bronze layer of the
# Medallion Lakehouse architecture.
#
# Reference:
#   - https://learn.microsoft.com/fabric/data-engineering/lakehouse-overview
#   - https://learn.microsoft.com/fabric/data-engineering/get-started-streaming
# ===========================================================================

# CELL 1 - Configuration
# ---------------------------------------------------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType, BooleanType
)
from pyspark.sql.functions import (
    col, current_timestamp, lit, input_file_name, year, month, dayofmonth, hour
)

spark = SparkSession.builder.appName("SGT400_Bronze_Ingestion").getOrCreate()

# Lakehouse paths
BRONZE_PATH = "Tables/bronze"
RAW_DATA_PATH = "Files/raw/sensor_data"
ALARM_RAW_PATH = "Files/raw/alarm_logs"
CHECKPOINT_PATH = "Files/checkpoints/bronze"

# CELL 2 - Define Sensor Data Schema
# ---------------------------------------------------------------------------
sensor_schema = StructType([
    StructField("timestamp", TimestampType(), False),
    StructField("turbine_id", StringType(), False),
    StructField("exhaust_gas_temp_c", DoubleType(), True),
    StructField("compressor_discharge_temp_c", DoubleType(), True),
    StructField("vibration_mm_s", DoubleType(), True),
    StructField("inlet_pressure_bar", DoubleType(), True),
    StructField("discharge_pressure_bar", DoubleType(), True),
    StructField("turbine_load_mw", DoubleType(), True),
    StructField("fuel_flow_kg_s", DoubleType(), True),
    StructField("rotor_speed_rpm", DoubleType(), True),
    StructField("lube_oil_temp_c", DoubleType(), True),
    StructField("ambient_temp_c", DoubleType(), True),
    StructField("fault_type", StringType(), True),
    StructField("days_to_failure", DoubleType(), True),
    StructField("pressure_ratio", DoubleType(), True),
    StructField("heat_rate", DoubleType(), True),
    StructField("efficiency_pct", DoubleType(), True),
])

alarm_schema = StructType([
    StructField("timestamp", TimestampType(), False),
    StructField("turbine_id", StringType(), False),
    StructField("alarm_code", StringType(), True),
    StructField("alarm_description", StringType(), True),
    StructField("severity", StringType(), True),
    StructField("sensor_value", DoubleType(), True),
    StructField("threshold", DoubleType(), True),
    StructField("acknowledged", BooleanType(), True),
])

# CELL 3 - Batch Ingestion: Sensor Data to Bronze
# ---------------------------------------------------------------------------
def ingest_sensor_data_batch():
    """Ingest sensor data files (Parquet/CSV) into Bronze Delta table."""
    
    # Read raw Parquet files
    df_raw = (
        spark.read
        .schema(sensor_schema)
        .parquet(RAW_DATA_PATH)
    )
    
    # Add metadata columns
    df_bronze = (
        df_raw
        .withColumn("ingestion_timestamp", current_timestamp())
        .withColumn("source_file", input_file_name())
        .withColumn("data_quality_flag", lit("raw"))
        .withColumn("year", year(col("timestamp")))
        .withColumn("month", month(col("timestamp")))
        .withColumn("day", dayofmonth(col("timestamp")))
    )
    
    # Write to Bronze Delta table with partitioning
    (
        df_bronze.write
        .format("delta")
        .mode("append")
        .partitionBy("year", "month", "turbine_id")
        .saveAsTable("bronze_sensor_readings")
    )
    
    print(f"Ingested {df_bronze.count()} sensor records to Bronze layer")
    return df_bronze

df_bronze_sensors = ingest_sensor_data_batch()

# CELL 4 - Batch Ingestion: Alarm Logs to Bronze
# ---------------------------------------------------------------------------
def ingest_alarm_logs_batch():
    """Ingest alarm log files into Bronze Delta table."""
    
    df_raw = (
        spark.read
        .schema(alarm_schema)
        .parquet(ALARM_RAW_PATH)
    )
    
    df_bronze = (
        df_raw
        .withColumn("ingestion_timestamp", current_timestamp())
        .withColumn("source_file", input_file_name())
        .withColumn("year", year(col("timestamp")))
        .withColumn("month", month(col("timestamp")))
    )
    
    (
        df_bronze.write
        .format("delta")
        .mode("append")
        .partitionBy("year", "month")
        .saveAsTable("bronze_alarm_logs")
    )
    
    print(f"Ingested {df_bronze.count()} alarm records to Bronze layer")
    return df_bronze

df_bronze_alarms = ingest_alarm_logs_batch()

# CELL 5 - Streaming Ingestion (for real-time data)
# ---------------------------------------------------------------------------
def start_streaming_ingestion():
    """
    Start Spark Structured Streaming for near-real-time ingestion.
    Use this when turbine data arrives as a continuous stream.
    
    Reference: https://learn.microsoft.com/fabric/data-engineering/get-started-streaming
    """
    df_stream = (
        spark.readStream
        .format("cloudFiles")  # Auto Loader
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", f"{CHECKPOINT_PATH}/schema")
        .option("cloudFiles.inferColumnTypes", "true")
        .schema(sensor_schema)
        .load(RAW_DATA_PATH)
        .withColumn("ingestion_timestamp", current_timestamp())
        .withColumn("source_file", input_file_name())
        .withColumn("year", year(col("timestamp")))
        .withColumn("month", month(col("timestamp")))
    )
    
    query = (
        df_stream.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/sensor_stream")
        .partitionBy("year", "month", "turbine_id")
        .toTable("bronze_sensor_readings_stream")
    )
    
    return query

# Uncomment to start streaming:
# streaming_query = start_streaming_ingestion()

# CELL 6 - Data Quality Checks
# ---------------------------------------------------------------------------
def run_bronze_quality_checks():
    """Run basic quality checks on Bronze data."""
    
    df = spark.table("bronze_sensor_readings")
    
    total_rows = df.count()
    null_counts = {}
    for c in sensor_schema.fieldNames():
        null_count = df.filter(col(c).isNull()).count()
        if null_count > 0:
            null_counts[c] = null_count
    
    duplicate_count = total_rows - df.dropDuplicates(["timestamp", "turbine_id"]).count()
    
    # Date range
    date_range = df.agg(
        {"timestamp": "min", "timestamp": "max"}
    ).collect()[0]
    
    print("=" * 60)
    print("BRONZE LAYER QUALITY REPORT")
    print("=" * 60)
    print(f"Total Records: {total_rows:,}")
    print(f"Duplicate Records: {duplicate_count:,}")
    print(f"Null Values: {null_counts}")
    print(f"Turbine IDs: {[r[0] for r in df.select('turbine_id').distinct().collect()]}")
    print("=" * 60)

run_bronze_quality_checks()
