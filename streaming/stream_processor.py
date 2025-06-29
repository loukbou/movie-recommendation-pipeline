from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, expr
from pyspark.sql.types import StructType, StringType

# Initialize Spark with Delta Lake configuration (updated packages)
builder = SparkSession.builder \
    .appName("MultiTopicKafkaProcessor") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.streaming.schemaInference", "true")

spark = builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("Spark session initialized successfully!")

# Define schemas for each topic
schemas = {
    "movie-stream": StructType()
        .add("user_id", StringType())
        .add("movie_id", StringType())
        .add("watch_time", StringType()),

    "movie-likes": StructType()
        .add("user_id", StringType())
        .add("movie_id", StringType())
        .add("liked_at", StringType()),

    "movie-reviews": StructType()
        .add("user_id", StringType())
        .add("movie_id", StringType())
        .add("text", StringType())
        .add("reviewed_at", StringType())
}

# Read all topics into one DataFrame
print("Setting up Kafka stream...")
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", ",".join(schemas.keys())) \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Extract topic and value
df = df.select(
    col("topic").cast("string"),
    col("value").cast("string"),
    col("timestamp")
)

# Process each topic with separate logic
def process_topic(topic_name, schema, timestamp_col, output_path):
    print(f"Setting up processing for topic: {topic_name}")
    
    # Filter by topic and parse JSON
    topic_df = df.filter(col("topic") == topic_name) \
        .select(
            from_json(col("value"), schema).alias("data"),
            col("timestamp").alias("kafka_timestamp")
        ) \
        .select("data.*", "kafka_timestamp")
    
    # Convert timestamp if specified
    if timestamp_col:
        topic_df = topic_df.withColumn("event_time", to_timestamp(col(timestamp_col)))
    
    # Write to Delta Lake with unique checkpoint location
    query = topic_df.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", f"checkpoints/{topic_name}") \
        .trigger(processingTime="10 seconds") \
        .start(output_path)
    
    print(f"Started streaming query for {topic_name}")
    return query

# Start all streaming queries
print("Starting all streaming queries...")
queries = [
    process_topic("movie-stream", schemas["movie-stream"], "watch_time", "output/delta/stream_watches"),
    process_topic("movie-likes", schemas["movie-likes"], "liked_at", "output/delta/stream_likes"),
    process_topic("movie-reviews", schemas["movie-reviews"], "reviewed_at", "output/delta/stream_reviews")
]

print("All streaming queries started successfully!")
print("Streaming processor is running... Press Ctrl+C to stop")

# Keep all streams running
try:
    spark.streams.awaitAnyTermination()
except KeyboardInterrupt:
    print("\nStopping all streaming queries...")
    for query in queries:
        if query.isActive:
            query.stop()
    print("All queries stopped.")
    spark.stop()
    print("Spark session closed.")