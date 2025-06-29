from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, count, when, isnan, isnull
import os

# Initialize Spark with updated Delta Lake configuration
builder = SparkSession.builder \
    .appName("ALSRecommenderTraining") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0,org.apache.spark:spark-mllib_2.12:3.5.2") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("Spark session initialized for ALS training!")

# Check if Delta tables exist
def check_data_availability():
    paths = [
        "output/delta/stream_watches",
        "output/delta/stream_likes", 
        "output/delta/stream_reviews"
    ]
    
    available_paths = []
    for path in paths:
        try:
            # Try to read the delta table
            df = spark.read.format("delta").load(path)
            count = df.count()
            print(f"✅ {path}: {count} records")
            available_paths.append(path)
        except Exception as e:
            print(f"❌ {path}: Not available - {str(e)}")
            # Check if directory exists but is empty/not delta
            if os.path.exists(path):
                print(f"   Directory exists but may not be a valid Delta table")
            else:
                print(f"   Directory does not exist")
    
    return available_paths

# Load and prepare data
def load_and_prepare_data():
    available_paths = check_data_availability()
    
    if "output/delta/stream_watches" not in available_paths:
        print("❌ No watch data available. Options:")
        print("1. Make sure your streaming processor is running")
        print("2. Generate some test data")
        print("3. Use sample data for testing")
        
        # Generate sample data for testing
        print("🔄 Generating sample data for testing...")
        sample_data = [
            ("1", "101", "2024-01-01 10:00:00"),
            ("1", "102", "2024-01-01 11:00:00"), 
            ("1", "101", "2024-01-01 12:00:00"),  # User 1 watched movie 101 twice
            ("2", "101", "2024-01-01 13:00:00"),
            ("2", "103", "2024-01-01 14:00:00"),
            ("3", "102", "2024-01-01 15:00:00"),
            ("3", "103", "2024-01-01 16:00:00"),
            ("4", "101", "2024-01-01 17:00:00"),
            ("4", "102", "2024-01-01 18:00:00"),
            ("4", "103", "2024-01-01 19:00:00"),
        ]
        
        from pyspark.sql.types import StructType, StructField, StringType
        schema = StructType([
            StructField("user_id", StringType(), True),
            StructField("movie_id", StringType(), True), 
            StructField("watch_time", StringType(), True)
        ])
        
        watch_df = spark.createDataFrame(sample_data, schema)
        print(f"✅ Created sample dataset with {watch_df.count()} records")
    else:
        # Load actual streaming data
        watch_df = spark.read.format("delta").load("output/delta/stream_watches")
        print(f"✅ Loaded streaming data with {watch_df.count()} records")
    
    # Show sample of data
    print("📊 Sample watch data:")
    watch_df.show(10)
    
    # Convert implicit feedback (watch counts as ratings)
    print("🔄 Aggregating watch counts...")
    ratings_df = watch_df.groupBy("user_id", "movie_id") \
        .agg(count("*").alias("watch_count"))
    
    print("📊 Sample ratings data:")
    ratings_df.show(10)
    
    # Check for data quality
    total_ratings = ratings_df.count()
    unique_users = ratings_df.select("user_id").distinct().count()
    unique_movies = ratings_df.select("movie_id").distinct().count()
    
    print(f"📈 Data Summary:")
    print(f"   Total ratings: {total_ratings}")
    print(f"   Unique users: {unique_users}")
    print(f"   Unique movies: {unique_movies}")
    
    if total_ratings < 10:
        print("⚠️  Warning: Very small dataset. Consider generating more sample data.")
    
    # ALS requires integer IDs - ensure proper casting
    ratings_df = ratings_df \
        .withColumn("user_id", col("user_id").cast("int")) \
        .withColumn("movie_id", col("movie_id").cast("int")) \
        .withColumn("watch_count", col("watch_count").cast("float"))
    
    # Remove any null values
    ratings_df = ratings_df.filter(
        col("user_id").isNotNull() & 
        col("movie_id").isNotNull() & 
        col("watch_count").isNotNull()
    )
    
    final_count = ratings_df.count()
    print(f"✅ Final clean dataset: {final_count} ratings")
    
    return ratings_df

# Load data
try:
    ratings_df = load_and_prepare_data()
    
    if ratings_df.count() == 0:
        print("❌ No data available for training. Exiting.")
        spark.stop()
        exit(1)
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
    spark.stop()
    exit(1)

# Split into training and test sets
print("🔄 Splitting data into train/test sets...")
train, test = ratings_df.randomSplit([0.8, 0.2], seed=42)

print(f"📊 Training set: {train.count()} ratings")
print(f"📊 Test set: {test.count()} ratings")

# Configure and train ALS model
def train_als_model(train_data):
    print("🚀 Training ALS model...")
    als = ALS(
        userCol="user_id",
        itemCol="movie_id", 
        ratingCol="watch_count",
        coldStartStrategy="drop",
        implicitPrefs=True,  # Using watch counts as implicit feedback
        rank=10,            # Number of latent factors
        maxIter=10,
        regParam=0.1,
        alpha=1.0           # Confidence parameter for implicit feedback
    )
    return als.fit(train_data)

try:
    model = train_als_model(train)
    print("✅ Model training completed!")
except Exception as e:
    print(f"❌ Error training model: {e}")
    spark.stop()
    exit(1)

# Evaluate model
def evaluate_model(model, test_data):
    print("🔄 Evaluating model...")
    
    if test_data.count() == 0:
        print("⚠️  No test data available for evaluation")
        return None
        
    predictions = model.transform(test_data)
    
    # Filter out NaN predictions (cold start items)
    predictions = predictions.filter(
        ~isnan(col("prediction")) & 
        ~isnull(col("prediction"))
    )
    
    pred_count = predictions.count()
    print(f"📊 Valid predictions: {pred_count}")
    
    if pred_count == 0:
        print("⚠️  No valid predictions for evaluation")
        return None
    
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="watch_count", 
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"✅ RMSE: {rmse:.4f}")
    
    # Show sample predictions
    print("📊 Sample predictions:")
    predictions.orderBy(col("prediction").desc()).show(10)
    
    return rmse

# Evaluate model
rmse = evaluate_model(model, test)

# Save model 
try:
    print("💾 Saving model...")
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Remove existing model if it exists
    model_path = "models/als_recommender"
    if os.path.exists(model_path):
        import shutil
        shutil.rmtree(model_path)
        print("🗑️  Removed existing model")
    
    model.save(model_path)
    print("✅ Model saved successfully!")
    
except Exception as e:
    print(f"❌ Error saving model: {e}")

print("=" * 50)
print("Model training complete!")
print("=" * 50)


spark.stop()
print("🛑 Spark session closed")