from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col, desc, asc
from pyspark.sql.types import StructType, StructField, IntegerType
import os

# Initialize Spark with Delta Lake configuration
builder = SparkSession.builder \
    .appName("TestRecommendations") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0,org.apache.spark:spark-mllib_2.12:3.5.2") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("Testing ALS Recommendation System")
print("=" * 50)

# Load the trained model
def load_model():
    model_path = "models/als_recommender"
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first by running:")
        print("   python train_asl.py")
        return None
    
    try:
        model = ALSModel.load(model_path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Load original data for context
def load_data():
    try:
        # Load watch data to understand user preferences
        watch_df = spark.read.format("delta").load("output/delta/stream_watches")
        
        # Create ratings data same as training
        ratings_df = watch_df.groupBy("user_id", "movie_id") \
            .agg({"*": "count"}) \
            .withColumnRenamed("count(1)", "watch_count") \
            .withColumn("user_id", col("user_id").cast("int")) \
            .withColumn("movie_id", col("movie_id").cast("int")) \
            .withColumn("watch_count", col("watch_count").cast("float"))
        
        print(f"📊 Loaded {ratings_df.count()} user-movie interactions")
        return watch_df, ratings_df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

# Show user's viewing history
def show_user_history(watch_df, ratings_df, user_id):
    print(f"\n👤 User {user_id}'s Viewing History:")
    print("-" * 30)
    
    # Show raw watch events
    user_watches = watch_df.filter(col("user_id") == str(user_id)) \
        .select("movie_id", "watch_time") \
        .orderBy("watch_time")
    
    if user_watches.count() > 0:
        user_watches.show(10, truncate=False)
        
        # Show aggregated ratings
        user_ratings = ratings_df.filter(col("user_id") == user_id) \
            .orderBy(desc("watch_count"))
        
        print(f"\n📈 User {user_id}'s Movie Preferences (by watch count):")
        user_ratings.show(10)
        
        return True
    else:
        print(f"No viewing history found for user {user_id}")
        return False

# Generate recommendations for a specific user
def recommend_for_user(model, user_id, num_recommendations=10):
    print(f"\n🎬 Top {num_recommendations} Recommendations for User {user_id}:")
    print("-" * 50)
    
    try:
        # Create DataFrame with single user
        user_df = spark.createDataFrame([(user_id,)], ["user_id"])
        
        # Generate recommendations
        user_recs = model.recommendForUserSubset(user_df, num_recommendations)
        
        if user_recs.count() > 0:
            # Extract recommendations
            recommendations = user_recs.collect()[0]["recommendations"]
            
            print("Movie ID | Predicted Rating")
            print("-" * 25)
            for i, rec in enumerate(recommendations, 1):
                movie_id = rec["movie_id"]
                rating = rec["rating"]
                print(f"{movie_id:8} | {rating:14.4f}")
            
            return recommendations
        else:
            print(f"No recommendations available for user {user_id}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        return None

# Generate recommendations for multiple users
def recommend_for_multiple_users(model, ratings_df, num_users=5, num_recommendations=5):
    print(f"\n🎯 Sample Recommendations for {num_users} Users:")
    print("=" * 60)
    
    # Get sample users with most activity
    active_users = ratings_df.groupBy("user_id") \
        .agg({"watch_count": "sum"}) \
        .withColumnRenamed("sum(watch_count)", "total_watches") \
        .orderBy(desc("total_watches")) \
        .limit(num_users)
    
    user_ids = [row["user_id"] for row in active_users.collect()]
    
    for user_id in user_ids:
        try:
            user_df = spark.createDataFrame([(user_id,)], ["user_id"])
            user_recs = model.recommendForUserSubset(user_df, num_recommendations)
            
            if user_recs.count() > 0:
                recommendations = user_recs.collect()[0]["recommendations"]
                print(f"\n👤 User {user_id}:")
                rec_str = ", ".join([f"Movie {rec['movie_id']} ({rec['rating']:.2f})" 
                                   for rec in recommendations])
                print(f"   {rec_str}")
        except Exception as e:
            print(f"   ❌ Error for user {user_id}: {e}")

# Generate movie recommendations (popular movies)
def recommend_movies_for_all_users(model, num_movies=10):
    print(f"\n🏆 Top {num_movies} Movies to Recommend:")
    print("-" * 40)
    
    try:
        # Get all movies and recommend them to all users
        all_movies = model.recommendForAllItems(num_movies)
        
        if all_movies.count() > 0:
            print("Movie ID | Avg Predicted Rating")
            print("-" * 30)
            
            movies_data = all_movies.collect()
            for movie_row in movies_data[:num_movies]:
                movie_id = movie_row["movie_id"]
                recommendations = movie_row["recommendations"]
                avg_rating = sum([rec["rating"] for rec in recommendations]) / len(recommendations)
                print(f"{movie_id:8} | {avg_rating:16.4f}")
                
    except Exception as e:
        print(f"❌ Error generating movie recommendations: {e}")

# Interactive testing function
def interactive_test(model, watch_df, ratings_df):
    print("\n🔍 Interactive Testing Mode")
    print("Available commands:")
    print("  1. Enter user ID to get recommendations")
    print("  2. Type 'users' to see available user IDs")
    print("  3. Type 'quit' to exit")
    
    # Show available users
    available_users = ratings_df.select("user_id").distinct().orderBy("user_id").collect()
    user_ids = [row["user_id"] for row in available_users]
    print(f"\nAvailable user IDs: {user_ids[:10]}{'...' if len(user_ids) > 10 else ''}")
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip().lower()
            
            if user_input == 'quit':
                break
            elif user_input == 'users':
                print(f"All available user IDs: {user_ids}")
            else:
                try:
                    user_id = int(user_input)
                    if user_id in user_ids:
                        show_user_history(watch_df, ratings_df, user_id)
                        recommend_for_user(model, user_id)
                    else:
                        print(f"User {user_id} not found in dataset")
                except ValueError:
                    print("Please enter a valid user ID or command")
                    
        except KeyboardInterrupt:
            break
    
    print("\n👋 Interactive testing completed!")

# Main testing function
def main():
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Load data
    watch_df, ratings_df = load_data()
    if watch_df is None or ratings_df is None:
        return
    
    print(f"\n📊 Dataset Overview:")
    print(f"Total users: {ratings_df.select('user_id').distinct().count()}")
    print(f"Total movies: {ratings_df.select('movie_id').distinct().count()}")
    print(f"Total interactions: {ratings_df.count()}")
    
    # Run different types of tests
    try:
        # Test 1: Multiple users recommendations
        recommend_for_multiple_users(model, ratings_df)
        
        # Test 2: Single user detailed test
        sample_user = ratings_df.select("user_id").first()["user_id"]
        show_user_history(watch_df, ratings_df, sample_user)
        recommend_for_user(model, sample_user)
        
        # Test 3: Popular movies
        recommend_movies_for_all_users(model)
        
        # Test 4: Interactive mode (optional)
        interactive_choice = input("\n🤔 Do you want to enter interactive testing mode? (y/n): ").strip().lower()
        if interactive_choice == 'y':
            interactive_test(model, watch_df, ratings_df)
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    
    print("\n🎉 Recommendation testing completed!")

if __name__ == "__main__":
    try:
        main()
    finally:
        spark.stop()
        print("🛑 Spark session closed")