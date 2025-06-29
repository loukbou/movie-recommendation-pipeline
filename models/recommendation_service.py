from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col
import os

class RecommendationService:
    def __init__(self, model_path="models/als_recommender"):
        self.model_path = model_path
        self.model = None
        self.spark = None
        self._initialize_spark()
        self._load_model()
    
    def _initialize_spark(self):
        """Initialize Spark session"""
        try:
            self.spark = SparkSession.builder \
                .appName("RecommendationService") \
                .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0,org.apache.spark:spark-mllib_2.12:3.5.2") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .getOrCreate()
            
            self.spark.sparkContext.setLogLevel("ERROR")  # Reduce logging
            print("✅ Spark session initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Spark: {e}")
            raise
    
    def _load_model(self):
        """Load the trained ALS model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train the model first.")
        
        try:
            self.model = ALSModel.load(self.model_path)
            print("✅ ALS model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def get_user_recommendations(self, user_id, num_recommendations=10):
        """
        Get movie recommendations for a specific user
        
        Args:
            user_id (int): User ID
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended movie IDs with ratings
        """
        try:
            # Create DataFrame with single user
            user_df = self.spark.createDataFrame([(user_id,)], ["user_id"])
            
            # Generate recommendations
            user_recs = self.model.recommendForUserSubset(user_df, num_recommendations)
            
            if user_recs.count() > 0:
                recommendations = user_recs.collect()[0]["recommendations"]
                return [{"movie_id": rec["movie_id"], "rating": float(rec["rating"])} 
                       for rec in recommendations]
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error generating recommendations for user {user_id}: {e}")
            return []
    
    def get_movie_recommendations_for_users(self, movie_id, num_users=10):
        """
        Get users who might be interested in a specific movie
        
        Args:
            movie_id (int): Movie ID
            num_users (int): Number of users to return
            
        Returns:
            list: List of user IDs with predicted ratings
        """
        try:
            # Create DataFrame with single movie
            movie_df = self.spark.createDataFrame([(movie_id,)], ["movie_id"])
            
            # Generate recommendations
            movie_recs = self.model.recommendForItemSubset(movie_df, num_users)
            
            if movie_recs.count() > 0:
                recommendations = movie_recs.collect()[0]["recommendations"]
                return [{"user_id": rec["user_id"], "rating": float(rec["rating"])} 
                       for rec in recommendations]
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error generating user recommendations for movie {movie_id}: {e}")
            return []
    
    def get_similar_users(self, user_id, num_similar=5):
        """
        Find users similar to the given user (based on model factors)
        This is a simplified approach - in practice, you'd want more sophisticated similarity
        """
        try:
            # Get user factors from the model
            user_factors = self.model.userFactors
            target_user = user_factors.filter(col("id") == user_id)
            
            if target_user.count() == 0:
                return []
            
            # This is a simplified similarity - you could implement cosine similarity here
            # For now, just return users with similar IDs (demo purposes)
            similar_users = []
            for i in range(1, num_similar + 1):
                similar_users.append({"user_id": user_id + i, "similarity": 0.8 - (i * 0.1)})
            
            return similar_users
            
        except Exception as e:
            print(f"❌ Error finding similar users for {user_id}: {e}")
            return []
    
    def get_popular_movies(self, num_movies=10):
        """
        Get most popular movies based on average predicted ratings
        """
        try:
            # Get all items and their average ratings
            all_items = self.model.recommendForAllItems(1)  # Get top user for each item
            
            # This is simplified - in practice you'd want to calculate actual popularity
            popular_movies = []
            items_data = all_items.collect()
            
            for item_row in items_data[:num_movies]:
                movie_id = item_row["movie_id"]
                # Use the rating as a proxy for popularity
                rating = item_row["recommendations"][0]["rating"] if item_row["recommendations"] else 0
                popular_movies.append({"movie_id": movie_id, "popularity_score": float(rating)})
            
            # Sort by popularity score
            popular_movies.sort(key=lambda x: x["popularity_score"], reverse=True)
            return popular_movies[:num_movies]
            
        except Exception as e:
            print(f"❌ Error getting popular movies: {e}")
            return []
    
    def close(self):
        """Close the Spark session"""
        if self.spark:
            self.spark.stop()
            print("🛑 Spark session closed")

