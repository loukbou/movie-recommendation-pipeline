from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("BatchIngestion").getOrCreate()

# === 1. Load MovieLens files ===
ratings = spark.read.option("delimiter", "::").option("header", False) \
    .csv("data/raw/ratings.dat") \
    .toDF("userId", "movieId", "rating", "timestamp")

movies = spark.read.option("delimiter", "::").option("header", False) \
    .csv("data/raw/movies.dat") \
    .toDF("movieId", "title", "genres")

# === 2. Load IMDb files ===
imdb_basics = spark.read.option("header", True).option("delimiter", "\t") \
    .csv("data/raw/title.basics.tsv")

imdb_ratings = spark.read.option("header", True).option("delimiter", "\t") \
    .csv("data/raw/title.ratings.tsv")

# === 3. Join IMDb datasets ===
imdb = imdb_basics.join(imdb_ratings, "tconst", "left") \
    .select("tconst", "primaryTitle", "startYear", "genres", "averageRating")

# === 4. Save MovieLens ratings and movie info ===
ratings.write.mode("overwrite").parquet("data/processed/ratings")
movies.write.mode("overwrite").parquet("data/processed/movies")
imdb.write.mode("overwrite").parquet("data/processed/imdb")

print("✅ Batch data processed and saved to /data/processed/")
