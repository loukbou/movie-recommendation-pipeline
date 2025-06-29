from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime
import pandas as pd

# Load movie IDs from the batch parquet file
try:
    df_movies = pd.read_parquet("data/processed/movies")
    movie_ids = df_movies['movieId'].dropna().astype(str).unique().tolist()
except Exception as e:
    print("❌ Failed to load movie IDs from data/processed/movies:")
    print(e)
    exit(1)

# Sample review texts
sample_reviews = [
    "Fantastic!", "Meh...", "I loved it", "Could be better", 
    "Too long", "A masterpiece", "Not my type", "Brilliant cinematography"
]

# Create Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print(f"✅ Kafka review producer started. Loaded {len(movie_ids)} movie IDs.")

# Simulate reviews
try:
    while True:
        event = {
            "user_id": str(random.randint(1000, 1100)),
            "movie_id": random.choice(movie_ids),
            "text": random.choice(sample_reviews),
            "reviewed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        producer.send("movie-reviews", value=event)
        print(f"📝 Sent review: {event}")
        time.sleep(random.uniform(2.5, 5.0))

except KeyboardInterrupt:
    print("\n🛑 Review producer manually stopped.")
