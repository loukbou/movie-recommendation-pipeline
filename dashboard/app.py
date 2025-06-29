import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add model path
sys.path.append(str(Path(__file__).parent.parent))
from models.recommendation_service import RecommendationService

st.set_page_config(page_title="🎥 Recommender Dashboard", layout="centered")
st.title("🎬 Movie Recommender System")

# 🔁 Load movie titles from preprocessed parquet
@st.cache_data
def load_movie_titles():
    try:
        df_movies = pd.read_parquet("data/processed/movies")
        # Normalize movieId to int
        df_movies["movieId"] = df_movies["movieId"].astype(int)
        return dict(zip(df_movies["movieId"], df_movies["title"]))
    except Exception as e:
        st.error(f"Failed to load movie titles: {e}")
        return {}

movie_id_to_title = load_movie_titles()

# 🚀 Load recommendation service
@st.cache_resource
def load_service():
    return RecommendationService()

service = load_service()

# 🔍 Sidebar selection
option = st.sidebar.radio("Select an action", [
    "🎯 Recommend movies to user",
    "👥 Recommend users for a movie",
    "🏆 Show popular movies",
    "🧍 Find similar users"
])

# 🎯 Recommend movies to a user
if option == "🎯 Recommend movies to user":
    user_id = st.number_input("Enter User ID", min_value=0, step=1)
    top_n = st.slider("Number of recommendations", 1, 20, 10)
    if st.button("Get Recommendations"):
        recs = service.get_user_recommendations(user_id, top_n)
        if recs:
            for r in recs:
                try:
                    movie_key = int(float(r["movie_id"]))
                    r["title"] = movie_id_to_title.get(movie_key, "❓ Unknown Title")
                except:
                    r["title"] = "❓ Unknown Title"
            st.success(f"Top {top_n} movie recommendations for user {user_id}:")
            st.table([{"Title": r["title"], "Predicted Rating": round(r["rating"], 2)} for r in recs])
        else:
            st.warning("No recommendations found.")

# 👥 Recommend users for a movie
elif option == "👥 Recommend users for a movie":
    movie_id = st.number_input("Enter Movie ID", min_value=0, step=1)
    top_n = st.slider("Number of users", 1, 20, 10)
    if st.button("Find Users"):
        users = service.get_movie_recommendations_for_users(movie_id, top_n)
        if users:
            movie_title = movie_id_to_title.get(int(movie_id), f"Movie {movie_id}")
            st.success(f"Top {top_n} users who may like '{movie_title}':")
            st.table([{"User ID": u["user_id"], "Predicted Rating": round(u["rating"], 2)} for u in users])
        else:
            st.warning("No user recommendations found.")

# 🏆 Show popular movies
elif option == "🏆 Show popular movies":
    top_n = st.slider("Number of popular movies", 1, 20, 10)
    if st.button("Show Popular"):
        popular = service.get_popular_movies(top_n)
        if popular:
            for p in popular:
                try:
                    movie_key = int(float(p["movie_id"]))
                    p["title"] = movie_id_to_title.get(movie_key, f"Movie {movie_key}")
                except:
                    p["title"] = f"Movie {p['movie_id']}"
            st.success(f"Top {top_n} most popular movies:")
            st.table([{"Title": p["title"], "Popularity Score": round(p["popularity_score"], 2)} for p in popular])
        else:
            st.warning("No popular movie data available.")

# 🧍 Similar users
elif option == "🧍 Find similar users":
    user_id = st.number_input("Enter User ID to find similar users", min_value=0, step=1)
    top_n = st.slider("Number of similar users", 1, 10, 5)
    if st.button("Find Similar Users"):
        similar = service.get_similar_users(user_id, top_n)
        if similar:
            st.success(f"Users similar to {user_id}:")
            st.table(similar)
        else:
            st.warning("No similar users found.")

# ❌ Spark session shutdown
if st.button("❌ Shutdown Spark"):
    service.close()
    st.success("Spark session closed.")
