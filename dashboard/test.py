import pandas as pd
import streamlit as st

st.title("🧪 Movie ID → Title Test")

try:
    df_movies = pd.read_parquet("data/processed/movies")
    st.write("✅ Successfully loaded movie metadata")

    # Show number of records
    st.write(f"Total records: {len(df_movies)}")
    
    # Print schema
    st.write("Columns:", df_movies.columns.tolist())

    # Show sample
    st.dataframe(df_movies.head(10))

    # Check for missing titles
    missing_titles = df_movies["title"].isnull().sum()
    st.write(f"🔍 Missing titles: {missing_titles}")

    # Check for unique movieIds with no duplicates
    duplicate_ids = df_movies["movieId"].duplicated().sum()
    st.write(f"❗ Duplicate movieId count: {duplicate_ids}")

    # print 10 random (id, title) pairs
    st.write("🎬 Sample movieId → title mappings:")
    st.dataframe(df_movies[["movieId", "title"]].dropna().sample(10))

except Exception as e:
    st.error(f"❌ Failed to load or process parquet file: {e}")
