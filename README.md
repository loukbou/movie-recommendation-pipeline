# Movie Recommendation Pipeline

This project implements a **real-time movie recommendation system** using **Apache Spark**, **Kafka**, **Delta Lake**, and **Streamlit**. It simulates a real-world streaming architecture for ingesting user events (views, likes, reviews), processes them in real time, trains an ALS (collaborative filtering) recommender, and provides interactive dashboards for monitoring and recommendation exploration.

---

## Features

* Real-time ingestion of simulated user events using Kafka
* Spark Structured Streaming with Delta Lake for durable, queryable storage
* ALS-based recommendation engine using Spark MLlib
* Streamlit dashboards for:

  * Top trending movies
  * Interactive movie and user recommendations
* Modular architecture (producers, stream processors, batch training, dashboards)

---

## Architecture Overview

```
Kafka (multi-topic)
   ├── movie-stream (view events)
   ├── movie-likes
   └── movie-reviews
        ↓
  Spark Streaming → Delta Lake tables
        ↓
     ALS Model Training
        ↓
   Streamlit Dashboard (interactive)
```

---

## Simulated Streaming Data

Since we do not have access to real production logs or live user interactions, the project simulates real-time data ingestion by using custom Kafka producers. These producers generate **fake but realistic events** for:

* **Movie views** (`movie-stream`)
* **Likes** (`movie-likes`)
* **Reviews** (`movie-reviews`)

Producers use a real list of movies (loaded from preprocessed batch files) and emit events at regular intervals.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:loukbou/movie-recommendation-pipeline.git
cd movie-recommendation-pipeline
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Kafka & Zookeeper (Docker)

```bash
docker-compose up -d
```

> Make sure Kafka is running on `localhost:9092`.

### 4. Simulate Streaming Data

```bash
python streaming/producer_watch.py
python streaming/producer_likes.py
python streaming/producer_reviews.py
```

Each script will send events to a Kafka topic.

### 5. Run Spark Stream Processor

```bash
spark-submit streaming/stream_processor.py
```

This script reads from Kafka and writes each topic to Delta Lake.

### 6. Train the ALS Recommendation Model

```bash
spark-submit models/train_als.py
```

This reads Delta data, preprocesses ratings, and trains an ALS model saved under `models/als_recommender`.

### 7. Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Explore recommendations, top movies, user similarity, and more.

---

## Project Structure

```
.
├── data/
│   └── processed/            # Cleaned movie data for simulation
├── models/
│   ├── train_als.py          # ALS model training script
│   ├── recommendation_service.py  # Model wrapper used in dashboard
├── streaming/
│   ├── producer_watch.py     # Kafka producer for view events
│   ├── producer_likes.py     # Kafka producer for likes
│   ├── producer_reviews.py   # Kafka producer for reviews
│   └── stream_processor.py   # Spark Structured Streaming logic
├── dashboard/
│   └── app.py                # Streamlit app for recommendations
├── output/
│   └── delta/                # Delta Lake output folders
├── checkpoints/              # Spark streaming checkpoint state
├── requirements.txt
└── README.md
```

---

## Technologies Used

* **Apache Kafka** for real-time ingestion
* **Apache Spark (Structured Streaming + MLlib)** for real-time ETL and collaborative filtering
* **Delta Lake** for ACID-compliant storage and time-travel
* **Streamlit** for real-time interactive dashboards
* **Python** and **Pandas** for producers and integration

---

## Notes

* Checkpoints are automatically handled by Spark to ensure **exactly-once** processing semantics and recovery on failure.
* Movie metadata (`data/processed/movies`) must include `movieId` and `title` for dashboard mapping.
* The model and pipeline are designed for **local simulation**, but the architecture is extendable to production-grade clusters.

