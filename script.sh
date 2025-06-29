#!/bin/bash

# Setup script for Delta Lake environment
echo "Setting up Delta Lake environment..."

# Install required packages
pip install delta-spark==3.2.0
pip install pyspark==3.5.2

# Set environment variables
export PYSPARK_SUBMIT_ARGS="--packages io.delta:delta-spark_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2 pyspark-shell"

# Alternative: Use spark-submit instead of python
# spark-submit \
#   --packages io.delta:delta-spark_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2 \
#   stream_processor.py

echo "Environment setup complete!"
