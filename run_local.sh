#!/bin/bash

# Script for local API and Streamlit UI launch

echo "🚀 Starting local server..."

# Activate virtual environment
source venv/bin/activate

# Start FastAPI server in background
echo "📡 Starting FastAPI server on http://localhost:8000"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait a bit for API to start
sleep 3

# Start Streamlit
echo "🎨 Starting Streamlit UI on http://localhost:8501"
streamlit run app_ui.py

# Kill API process on exit
trap "kill $API_PID" EXIT
