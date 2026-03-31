#!/bin/bash
echo "Starting FastAPI backend on port 8000..."
uvicorn backend.api:app --reload --port 8000 &
BACKEND_PID=$!

echo "Starting Streamlit frontend..."
streamlit run app.py
FRONTEND_PID=$!

function cleanup {
    echo "Shutting down..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
}

trap cleanup EXIT
wait
