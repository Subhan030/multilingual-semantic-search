#!/bin/bash
echo "Starting FastAPI backend on port 8000..."
uvicorn backend.api:app --port 8000 &
BACKEND_PID=$!

echo "Starting Streamlit frontend..."
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
FRONTEND_PID=$!

function cleanup {
    echo "Shutting down..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
}

trap cleanup EXIT
wait
