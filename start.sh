#!/bin/bash

# Function to handle cleanup
cleanup() {
    echo "Stopping services..."
    if [ -n "$BACKEND_PID" ]; then kill $BACKEND_PID; fi
    if [ -n "$FRONTEND_PID" ]; then kill $FRONTEND_PID; fi
    exit
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

# Start Backend
echo "Starting Backend..."
if [ -d "backend" ]; then
    cd backend
    # Use the project venv explicitly — a bare `python3` has none of the backend
    # packages, dies instantly in the background, and the error scrolls away.
    if [ -x "venv/bin/python" ]; then
        PY="venv/bin/python"
    elif [ -x ".venv/bin/python" ]; then
        PY=".venv/bin/python"
    else
        echo "ERROR: no venv found in backend/. Create it with:"
        echo "  cd backend && uv venv --python 3.11 venv && uv pip install -r requirements.txt -p venv/bin/python"
        exit 1
    fi
    if ! "$PY" -c "import uvicorn" 2>/dev/null; then
        echo "ERROR: uvicorn is not installed in backend/$PY. Install deps with:"
        echo "  cd backend && uv pip install -r requirements.txt -p venv/bin/python"
        exit 1
    fi
    "$PY" -m uvicorn main:app --reload &
    BACKEND_PID=$!
    cd ..
else
    echo "Error: backend directory not found"
    exit 1
fi

# Start Frontend
echo "Starting Frontend..."
if [ -d "frontend" ]; then
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
else
    echo "Error: frontend directory not found"
    cleanup
fi

echo "Both services started. Press Ctrl+C to stop."
wait
