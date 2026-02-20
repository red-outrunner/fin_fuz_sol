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
    python3 -m uvicorn main:app --reload &
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
