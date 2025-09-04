#!/bin/bash
# Script to start faster-whisper services with load balancing

# Exit on any error
set -e

# Function to clean up processes on exit
cleanup() {
    echo "Stopping all services..."
    kill $(jobs -p) 2>/dev/null || true
    exit
}

# Trap exit signals to clean up
trap cleanup EXIT INT TERM

# Start 3 GPU services on different ports (GTX 1080 only)
echo "Starting GPU service instances..."

# GPU 0 on port 5002
export CUDA_VISIBLE_DEVICES=0
export API_PORT=5002
export GPU_DEVICE_ID=0
python faster_whisper_api.py &
echo "Started GPU 0 service on port 5002"

# GPU 1 on port 5003
export CUDA_VISIBLE_DEVICES=1
export API_PORT=5003
export GPU_DEVICE_ID=1
python faster_whisper_api.py &
echo "Started GPU 1 service on port 5003"

# GPU 2 on port 5004
export CUDA_VISIBLE_DEVICES=2
export API_PORT=5004
export GPU_DEVICE_ID=2
python faster_whisper_api.py &
echo "Started GPU 2 service on port 5004"

# Note: GPU 3 (GTX 1050 Ti) is not supported due to compute capability limitations
echo "Note: GPU 3 (GTX 1050 Ti) is not supported due to compute capability limitations"

# Wait a moment for services to initialize
sleep 30

# Start load balancer on port 5001
echo "Starting load balancer on port 5001"
export LB_PORT=5001
export BACKEND_SERVICES="http://localhost:5002,http://localhost:5003,http://localhost:5004"
python load_balancer.py &

echo "All services started successfully!"
echo "Load balancer running on http://localhost:5001"
echo "Backend services running on ports 5002-5004"
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait