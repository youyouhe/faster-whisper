#!/bin/bash
# Script to start faster-whisper services with load balancing

# Exit on any error
set -e

# Function to clean up processes on exit
cleanup() {
    echo ""
    echo "Stopping all services..."
    echo "Stopping load balancer (PID: $LB_PID)..."
    kill $LB_PID 2>/dev/null || true
    
    echo "Stopping backend services..."
    kill $GPU0_PID $GPU1_PID $GPU2_PID 2>/dev/null || true
    
    # Wait a moment for processes to exit
    sleep 2
    
    # Force kill if still running
    kill -9 $LB_PID $GPU0_PID $GPU1_PID $GPU2_PID 2>/dev/null || true
    
    echo "All services stopped."
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
GPU0_PID=$!
echo "Started GPU 0 service on port 5002 (PID: $GPU0_PID)"

# GPU 1 on port 5003
export CUDA_VISIBLE_DEVICES=1
export API_PORT=5003
export GPU_DEVICE_ID=1
python faster_whisper_api.py &
GPU1_PID=$!
echo "Started GPU 1 service on port 5003 (PID: $GPU1_PID)"

# GPU 2 on port 5004
export CUDA_VISIBLE_DEVICES=2
export API_PORT=5004
export GPU_DEVICE_ID=2
python faster_whisper_api.py &
GPU2_PID=$!
echo "Started GPU 2 service on port 5004 (PID: $GPU2_PID)"

# Note: GPU 3 (GTX 1050 Ti) is not supported due to compute capability limitations
echo "Note: GPU 3 (GTX 1050 Ti) is not supported due to compute capability limitations"

# Wait a moment for services to initialize
sleep 30

# Start load balancer on port 5001
echo "Starting load balancer on port 5001"
export LB_PORT=5001
export BACKEND_SERVICES="http://localhost:5002,http://localhost:5003,http://localhost:5004"
export REQUEST_TIMEOUT=1800  # 30 minutes for large audio files
export MAX_QUEUE_SIZE=100  # Maximum requests in queue
export HEALTH_CHECK_INTERVAL=30  # Health check interval
python load_balancer.py &
LB_PID=$!
echo "Started load balancer on port 5001 (PID: $LB_PID)"

echo ""
echo "✅ All services started successfully!"
echo "🌐 Load balancer running on http://localhost:5001"
echo "🔧 Backend services running on ports 5002-5004"
echo "⚙️  Configuration:"
echo "   - Request timeout: 30 minutes (for large audio files)"
echo "   - Max queue size: 100 requests"
echo "   - Health check interval: 30 seconds"
echo ""
echo "💡 Services started with PIDs:"
echo "   - Load Balancer: $LB_PID"
echo "   - Backend 5002: $GPU0_PID"
echo "   - Backend 5003: $GPU1_PID"
echo "   - Backend 5004: $GPU2_PID"
echo ""
echo "📊 Service status:"
echo "   - Health endpoint: http://localhost:5001/health"
echo "   - Load balancer ready to accept requests"
echo ""
echo "⚠️  Press Ctrl+C to stop all services"

# Wait for all background processes
wait
