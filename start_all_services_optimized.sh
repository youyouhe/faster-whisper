#!/bin/bash
# Script to start faster-whisper services with load balancing
# Optimized to dynamically ask user for number of GPUs

# Exit on any error
set -e

# Activate virtual environment
source faster-whisper-env/bin/activate

# Set LD_LIBRARY_PATH for CUDA
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cublas.lib; import nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ':' + nvidia.cudnn.lib.__path__[0])")

# Function to clean up processes on exit
cleanup() {
    echo ""
    echo "Stopping all services..."
    
    # Kill load balancer
    if [[ -n "$LB_PID" ]]; then
        echo "Stopping load balancer (PID: $LB_PID)..."
        kill $LB_PID 2>/dev/null || true
    fi
    
    # Kill GPU services
    echo "Stopping backend services..."
    for i in "${!GPU_PIDS[@]}"; do
        if [[ -n "${GPU_PIDS[$i]}" ]]; then
            echo "Stopping GPU $i service (PID: ${GPU_PIDS[$i]})..."
            kill ${GPU_PIDS[$i]} 2>/dev/null || true
        fi
    done
    
    # Wait a moment for processes to exit
    sleep 2
    
    # Force kill if still running
    if [[ -n "$LB_PID" ]]; then
        kill -9 $LB_PID 2>/dev/null || true
    fi
    
    for i in "${!GPU_PIDS[@]}"; do
        if [[ -n "${GPU_PIDS[$i]}" ]]; then
            kill -9 ${GPU_PIDS[$i]} 2>/dev/null || true
        fi
    done
    
    echo "All services stopped."
    exit
}

# Trap exit signals to clean up
trap cleanup EXIT INT TERM

# Ask user for number of GPUs
echo "🔍 Detecting available GPUs..."
nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -n 1 > /tmp/gpu_count.txt || echo "0" > /tmp/gpu_count.txt
AVAILABLE_GPUS=$(cat /tmp/gpu_count.txt)
rm /tmp/gpu_count.txt

if [[ $AVAILABLE_GPUS -eq 0 ]]; then
    echo "⚠️  No GPUs detected. Please ensure NVIDIA drivers and nvidia-smi are properly installed."
    read -p "Enter number of GPU services to start (or 0 to use CPU only): " NUM_GPUS
else
    echo "✅ Detected $AVAILABLE_GPUS available GPU(s)."
    read -p "Enter number of GPU services to start (0-$AVAILABLE_GPUS) [default: $AVAILABLE_GPUS]: " NUM_GPUS
    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS=$AVAILABLE_GPUS
    fi
fi

# Validate input
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 0 ]] || [[ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]]; then
    echo "❌ Invalid input. Using all available GPUs: $AVAILABLE_GPUS"
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Arrays to store PIDs
declare -a GPU_PIDS
BACKEND_URLS=()

# Start GPU services on different ports
echo "🚀 Starting $NUM_GPUS GPU service instance(s)..."
START_PORT=5002

for ((i=0; i<NUM_GPUS; i++)); do
    export CUDA_VISIBLE_DEVICES=$i
    export API_PORT=$((START_PORT + i))
    export GPU_DEVICE_ID=$i
    
    echo "Starting GPU $i service on port $API_PORT..."
    python faster_whisper_api.py &
    GPU_PIDS[$i]=$!
    BACKEND_URLS+=("http://localhost:$API_PORT")
    echo "✅ Started GPU $i service on port $API_PORT (PID: ${GPU_PIDS[$i]})"
done

# Wait a moment for services to initialize
if [[ $NUM_GPUS -gt 0 ]]; then
    echo "⏳ Waiting for GPU services to initialize..."
    sleep 30
else
    echo "ℹ️  No GPU services to initialize."
fi

# Start load balancer on port 5001
echo "🔄 Starting load balancer on port 5001"
export LB_PORT=5001
export BACKEND_SERVICES=$(IFS=,; echo "${BACKEND_URLS[*]}")
export REQUEST_TIMEOUT=1800  # 30 minutes for large audio files
export MAX_QUEUE_SIZE=100    # Maximum requests in queue
export HEALTH_CHECK_INTERVAL=30  # Health check interval

python load_balancer.py &
LB_PID=$!
echo "✅ Started load balancer on port 5001 (PID: $LB_PID)"

echo ""
echo "✅ All services started successfully!"
echo "🌐 Load balancer running on http://localhost:5001"
if [[ $NUM_GPUS -gt 0 ]]; then
    echo "🔧 Backend services running on ports $START_PORT-$((START_PORT + NUM_GPUS - 1))"
else
    echo "🔧 No GPU backend services running (CPU mode)"
fi
echo "⚙️  Configuration:"
echo "   - Request timeout: 30 minutes (for large audio files)"
echo "   - Max queue size: 100 requests"
echo "   - Health check interval: 30 seconds"
echo ""
echo "💡 Services started with PIDs:"
echo "   - Load Balancer: $LB_PID"
for ((i=0; i<NUM_GPUS; i++)); do
    echo "   - Backend $((START_PORT + i)): ${GPU_PIDS[$i]}"
done
echo ""
echo "📊 Service status:"
echo "   - Health endpoint: http://localhost:5001/health"
echo "   - Load balancer ready to accept requests"
echo ""
echo "⚠️  Press Ctrl+C to stop all services"

# Wait for all background processes
wait