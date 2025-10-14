#!/bin/bash
# Script to dynamically detect GPUs and start faster-whisper services with load balancing

# Exit on any error
set -e
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cublas.lib; import nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ':' + nvidia.cudnn.lib.__path__[0])")

# Function to detect available GPUs
detect_gpus() {
    echo "🔍 Detecting available GPUs..."


    # Use Python script for robust GPU detection
    if [[ -f "/app/docker/detect_gpus.py" ]]; then
        GPU_COUNT=$(python3 /app/docker/detect_gpus.py 2>/dev/null)
        if [[ $? -eq 0 && -n "$GPU_COUNT" ]]; then
            echo "✅ Detected $GPU_COUNT available GPU(s) using Python detection"
            echo $GPU_COUNT
            return
        fi
    fi

    # Fallback: Try to detect GPUs using nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        # Get the count of available GPUs
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -n 1)
        if [[ $? -eq 0 && -n "$GPU_COUNT" ]]; then
            echo "✅ Detected $GPU_COUNT available GPU(s) using nvidia-smi"
            echo $GPU_COUNT
            return
        fi
    fi

    # Fallback: Check CUDA_VISIBLE_DEVICES environment variable
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        # Count the number of devices in the comma-separated list
        IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
        GPU_COUNT=${#DEVICES[@]}
        echo "✅ Detected $GPU_COUNT GPU(s) from CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        echo $GPU_COUNT
        return
    fi

    # Fallback: Check for NVIDIA environment variables
    if [[ -n "$NVIDIA_VISIBLE_DEVICES" ]]; then
        if [[ "$NVIDIA_VISIBLE_DEVICES" == "all" ]]; then
            # Try to count available GPUs
            GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
            if [[ $GPU_COUNT -gt 0 ]]; then
                echo "✅ Detected $GPU_COUNT GPU(s) from NVIDIA_VISIBLE_DEVICES=all"
                echo $GPU_COUNT
                return
            fi
        else
            # Count the number of devices in the comma-separated list
            IFS=',' read -ra DEVICES <<< "$NVIDIA_VISIBLE_DEVICES"
            GPU_COUNT=${#DEVICES[@]}
            echo "✅ Detected $GPU_COUNT GPU(s) from NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
            echo $GPU_COUNT
            return
        fi
    fi

    # Final fallback: Default to 1 GPU
    echo "⚠️  Could not detect GPU count, defaulting to 1 GPU"
    echo "1"
}

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

# Detect number of GPUs
NUM_GPUS=$(detect_gpus)

# Extract just the numeric value from the output (removing any extra text)
NUM_GPUS=$(echo "$NUM_GPUS" | grep -o '[0-9]*$' | head -1)

# Validate GPU count
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "❌ Invalid GPU count detected: $NUM_GPUS. Using 1 GPU."
    NUM_GPUS=1
fi

# Arrays to store PIDs
declare -a GPU_PIDS
BACKEND_URLS=()

# Start GPU services on different ports
echo "🚀 Starting $NUM_GPUS GPU service instance(s)..."
START_PORT=5002

for ((i=0; i<NUM_GPUS; i++)); do
    # Set environment variables for this GPU service
    export CUDA_VISIBLE_DEVICES=$i
    export API_PORT=$((START_PORT + i))
    export GPU_DEVICE_ID=$i

    echo "Starting GPU $i service on port $API_PORT..."
    python3 faster_whisper_api.py &
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

python3 load_balancer.py &
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