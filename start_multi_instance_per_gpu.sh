#!/bin/bash
# Script to start multiple faster-whisper instances per GPU with load balancing
# 每个GPU启动2个实例的简化版本

# Exit on any error
set -e
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cublas.lib; import nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ':' + nvidia.cudnn.lib.__path__[0])")

# 配置参数
INSTANCES_PER_GPU=${INSTANCES_PER_GPU:-2}  # 每GPU实例数，默认2个

# Function to detect available GPUs (复用原有逻辑)
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

    # Kill all GPU service instances
    echo "Stopping backend service instances..."
    for i in "${!GPU_PIDS[@]}"; do
        if [[ -n "${GPU_PIDS[$i]}" ]]; then
            echo "Stopping instance ${GPU_INSTANCE_IDS[$i]} (PID: ${GPU_PIDS[$i]})..."
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

# Arrays to store PIDs and information
declare -a GPU_PIDS
declare -a GPU_INSTANCE_IDS
BACKEND_URLS=()

# Start GPU service instances
echo "🚀 Starting $NUM_GPUS GPU(s) with $INSTANCES_PER_GPU instances per GPU..."
START_PORT=5002

for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
    for ((instance=0; instance<INSTANCES_PER_GPU; instance++)); do
        # Calculate port for this instance
        port=$((START_PORT + gpu_id * INSTANCES_PER_GPU + instance))
        instance_id="${gpu_id}_${instance}"

        # Set environment variables for this GPU service instance
        export CUDA_VISIBLE_DEVICES=$gpu_id
        export API_PORT=$port
        export GPU_DEVICE_ID=$gpu_id
        export INSTANCE_ID=$instance_id

        echo "Starting GPU $gpu_id instance $instance on port $port (Instance ID: $instance_id)..."

        # Start the service instance
        python3 faster_whisper_api.py &

        # Store PID and information
        pid=$!
        GPU_PIDS+=($pid)
        GPU_INSTANCE_IDS+=($instance_id)
        BACKEND_URLS+=("http://localhost:$port")

        echo "✅ Started GPU $gpu_id instance $instance on port $port (PID: $pid, Instance ID: $instance_id)"

        # Small delay between starting instances to avoid conflicts
        sleep 1
    done
done

# Wait a moment for all services to initialize
if [[ ${#GPU_PIDS[@]} -gt 0 ]]; then
    echo "⏳ Waiting for ${#GPU_PIDS[@]} service instances to initialize..."
    sleep 30
else
    echo "ℹ️  No GPU service instances to initialize."
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
echo "🔧 Backend service instances running:"
for ((i=0; i<${#GPU_PIDS[@]}; i++)); do
    echo "   - Instance ${GPU_INSTANCE_IDS[$i]}: Port ${BACKEND_URLS[$i]##*:} (PID: ${GPU_PIDS[$i]})"
done
echo ""
echo "⚙️  Configuration:"
echo "   - GPUs detected: $NUM_GPUS"
echo "   - Instances per GPU: $INSTANCES_PER_GPU"
echo "   - Total instances: ${#GPU_PIDS[@]}"
echo "   - Request timeout: 30 minutes (for large audio files)"
echo "   - Max queue size: 100 requests"
echo "   - Health check interval: 30 seconds"
echo ""
echo "💡 Service instances started with PIDs:"
echo "   - Load Balancer: $LB_PID"
for ((i=0; i<${#GPU_PIDS[@]}; i++)); do
    echo "   - Instance ${GPU_INSTANCE_IDS[$i]}: ${GPU_PIDS[$i]}"
done
echo ""
echo "📊 Service status:"
echo "   - Health endpoint: http://localhost:5001/health"
echo "   - Load balancer ready to accept requests"
echo ""
echo "🔍 Monitoring commands:"
echo "   - Check GPU utilization: nvidia-smi"
echo "   - Check service health: curl http://localhost:5001/health"
echo "   - View logs: docker logs <container_name>"
echo ""
echo "⚠️  Press Ctrl+C to stop all services"

# Wait for all background processes
wait