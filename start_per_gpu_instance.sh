#!/bin/bash
# 脚本：每GPU一个独立进程，每进程多实例架构
# 基于run_multi_instance_local.py的改进版本

# Exit on any error
set -e

# 设置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cublas.lib; import nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ':' + nvidia.cudnn.lib.__path__[0])")

echo "🚀 启动每GPU独立进程多实例Whisper服务"
echo "基于run_multi_instance_local.py的架构改进"

# 配置参数
GPU_IDS="0,1,2"  # 使用GPU 0,1,2
INSTANCES_PER_PROCESS=4  # 每个进程4个实例（生产配置）
MODEL_SIZE="large-v3-turbo"  # 使用大模型
COMPUTE_TYPE="int8"
START_PORT=5002

# 解析GPU IDs
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "📋 配置信息:"
echo "   GPU IDs: ${GPU_ARRAY[@]}"
echo "   GPU数量: $NUM_GPUS"
echo "   每进程实例数: $INSTANCES_PER_PROCESS"
echo "   总实例数: $((NUM_GPUS * INSTANCES_PER_PROCESS))"
echo "   模型: $MODEL_SIZE"
echo "   计算类型: $COMPUTE_TYPE"

# 清理函数
cleanup() {
    echo ""
    echo "🛑 停止所有服务..."

    # 杀死所有后台进程
    for pid in "${PIDS[@]}"; do
        if [[ -n "$pid" ]]; then
            echo "   停止进程 $pid..."
            kill "$pid" 2>/dev/null || true
        fi
    done

    # 等待进程结束
    sleep 3

    # 强制杀死
    for pid in "${PIDS[@]}"; do
        if [[ -n "$pid" ]]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    echo "✅ 所有服务已停止"
    exit 0
}

# 捕获退出信号
trap cleanup EXIT INT TERM

# 存储进程ID和端口
PIDS=()
BACKEND_URLS=()

# 为每个GPU启动一个独立进程
echo ""
echo "🔧 启动GPU进程..."

for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    gpu_id=${GPU_ARRAY[$gpu_idx]}
    port=$((START_PORT + gpu_idx))

    echo "📍 启动GPU $gpu_id 进程 (端口: $port)..."

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export API_PORT=$port
    export GPU_DEVICE_ID=$gpu_id
    export WHISPER_MODEL=$MODEL_SIZE
    export WHISPER_COMPUTE_TYPE=$COMPUTE_TYPE
    export WHISPER_DEVICE="cuda"

    # 关键修改：设置每进程的实例数
    export NUM_WHISPER_INSTANCES=$INSTANCES_PER_PROCESS
    export MAX_QUEUE_SIZE=20
    export MAX_CONCURRENT_TASKS=$INSTANCES_PER_PROCESS

    export MAX_FILE_SIZE_MB=50
    export REQUEST_TIMEOUT=1800
    export LOG_LEVEL=INFO

    echo "   📊 GPU $gpu_id 配置:"
    echo "      - CUDA_VISIBLE_DEVICES=$gpu_id"
    echo "      - NUM_WHISPER_INSTANCES=$INSTANCES_PER_PROCESS"
    echo "      - MAX_CONCURRENT_TASKS=$MAX_CONCURRENT_TASKS"

    # 启动独立的多实例进程
    python run_multi_instance_local.py > "logs/gpu_${gpu_id}_process.log" 2>&1 &
    pid=$!

    PIDS+=($pid)
    BACKEND_URLS+=("http://localhost:$port")

    echo "✅ GPU $gpu_id 进程已启动 (PID: $pid, 端口: $port, $INSTANCES_PER_PROCESS 个实例)"

    # 短待初始化完成
    sleep 5

    # 检查进程状态
    if kill -0 "$pid" 2>/dev/null; then
        echo "   ✅ GPU $gpu_id 进程运行正常"
    else
        echo "   ❌ GPU $gpu_id 进程启动失败"
    fi
done

# 等待所有进程完成初始化
echo ""
echo "⏳ 等待所有进程初始化完成..."
sleep 60

# 检查进程状态
echo ""
echo "🔍 检查进程状态:"
for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    gpu_id=${GPU_ARRAY[$gpu_idx]}
    port=$((START_PORT + gpu_idx))

    # 检查健康状态
    if curl -s "http://localhost:$port/health" > /dev/null; then
        echo "✅ GPU $gpu_id (端口 $port): 健康"
    else
        echo "❌ GPU $gpu_id (端口 $port): 未响应"
    fi

    # 检查进程状态
    if kill -0 "${PIDS[$gpu_idx]}" 2>/dev/null; then
        echo "   ✅ GPU $gpu_id 进程运行正常"
    else
        echo "   ❌ GPU $gpu_id 进程已退出"
    fi
done

# 检查GPU内存使用
echo ""
echo "💾 GPU内存使用情况:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "📊 实例配置详情:"
for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    gpu_id=${GPU_ARRAY[$gpu_idx]}
    port=$((START_PORT + gpu_idx))

    echo "📍 GPU $gpu_id (端口 $port):"
    echo "   进程PID: ${PIDS[$gpu_idx]}"
    echo "   实例数: $INSTANCES_PER_PROCESS"
    echo "   日志: logs/gpu_${gpu_id}_process.log"
done

echo ""
echo "✅ 所有服务启动完成!"
echo ""
echo "📝 日志文件:"
for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    gpu_id=${GPU_ARRAY[$gpu_idx]}
    echo "   GPU $gpu_id: logs/gpu_${gpu_id}_process.log"
done
echo ""
echo "🌐 主服务地址: http://localhost:5001"
echo "🔍 健康检查: http://localhost:5001/health"
echo "📈 统计信息: http://localhost:5001/stats"
echo ""
echo "⚠️  按 Ctrl+C 停止所有服务"

# 等待所有后台进程
wait