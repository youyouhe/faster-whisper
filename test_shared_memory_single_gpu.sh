#!/bin/bash
# 单GPU Shared Memory架构测试脚本

# Exit on any error
set -e

# 配置 - 单GPU测试
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-5002}
WORKERS=${WORKERS:-2}
MEMORY_POOL_SIZE_MB=${MEMORY_POOL_SIZE_MB:-100}
MEMORY_CHUNK_SIZE_MB=${MEMORY_CHUNK_SIZE_MB:-25}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}ℹ️  $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_debug() {
    echo -e "${BLUE}🔍 $1${NC}"
}

log_header() {
    echo -e "${CYAN}=== $1 ===${NC}"
}

# 创建日志目录
mkdir -p logs

# 函数：清理进程
cleanup() {
    echo ""
    log_header "🛑 Stopping Shared Memory Architecture"

    # 杀死master进程
    if [[ -n "$MASTER_PID" ]]; then
        log_info "Stopping master process (PID: $MASTER_PID)..."
        kill $MASTER_PID 2>/dev/null || true
    fi

    # 等待进程退出
    sleep 3

    # 强制杀死残留进程
    if [[ -n "$MASTER_PID" ]]; then
        kill -9 $MASTER_PID 2>/dev/null || true
    fi

    # 清理共享内存
    log_info "🧹 Cleaning up shared memory..."
    python3 -c "
import sys
sys.path.append('.')
from shared_memory_manager import SharedMemoryPool
try:
    pool = SharedMemoryPool(gpu_id=$GPU_ID)
    pool.cleanup()
    print(f'✅ Shared memory for GPU $GPU_ID cleaned up')
except Exception as e:
    print(f'Error cleaning shared memory: {e}')
"

    log_info "✅ Cleanup complete"
    exit 0
}

# Trap退出信号
trap cleanup EXIT INT TERM

# 设置环境变量
export GPU_ID=$GPU_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MEMORY_POOL_SIZE_MB=$MEMORY_POOL_SIZE_MB
export MEMORY_CHUNK_SIZE_MB=$MEMORY_CHUNK_SIZE_MB
export WORKERS_PER_GPU=$WORKERS

log_header "🚀 Single GPU Shared Memory Test"
echo "  🎯 GPU ID: $GPU_ID"
echo "  🚪 Port: $PORT"
echo "  👥 Workers: $WORKERS"
echo "  💾 Memory Pool: ${MEMORY_POOL_SIZE_MB}MB"
echo "  📦 Chunk Size: ${MEMORY_CHUNK_SIZE_MB}MB"
echo ""

# 检查GPU可用性
log_info "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits)
    if [[ -n "$gpu_info" ]]; then
        echo "GPU Info:"
        echo "$gpu_info" | while IFS=',' read -r index name total_mem used_mem; do
            echo "  GPU $index: $name"
            echo "  Memory: ${used_mem}MB / ${total_mem}MB"
        done
    else
        log_warn "⚠️  Could not get GPU info"
    fi
else
    log_warn "⚠️  nvidia-smi not found, proceeding without GPU info"
fi

# 启动master进程
log_header "🚀 Starting Master Process"
log_info "Starting master process for GPU $GPU_ID on port $PORT..."

python3 master_process.py \
    --port $PORT \
    --gpu-id $GPU_ID \
    --workers $WORKERS \
    > logs/master_gpu_${GPU_ID}.log 2>&1 &

MASTER_PID=$!
log_info "✅ Started master process (PID: $MASTER_PID, Port: $PORT)"

# 等待初始化完成
log_info "⏳ Waiting for service to initialize..."
sleep 20

# 检查服务状态
log_header "🔍 Service Health Check"

max_retries=30
retry_count=0
while [[ $retry_count -lt $max_retries ]]; do
    if curl -f "http://localhost:$PORT/health" > /dev/null 2>&1; then
        log_info "✅ Service is healthy!"
        break
    else
        retry_count=$((retry_count + 1))
        log_debug "Health check attempt $retry_count/$max_retries failed, retrying in 2 seconds..."
        sleep 2
    fi
done

if [[ $retry_count -eq $max_retries ]]; then
    log_error "❌ Service failed to start properly"
    log_info "Checking logs:"
    echo "   tail -f logs/master_gpu_${GPU_ID}.log"
    exit 1
fi

# 获取统计信息
log_header "📊 Service Statistics"
stats=$(curl -s "http://localhost:$PORT/stats" 2>/dev/null)

if [[ -n "$stats" ]]; then
    echo "Master Process Stats:"
    echo "$stats" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)

    # Master进程统计
    master = data.get('master_process', {})
    if master:
        print(f'  GPU ID: {master.get(\"gpu_id\", \"unknown\")}')
        print(f'  Port: {master.get(\"port\", \"unknown\")}')
        print(f'  Uptime: {master.get(\"uptime_seconds\", 0):.1f}s')
        print(f'  Active Tasks: {master.get(\"active_tasks\", 0)}')
        print(f'  Total Tasks: {master.get(\"total_tasks\", 0)}')

    # 内存池统计
    pool = data.get('memory_pool', {})
    if pool:
        print(f'  Pool Size: {pool.get(\"pool_size_mb\", 0)}MB')
        print(f'  Free Chunks: {pool.get(\"free_chunks\", 0)}')
        print(f'  Used Size: {pool.get(\"used_size_mb\", 0):.2f}MB')
        print(f'  Active Tasks: {pool.get(\"active_tasks\", 0)}')

    # Worker统计
    workers = data.get('workers', [])
    if workers:
        print(f'  Active Workers: {len(workers)}')
        for worker in workers:
            print(f'    Worker {worker.get(\"worker_id\", \"unknown\")}:')
            print(f'      Tasks: {worker.get(\"tasks_processed\", 0)}')
            print(f'      Success Rate: {worker.get(\"success_rate\", 0)}%')
            print(f'      Avg Processing: {worker.get(\"average_processing_time\", 0):.2f}s')
            print(f'      Throughput: {worker.get(\"throughput_mb_per_hour\", 0):.2f}MB/h')
    else:
        print('  No worker data available')

except Exception as e:
    print(f'Error parsing stats: {e}')
"
else
    log_error "❌ Failed to get service statistics"
fi

echo ""
log_info "✅ Service is running successfully!"
echo ""
echo "🌐 Service URL: http://localhost:$PORT"
echo "📊 Stats URL: http://localhost:$PORT/stats"
echo "📋 Logs: tail -f logs/master_gpu_${GPU_ID}.log"
echo ""
echo "🧪 Ready for testing!"

# 保持进程运行，等待用户停止
log_info "Press Ctrl+C to stop the service"
wait $MASTER_PID