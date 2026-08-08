#!/bin/bash
# 本机（非 Docker）一键启动脚本
# 启动: Redis(若未运行) + GPU后端(每卡1实例, 5002+) + 负载均衡(5001)
#       + TUS上传(1080) + TUS API(8000) + ASR Worker + 回调服务
# 用法: ./start_local.sh          前台启动（Ctrl+C 停止全部）
# 模型首次启动会自动下载 large-v3-turbo（约3GB），需等待数分钟。

set -e
cd "$(dirname "$0")"
REPO_DIR=$(pwd)

# --- Python 环境 ---
export PATH="$REPO_DIR/faster-whisper-env/bin:$PATH"
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cublas.lib, nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ':' + nvidia.cudnn.lib.__path__[0])")

# --- 本机环境变量（覆盖 Docker 主机名默认值）---
export REDIS_URL=${REDIS_URL:-redis://localhost:6379}
export LOAD_BALANCER_URL=${LOAD_BALANCER_URL:-http://localhost:5001}
export TUS_SERVER_BASE_URL=${TUS_SERVER_BASE_URL:-http://localhost:1080/files}
export UPLOAD_DIR=${UPLOAD_DIR:-$REPO_DIR/data/uploads}
export SRT_STORAGE_DIR=${SRT_STORAGE_DIR:-$REPO_DIR/data/srt_results}
export API_KEY=${API_KEY:-$(grep -E '^API_KEY=' .env 2>/dev/null | cut -d= -f2)}
mkdir -p "$UPLOAD_DIR" "$SRT_STORAGE_DIR" logs

# --- Redis（未运行则启动本机实例）---
REDIS_BIN=/mnt/oldroot/home/bird/miniconda3/envs/fw312/bin/redis-server
REDIS_CLI=/mnt/oldroot/home/bird/miniconda3/envs/fw312/bin/redis-cli
if ! $REDIS_CLI ping &>/dev/null; then
    echo "🚀 启动本机 Redis..."
    $REDIS_BIN --daemonize yes --port 6379 --dir "$REPO_DIR/data"
    sleep 1
fi
echo "✅ Redis: $($REDIS_CLI ping)"

# --- GPU 检测 ---
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -n 1)
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "⚠️  未检测到 GPU，默认 1"
    NUM_GPUS=1
fi
echo "🔍 检测到 $NUM_GPUS 张 GPU，每卡启动 1 个实例（8G 显存跑 float32 large-v3-turbo，不建议多实例）"

# --- 退出时清理 ---
PIDS=()
cleanup() {
    echo ""
    echo "正在停止所有服务..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    echo "已全部停止。"
    exit
}
trap cleanup EXIT INT TERM

# --- 启动 GPU 后端 (5002+) ---
START_PORT=5002
BACKEND_URLS=()
for ((i=0; i<NUM_GPUS; i++)); do
    export CUDA_VISIBLE_DEVICES=$i
    API_PORT=$((START_PORT + i))
    echo "🚀 GPU $i 后端 → 端口 $API_PORT (日志: logs/gpu_$i.log)"
    CUDA_VISIBLE_DEVICES=$i GPU_DEVICE_ID=$i API_PORT=$API_PORT INSTANCE_ID=gpu$i \
        python3 faster_whisper_api.py > "logs/gpu_$i.log" 2>&1 &
    PIDS+=($!)
    BACKEND_URLS+=("http://localhost:$API_PORT")
done

# --- 负载均衡 (5001) ---
echo "🔄 负载均衡 → 端口 5001 (日志: logs/lb.log)"
LB_PORT=5001 BACKEND_SERVICES=$(IFS=,; echo "${BACKEND_URLS[*]}") \
    REQUEST_TIMEOUT=1800 MAX_QUEUE_SIZE=100 HEALTH_CHECK_INTERVAL=30 \
    python3 load_balancer.py > logs/lb.log 2>&1 &
PIDS+=($!)

# --- TUS 上传服务 (1080) ---
echo "📤 TUS 上传服务 → 端口 1080 (日志: logs/tus.log)"
python3 tus_server.py > logs/tus.log 2>&1 &
PIDS+=($!)

# --- TUS API (8000) ---
echo "🌐 TUS API → 端口 8000 (日志: logs/tus_api.log)"
API_PORT=8000 python3 tus_api_server.py > logs/tus_api.log 2>&1 &
PIDS+=($!)

# --- ASR Worker ---
echo "👷 ASR Worker (日志: logs/worker.log)"
python3 asr_worker.py > logs/worker.log 2>&1 &
PIDS+=($!)

# --- 回调服务 ---
echo "📞 回调服务 (日志: logs/callback.log)"
python3 callback_service.py > logs/callback.log 2>&1 &
PIDS+=($!)

echo ""
echo "✅ 全部服务已启动:"
echo "   负载均衡:  http://localhost:5001/health"
echo "   TUS API:   http://localhost:8000"
echo "   GPU 后端:  端口 $START_PORT-$((START_PORT + NUM_GPUS - 1))（模型加载需 2-3 分钟）"
echo "   日志目录:  logs/"
echo ""
echo "⚠️  按 Ctrl+C 停止所有服务"
wait
