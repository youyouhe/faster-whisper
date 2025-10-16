#!/bin/bash
# Docker内统一多GPU架构启动脚本 (修复版本)
# Unified Multi-GPU Architecture Startup Script for Docker (Fixed Version)

# 移除set -e，避免脚本意外退出

# 设置库路径
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cublas.lib; import nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ':' + nvidia.cudnn.lib.__path__[0])" 2>/dev/null || echo "")

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_blue() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# 获取配置参数
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
MAX_FILE_SIZE=${MAX_FILE_SIZE:-500}
MODEL=${WHISPER_MODEL:-large-v3-turbo}
LOG_LEVEL=${LOG_LEVEL:-INFO}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-180}  # 3分钟健康检查超时
MONITOR_INTERVAL=${MONITOR_INTERVAL:-60}           # 监控间隔1分钟

log_info "=== Docker Unified Multi-GPU Architecture Startup (Fixed Version) ==="
log_info "Configuration: $WORKERS_PER_GPU workers per GPU, $MAX_FILE_SIZE MB max file size"
log_info "Model: $MODEL, Log Level: $LOG_LEVEL"
log_info "Health Check Timeout: ${HEALTH_CHECK_TIMEOUT}s, Monitor Interval: ${MONITOR_INTERVAL}s"

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Please check NVIDIA driver installation."
    exit 1
fi

# 获取GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
log_info "Detected $GPU_COUNT GPU(s) in container"

# 使用所有检测到的GPU
log_info "Using all $GPU_COUNT GPUs"

if [ $GPU_COUNT -eq 0 ]; then
    log_error "No GPUs detected. Please check NVIDIA container runtime."
    exit 1
fi

# 全局变量存储进程PID
declare -A GPU_PIDS
LB_PID=""
RESTART_COUNT=0
MAX_RESTARTS=5

# 清理现有进程
cleanup() {
    log_info "Cleaning up existing processes..."

    # 清理unified_api进程
    PIDS=$(pgrep -f "unified_api.py" || true)
    if [ ! -z "$PIDS" ]; then
        log_warn "Found existing unified_api processes: $PIDS"
        kill $PIDS 2>/dev/null || true
        sleep 3

        REMAINING=$(pgrep -f "unified_api.py" || true)
        if [ ! -z "$REMAINING" ]; then
            log_warn "Force killing remaining processes: $REMAINING"
            kill -9 $REMAINING 2>/dev/null || true
        fi
    fi

    # 清理load_balancer进程
    LB_PIDS=$(pgrep -f "load_balancer.py" || true)
    if [ ! -z "$LB_PIDS" ]; then
        log_warn "Found existing load_balancer processes: $LB_PIDS"
        kill $LB_PIDS 2>/dev/null || true
        sleep 3

        LB_REMAINING=$(pgrep -f "load_balancer.py" || true)
        if [ ! -z "$LB_REMAINING" ]; then
            log_warn "Force killing remaining load_balancer processes: $LB_REMAINING"
            kill -9 $LB_REMAINING 2>/dev/null || true
        fi
    fi

    # 清理共享内存
    log_info "Cleaning up shared memory..."
    rm -f /dev/shm/whisper_* 2>/dev/null || true

    log_info "Cleanup completed"
}

# 检查端口是否被占用
check_port() {
    local port=$1
    if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
        return 1
    elif ss -tlnp 2>/dev/null | grep -q ":$port "; then
        return 1
    elif lsof -i:$port 2>/dev/null; then
        return 1
    fi
    return 0
}

# 检查服务健康状态
check_service_health() {
    local port=$1
    local timeout=$2
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))

    log_info "Checking health of service on port $port (timeout: ${timeout}s)"

    while [ $(date +%s) -lt $end_time ]; do
        if curl -s --max-time 5 "http://localhost:$port/health" >/dev/null 2>&1; then
            log_info "✅ Service on port $port is healthy"
            return 0
        fi
        sleep 2
    done

    log_error "❌ Service on port $port failed health check after ${timeout}s"
    return 1
}

# 启动单个GPU的unified_api实例
start_gpu_instance() {
    local gpu_id=$1
    local port=$2
    local workers=$3
    local restart=${4:-false}

    local restart_text=""
    if [ "$restart" = "true" ]; then
        restart_text=" (RESTART)"
    fi

    log_blue "Starting unified_api for GPU $gpu_id on port $port (workers: $workers)$restart_text"

    # 检查端口
    if ! check_port $port; then
        log_error "Port $port is already in use. Skipping GPU $gpu_id."
        return 1
    fi

    # 创建日志目录
    LOG_DIR="logs/gpu_${gpu_id}"
    mkdir -p "$LOG_DIR"

    # 修复：移除重复的--gpus参数，只使用CUDA_VISIBLE_DEVICES
    # 设置GPU可见性并启动服务
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 unified_api.py \
        --port "$port" \
        --workers-per-gpu "$workers" \
        --model "$MODEL" \
        --log-level "$LOG_LEVEL" \
        --max-file-size "$MAX_FILE_SIZE" \
        > "$LOG_DIR/unified_api.log" 2>&1 &

    local pid=$!
    GPU_PIDS[$gpu_id]=$pid
    log_info "Started unified_api for GPU $gpu_id with PID: $pid"

    # 等待服务启动
    sleep 5

    # 检查进程是否仍在运行
    if kill -0 $pid 2>/dev/null; then
        log_info "✅ GPU $gpu_id unified_api process is running (PID: $pid, Port: $port)"

        # 健康检查
        if check_service_health $port $HEALTH_CHECK_TIMEOUT; then
            log_info "✅ GPU $gpu_id unified_api is healthy and ready"
            return 0
        else
            log_error "❌ GPU $gpu_id unified_api failed health check"

            # 健康检查失败，清理进程
            log_warn "Cleaning up failed GPU $gpu_id process (PID: $pid)"
            kill $pid 2>/dev/null || true
            sleep 2
            kill -9 $pid 2>/dev/null || true
            unset GPU_PIDS[$gpu_id]

            return 1
        fi
    else
        log_error "❌ GPU $gpu_id unified_api failed to start (process died)"
        return 1
    fi
}

# 重启失败的GPU实例
restart_gpu_instance() {
    local gpu_id=$1
    local port=$((5002 + gpu_id))

    log_warn "Attempting to restart GPU $gpu_id instance..."

    # 清理可能残留的进程
    if [ -n "${GPU_PIDS[$gpu_id]}" ]; then
        local old_pid=${GPU_PIDS[$gpu_id]}
        log_info "Cleaning up old process PID: $old_pid"
        kill $old_pid 2>/dev/null || true
        sleep 2
        kill -9 $old_pid 2>/dev/null || true
    fi

    # 等待一段时间再重启
    sleep 10

    # 重启实例
    if start_gpu_instance $gpu_id $port $WORKERS_PER_GPU true; then
        log_info "✅ GPU $gpu_id restart successful"
        return 0
    else
        log_error "❌ GPU $gpu_id restart failed"
        return 1
    fi
}

# 检查所有GPU实例状态
check_gpu_instances() {
    local failed_instances=()

    for ((i=0; i<GPU_COUNT; i++)); do
        local port=$((5002 + i))
        local pid=${GPU_PIDS[$i]}

        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            # 进程存在，检查健康状态
            if ! curl -s --max-time 5 "http://localhost:$port/health" >/dev/null 2>&1; then
                log_warn "⚠️  GPU $i instance unhealthy (PID: $pid, Port: $port)"
                failed_instances+=($i)
            fi
        else
            log_warn "⚠️  GPU $i instance not running (PID: $pid)"
            failed_instances+=($i)
        fi
    done

    echo "${failed_instances[@]}"
}

# 生成load_balancer配置
generate_lb_config() {
    local config_file="load_balancer_config.env"

    log_info "Generating load balancer configuration: $config_file"

    # 只包含健康的实例
    local healthy_ports=""
    for ((i=0; i<GPU_COUNT; i++)); do
        local port=$((5002 + i))
        local pid=${GPU_PIDS[$i]}

        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            if curl -s --max-time 5 "http://localhost:$port/health" >/dev/null 2>&1; then
                if [ -z "$healthy_ports" ]; then
                    healthy_ports="http://localhost:$port"
                else
                    healthy_ports="$healthy_ports,http://localhost:$port"
                fi
            fi
        fi
    done

    if [ -z "$healthy_ports" ]; then
        log_error "No healthy GPU instances found for load balancer configuration"
        return 1
    fi

    cat > "$config_file" << EOF
# Load Balancer Configuration for Multi-GPU unified_api
# Generated by start_unified_multi_gpu_fixed.sh

# Backend services (healthy instances only)
BACKEND_SERVICES=$healthy_ports

# Load Balancer Configuration
LB_PORT=5001
MAX_QUEUE_SIZE=100
REQUEST_TIMEOUT=3600
HEALTH_CHECK_INTERVAL=30
MAX_FILE_SIZE=$MAX_FILE_SIZE

# Logging
LOG_LEVEL=INFO
EOF

    log_info "Configuration saved to: $config_file. Healthy backends: $healthy_ports"
    return 0
}

# 启动负载均衡器
start_load_balancer() {
    if [ -f "load_balancer_config.env" ]; then
        # 检查端口5001是否被占用
        if ! check_port 5001; then
            log_warn "Port 5001 is already in use, checking if load_balancer is running..."
            if pgrep -f "load_balancer.py" >/dev/null; then
                log_info "Load balancer already running, skipping restart"
                return 0
            else
                log_error "Port 5001 occupied but no load_balancer process found"
                return 1
            fi
        fi

        log_info "Starting load balancer..."
        source load_balancer_config.env
        nohup python3 load_balancer.py > logs/load_balancer.log 2>&1 &
        LB_PID=$!
        log_info "✅ Load balancer started (PID: $LB_PID, Port: $LB_PORT)"

        # 等待负载均衡器启动
        sleep 5

        if kill -0 $LB_PID 2>/dev/null; then
            log_info "✅ Load balancer is running"
            return 0
        else
            log_error "❌ Load balancer failed to start"
            return 1
        fi
    else
        log_error "Load balancer configuration file not found"
        return 1
    fi
}

# 监控和自动重启
monitor_services() {
    log_info "Starting service monitoring (interval: ${MONITOR_INTERVAL}s)"

    while true; do
        sleep $MONITOR_INTERVAL

        log_info "Performing health check on all services..."

        # 检查GPU实例
        failed_instances=($(check_gpu_instances))

        if [ ${#failed_instances[@]} -gt 0 ]; then
            log_warn "Found ${#failed_instances[@]} failed instances: ${failed_instances[*]}"

            # 重启失败的实例
            for instance in "${failed_instances[@]}"; do
                if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
                    log_info "Restarting failed GPU $instance instance (restart count: $((RESTART_COUNT + 1))/$MAX_RESTARTS)"

                    if restart_gpu_instance $instance; then
                        log_info "✅ GPU $instance restart successful"
                    else
                        log_error "❌ GPU $instance restart failed"
                        ((RESTART_COUNT++))
                    fi
                else
                    log_error "❌ Maximum restart limit ($MAX_RESTARTS) reached, giving up on GPU $instance"
                fi
            done

            # 如果有实例重启，重新生成负载均衡器配置
            if [ ${#failed_instances[@]} -gt 0 ] && [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
                log_info "Regenerating load balancer configuration after restarts..."
                if generate_lb_config; then
                    # 重启负载均衡器以应用新配置
                    if [ -n "$LB_PID" ]; then
                        log_info "Restarting load balancer with new configuration..."
                        kill $LB_PID 2>/dev/null || true
                        sleep 3
                        start_load_balancer
                    fi
                fi
            fi
        else
            log_info "✅ All GPU instances are healthy"
        fi

        # 检查负载均衡器
        if [ -n "$LB_PID" ] && kill -0 $LB_PID 2>/dev/null; then
            if curl -s --max-time 5 "http://localhost:5001/health" >/dev/null 2>&1; then
                log_info "✅ Load balancer is healthy"
            else
                log_warn "⚠️  Load balancer unhealthy, restarting..."
                kill $LB_PID 2>/dev/null || true
                sleep 3
                start_load_balancer
            fi
        else
            log_warn "⚠️  Load balancer not running, starting..."
            start_load_balancer
        fi

        log_info "Health check completed. Next check in ${MONITOR_INTERVAL}s"
    done
}

# 主函数
main() {
    # 清理现有进程
    cleanup

    # 创建日志目录
    mkdir -p logs

    # 启动所有GPU实例
    log_info "Starting unified_api instances..."
    SUCCESS_COUNT=0
    FAILED_INSTANCES=()

    for ((i=0; i<GPU_COUNT; i++)); do
        local port=$((5002 + i))

        log_info "=== Attempting to start GPU $i (port $port) ==="

        if start_gpu_instance $i $port $WORKERS_PER_GPU; then
            ((SUCCESS_COUNT++))
            log_info "✅ GPU $i startup completed successfully"
        else
            log_error "❌ GPU $i startup failed"
            FAILED_INSTANCES+=($i)
        fi

        # 错开启动时间避免资源竞争
        sleep 10
    done

    log_info "Deployment summary: $SUCCESS_COUNT/$GPU_COUNT instances started successfully"

    if [ ${#FAILED_INSTANCES[@]} -gt 0 ]; then
        log_warn "Failed instances: ${FAILED_INSTANCES[*]}"
    fi

    # 如果没有成功启动的实例，退出
    if [ $SUCCESS_COUNT -eq 0 ]; then
        log_error "No GPU instances started successfully. Exiting."
        exit 1
    fi

    # 等待所有服务完全启动
    log_info "Waiting for services to fully initialize..."
    sleep 10

    # 生成配置文件
    if generate_lb_config; then
        # 启动load balancer
        if start_load_balancer; then
            log_info "=== Unified Multi-GPU Architecture Started Successfully ==="
            log_info "Service endpoints:"
            for ((i=0; i<GPU_COUNT; i++)); do
                local port=$((5002 + i))
                local pid=${GPU_PIDS[$i]}
                if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                    echo "  http://localhost:$port/inference (GPU $i unified_api)"
                fi
            done
            echo "  http://localhost:5001/inference (Load Balancer)"

            log_info "To check status: curl http://localhost:5001/health"
            log_info "To stop all services: killall unified_api.py && killall load_balancer.py"
            log_info "Logs are stored in logs/ directory"

            # 启动监控
            monitor_services
        else
            log_error "Failed to start load balancer"
            exit 1
        fi
    else
        log_error "Failed to generate load balancer configuration"
        exit 1
    fi
}

# 信号处理
trap cleanup EXIT INT TERM

# 执行主函数
main "$@"