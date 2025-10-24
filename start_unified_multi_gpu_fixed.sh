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

# 获取配置参数 (将在GPU检测后根据情况调整)
# 使用base模型进行调试（减少资源使用）
MODEL=${WHISPER_MODEL:-base}
LOG_LEVEL=${LOG_LEVEL:-INFO}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-180}  # 3分钟健康检查超时
MONITOR_INTERVAL=${MONITOR_INTERVAL:-60}           # 监控间隔1分钟

log_info "=== Docker Unified Multi-GPU Architecture Startup (Multi-GPU with Base Model) ==="

# 显示使用说明
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  MAX_GPUS=4          Maximum number of GPUs to use (default: 4)"
    echo "  DESIRED_GPU_COUNT=2 Specific number of GPUs to use"
    echo "  WORKERS_PER_GPU=2   Workers per GPU (default: 2)"
    echo "  WHISPER_MODEL=base  Model to use (base, small, medium, large-v3, large-v3-turbo)"
    echo "  MAX_FILE_SIZE=500   Maximum file size in MB"
    echo ""
    echo "Examples:"
    echo "  $0                           # Auto-detect and use all available GPUs"
    echo "  MAX_GPUS=2 $0                # Use maximum 2 GPUs"
    echo "  DESIRED_GPU_COUNT=2 $0       # Use exactly 2 GPUs"
    echo "  WHISPER_MODEL=base $0        # Use base model with more workers"
    echo ""
}

# 如果用户请求帮助
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Please check NVIDIA driver installation."
    exit 1
fi

# 获取GPU数量和详细信息
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
log_info "Detected $GPU_COUNT GPU(s) in container"

# 验证每个GPU的可用性
AVAILABLE_GPUS=0
for ((i=0; i<GPU_COUNT; i++)); do
    if nvidia-smi --id=$i --query-gpu=name,memory.total --format=csv,noheader >/dev/null 2>&1; then
        gpu_name=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader)
        gpu_memory=$(nvidia-smi --id=$i --query-gpu=memory.total --format=csv,noheader | sed 's/,//g')
        log_info "GPU $i: $gpu_name (${gpu_memory} MiB) - ✅ Available"
        ((AVAILABLE_GPUS++))
    else
        log_warn "GPU $i: ❌ Not available"
    fi
done

if [ $AVAILABLE_GPUS -eq 0 ]; then
    log_error "No available GPUs detected. Please check NVIDIA driver and CUDA setup."
    exit 1
fi

# 动态检测可用GPU数量，但限制最大值以避免资源过度使用
DETECTED_GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
MAX_GPUS=${MAX_GPUS:-4}  # 最多使用4个GPU

# 计算实际使用的GPU数量
if [ "$DETECTED_GPU_COUNT" -le "$MAX_GPUS" ]; then
    GPU_COUNT=$DETECTED_GPU_COUNT
else
    GPU_COUNT=$MAX_GPUS
fi

# 恢复多GPU模式，但保持base模型
# 如果指定了GPU数量，则使用指定值
if [ -n "$DESIRED_GPU_COUNT" ] && [ "$DESIRED_GPU_COUNT" -le "$DETECTED_GPU_COUNT" ] && [ "$DESIRED_GPU_COUNT" -le "$MAX_GPUS" ]; then
    GPU_COUNT=$DESIRED_GPU_COUNT
fi

log_info "Detected $DETECTED_GPU_COUNT GPU(s), using $GPU_COUNT GPU(s) for processing"

# 修改每个GPU的worker数量为1（减少内存使用，避免崩溃）
WORKERS_PER_GPU=${WORKERS_PER_GPU:-1}

# 根据模型调整文件大小限制
if [ "$MODEL" = "large-v3-turbo" ] || [ "$MODEL" = "large-v3" ]; then
    # 大模型使用更大的文件大小限制
    MAX_FILE_SIZE=${MAX_FILE_SIZE:-500}
else
    # 小模型可以支持更大的文件
    MAX_FILE_SIZE=${MAX_FILE_SIZE:-500}
fi

# 显示GPU内存信息以便调优
for ((i=0; i<GPU_COUNT; i++)); do
    gpu_memory=$(nvidia-smi --id=$i --query-gpu=memory.total --format=csv,noheader | sed 's/,//g' | tr -d ' ')
    log_info "GPU $i Memory: ${gpu_memory} MiB"
done

log_info "Final Configuration: $WORKERS_PER_GPU workers per GPU, $MAX_FILE_SIZE MB max file size"
log_info "Model: $MODEL, Log Level: $LOG_LEVEL"
log_info "Health Check Timeout: ${HEALTH_CHECK_TIMEOUT}s, Monitor Interval: ${MONITOR_INTERVAL}s"

# 显示详细的配置摘要
echo ""
log_info "=== Multi-GPU Configuration Summary ==="
log_info "GPU Count: $GPU_COUNT (Available: $DETECTED_GPU_COUNT, Max: $MAX_GPUS)"
log_info "Workers per GPU: $WORKERS_PER_GPU"
log_info "Total Workers: $((GPU_COUNT * WORKERS_PER_GPU))"
log_info "Model: $MODEL"
log_info "Max File Size: ${MAX_FILE_SIZE}MB"
log_info "Service Ports: 5002-$((5001 + GPU_COUNT))"
log_info "Load Balancer Port: 5001"
echo ""

# 计算预估资源使用
ESTIMATED_MEMORY_PER_WORKER=2048  # 每个worker预估使用2GB内存
TOTAL_ESTIMATED_MEMORY=$((GPU_COUNT * WORKERS_PER_GPU * ESTIMATED_MEMORY_PER_WORKER))
log_info "Estimated GPU Memory Usage: ${TOTAL_ESTIMATED_MEMORY}MB total (${ESTIMATED_MEMORY_PER_WORKER}MB per worker)"
log_info "Configuration is optimized for model: $MODEL"
echo ""

# 全局变量存储进程PID
declare -A GPU_PIDS
declare -A GPU_RESTART_COUNT  # 为每个GPU维护独立的重启计数
LB_PID=""
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

    # 清理状态变量
    log_info "Resetting state variables..."
    if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 0 ]; then
        for ((i=0; i<GPU_COUNT; i++)); do
            unset GPU_PIDS[$i]
            unset GPU_RESTART_COUNT[$i]
        done
        log_info "Reset state for $GPU_COUNT GPU instances"
    fi
    LB_PID=""

    log_info "Cleanup completed"
}

# 检查GPU内存使用情况
check_gpu_memory() {
    local gpu_id=$1
    local memory_info=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    if [ $? -eq 0 ]; then
        # 修复：清理可能包含的逗号，并确保是数字
        local used=$(echo $memory_info | awk '{print $1}' | sed 's/,//g')
        local total=$(echo $memory_info | awk '{print $2}' | sed 's/,//g')

        # 确保是有效的数字
        if [[ "$used" =~ ^[0-9]+$ ]] && [[ "$total" =~ ^[0-9]+$ ]] && [ "$total" -gt 0 ]; then
            local usage_percent=$((used * 100 / total))
            log_info "GPU $gpu_id memory: ${used}MB/${total}MB (${usage_percent}%)"
            return $usage_percent
        else
            log_warn "GPU $gpu_id memory info invalid: '$memory_info'"
            return 0
        fi
    else
        log_warn "Failed to get GPU $gpu_id memory info"
        return 100
    fi
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

    # 检查GPU内存状态
    check_gpu_memory $gpu_id
    local memory_usage=$?
    if [ $memory_usage -gt 80 ]; then
        log_warn "GPU $gpu_id memory usage is high (${memory_usage}%), proceeding with caution..."
    fi

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

# 获取健康GPU实例数量
get_healthy_gpu_count() {
    local healthy_count=0

    for ((i=0; i<GPU_COUNT; i++)); do
        local port=$((5002 + i))
        local pid=${GPU_PIDS[$i]}

        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            # 进程存在，检查健康状态
            if curl -s --max-time 5 "http://localhost:$port/health" >/dev/null 2>&1; then
                ((healthy_count++))
            fi
        fi
    done

    echo $healthy_count
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

    # 修复：只有当有失败实例时才返回，否则返回空字符串
    if [ ${#failed_instances[@]} -gt 0 ]; then
        printf "%s" "${failed_instances[*]}"
    else
        printf ""
    fi
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

        # 修复：正确获取失败实例列表，避免空数组问题
        failed_instances_str=$(check_gpu_instances)

        # 只有当字符串非空时才转换为数组
        if [ -n "$failed_instances_str" ]; then
            IFS=' ' read -ra failed_instances <<< "$failed_instances_str"
        else
            failed_instances=()
        fi

        # 检查是否有真正的失败实例
        if [ ${#failed_instances[@]} -gt 0 ]; then
            local failed_list="${failed_instances[*]}"
            log_warn "Found ${#failed_instances[@]} failed instances: ${failed_list}"

            has_successful_restart=false
            has_failed_restart=false

            # 重启失败的实例
            for instance in "${failed_instances[@]}"; do
                local current_restart_count=${GPU_RESTART_COUNT[$instance]:-0}

                if [ $current_restart_count -lt $MAX_RESTARTS ]; then
                    log_info "Restarting failed GPU $instance instance (attempt: $((current_restart_count + 1))/$MAX_RESTARTS)"

                    if restart_gpu_instance $instance; then
                        log_info "✅ GPU $instance restart successful"
                        # 重启成功，重置计数
                        GPU_RESTART_COUNT[$instance]=0
                        has_successful_restart=true
                    else
                        log_error "❌ GPU $instance restart failed"
                        # 重启失败，增加计数
                        ((GPU_RESTART_COUNT[$instance]++))
                        has_failed_restart=true
                    fi
                else
                    log_error "❌ Maximum restart limit ($MAX_RESTARTS) reached for GPU $instance, giving up"
                    has_failed_restart=true
                fi
            done

            # 如果有重启成功或有失败但还有GPU在工作，重新生成负载均衡器配置
            if [ $has_successful_restart = true ] || ([ $has_failed_restart = true ] && [ $(get_healthy_gpu_count) -gt 0 ]); then
                log_info "Regenerating load balancer configuration after restarts..."
                if generate_lb_config; then
                    # 重启负载均衡器以应用新配置
                    if [ -n "$LB_PID" ]; then
                        log_info "Restarting load balancer with new configuration..."
                        kill $LB_PID 2>/dev/null || true
                        sleep 3
                        start_load_balancer
                    fi
                else
                    log_error "Failed to regenerate load balancer configuration"
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
                if ! start_load_balancer; then
                    log_error "Failed to restart load balancer"
                fi
            fi
        else
            log_warn "⚠️  Load balancer not running, starting..."
            if ! start_load_balancer; then
                log_error "Failed to start load balancer"
            fi
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

        # 智能调整启动间隔时间
        # 根据worker数量和模型大小调整等待时间
        if [ "$MODEL" = "large-v3-turbo" ] || [ "$MODEL" = "large-v3" ]; then
            # 大模型需要更长的初始化时间
            STARTUP_DELAY=45
        else
            # 小模型启动较快
            STARTUP_DELAY=30
        fi

        if [ $GPU_COUNT -gt 1 ]; then
            log_info "Waiting ${STARTUP_DELAY}s before starting next GPU (${WORKERS_PER_GPU} workers, model: ${MODEL})..."
            sleep $STARTUP_DELAY
        else
            # 单GPU情况下也要等待模型完全加载
            log_info "Waiting $((STARTUP_DELAY/2))s for model to fully load..."
            sleep $((STARTUP_DELAY/2))
        fi
    done

    log_info "Deployment summary: $SUCCESS_COUNT/$GPU_COUNT instances started successfully"

    if [ ${#FAILED_INSTANCES[@]} -gt 0 ]; then
        local failed_list="${FAILED_INSTANCES[*]}"
        log_warn "Failed instances: ${failed_list}"
    fi

    # 如果没有成功启动的实例，退出
    if [ $SUCCESS_COUNT -eq 0 ]; then
        log_error "No GPU instances started successfully. Exiting."
        exit 1
    fi

    # 智能等待所有服务完全启动
    log_info "Waiting for all services to fully initialize..."

    # 计算总的worker数量
    TOTAL_WORKERS=$((GPU_COUNT * WORKERS_PER_GPU))

    # 根据模型大小和总worker数量调整等待时间
    if [ "$MODEL" = "large-v3-turbo" ] || [ "$MODEL" = "large-v3" ]; then
        BASE_WAIT=60  # 大模型基础等待时间
    else
        BASE_WAIT=30  # 小模型基础等待时间
    fi

    # 每增加一个worker，增加额外等待时间
    ADDITIONAL_WAIT=$((TOTAL_WORKERS * 15))
    TOTAL_WAIT=$((BASE_WAIT + ADDITIONAL_WAIT))

    log_info "Multi-GPU setup: ${GPU_COUNT} GPUs × ${WORKERS_PER_GPU} workers = ${TOTAL_WORKERS} total workers"
    log_info "Waiting ${TOTAL_WAIT}s for full initialization (model: ${MODEL})..."
    sleep $TOTAL_WAIT

    # 生成配置文件
    if generate_lb_config; then
        # 启动load balancer
        if start_load_balancer; then
            log_info "=== Unified Multi-GPU Architecture Started Successfully ==="
            echo ""
            log_info "🚀 Service Architecture:"
            log_info "  Load Balancer: http://localhost:5001/inference (Main Entry Point)"
            log_info "  Backend Services:"
            for ((i=0; i<GPU_COUNT; i++)); do
                local port=$((5002 + i))
                local pid=${GPU_PIDS[$i]}
                if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                    echo "    - GPU $i: http://localhost:$port/inference (${WORKERS_PER_GPU} workers)"
                fi
            done
            echo ""

            log_info "📊 Performance Metrics:"
            log_info "  Total GPU Workers: $((GPU_COUNT * WORKERS_PER_GPU))"
            log_info "  Load Balancing: Round-robin with health checks"
            log_info "  Model: $MODEL"
            log_info "  Max Concurrent Requests: $((GPU_COUNT * WORKERS_PER_GPU))"
            echo ""

            log_info "🔧 Management Commands:"
            log_info "  Health Check: curl http://localhost:5001/health"
            log_info "  Stats: curl http://localhost:5001/stats"
            log_info "  Stop Services: killall unified_api.py && killall load_balancer.py"
            log_info "  View Logs: tail -f logs/load_balancer.log"
            log_info "  GPU Logs: ls logs/gpu_*/unified_api.log"
            echo ""

            log_info "✅ Ready for production load!"

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
