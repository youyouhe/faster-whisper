#!/bin/bash
"""
启动共享内存架构
Start Shared Memory Architecture
"""

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${CYAN}=== $1 ===${NC}"
}

# 检查依赖
check_dependencies() {
    log_info "Checking dependencies..."

    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi

    # 检查必要的Python包
    python3 -c "import fastapi, uvicorn" 2>/dev/null || {
        log_warning "Some Python packages are missing. Installing..."
        pip3 install fastapi uvicorn
    }

    log_success "Dependencies check completed"
}

# 清理旧的进程和共享内存
cleanup() {
    log_info "Cleaning up old processes and shared memory..."

    # 杀死可能存在的Python进程
    pkill -f "master_process.py" 2>/dev/null || true
    pkill -f "worker_process.py" 2>/dev/null || true
    pkill -f "unified_api.py" 2>/dev/null || true

    # 等待进程结束
    sleep 2

    # 清理共享内存
    if command -v python3 &> /dev/null; then
        python3 -c "
import sys
sys.path.append('.')
from shared_memory_manager import SharedMemoryPool
import logging
logging.basicConfig(level=logging.ERROR)

# 尝试清理所有GPU的共享内存
for gpu_id in range(4):
    try:
        pool = SharedMemoryPool(gpu_id=gpu_id, pool_size_mb=1)
        pool.cleanup()
        print(f'Cleaned up GPU {gpu_id} shared memory')
    except:
        pass
"
    fi

    log_success "Cleanup completed"
}

# 启动架构
start_architecture() {
    log_header "Starting Shared Memory Architecture"

    # 设置环境变量
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
    export API_PORT=${API_PORT:-5001}
    export LOG_LEVEL=${LOG_LEVEL:-INFO}
    export MAX_FILE_SIZE=${MAX_FILE_SIZE:-50}

    log_info "Configuration:"
    log_info "  Workers per GPU: $WORKERS_PER_GPU"
    log_info "  API Port: $API_PORT"
    log_info "  Log Level: $LOG_LEVEL"
    log_info "  Max File Size: ${MAX_FILE_SIZE}MB"

    # 创建日志目录
    mkdir -p logs

    # 启动统一API服务
    log_info "Starting Unified API Service..."
    python3 unified_api.py > logs/unified_api.log 2>&1 &
    API_PID=$!

    # 保存PID到文件
    echo $API_PID > .api_pid

    log_success "Unified API started (PID: $API_PID)"
    log_info "API will be available at: http://localhost:$API_PORT"
    log_info "API Documentation: http://localhost:$API_PORT/docs"

    # 等待服务启动
    log_info "Waiting for services to initialize..."
    sleep 10

    # 检查服务是否正常运行
    if ps -p $API_PID > /dev/null; then
        log_success "Shared Memory Architecture started successfully!"

        echo ""
        log_header "Service Information"
        echo "🌐 API Endpoint: http://localhost:$API_PORT"
        echo "📚 API Documentation: http://localhost:$API_PORT/docs"
        echo "📊 Health Check: http://localhost:$API_PORT/health"
        echo "📈 Statistics: http://localhost:$API_PORT/stats"
        echo ""
        echo "🔍 Example Usage:"
        echo "   # Health check:"
        echo "   curl http://localhost:$API_PORT/health"
        echo ""
        echo "   # Upload audio file:"
        echo "   curl -X POST -F 'file=@audio.wav' http://localhost:$API_PORT/transcribe"
        echo ""
        echo "📋 Monitoring:"
        echo "   # View logs:"
        echo "   tail -f logs/unified_api.log"
        echo ""
        echo "   # Check process status:"
        echo "   ps aux | grep -E '(master_process|worker_process|unified_api)'"
        echo ""
        echo "⚠️  Press Ctrl+C to stop all services"

        # 监控循环
        monitor_services

    else
        log_error "Failed to start API service!"
        log_error "Check logs: logs/unified_api.log"
        exit 1
    fi
}

# 监控服务状态
monitor_services() {
    while true; do
        # 检查API进程
        if ! ps -p $(cat .api_pid 2>/dev/null) > /dev/null 2>&1; then
            log_error "API process died!"
            break
        fi

        # 每分钟显示一次状态
        sleep 60

        # 尝试健康检查
        if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
            log_info "Service health check passed"
        else
            log_warning "Service health check failed"
        fi
    done
}

# 测试架构
test_architecture() {
    log_info "Testing Shared Memory Architecture..."

    # 等待服务启动
    sleep 15

    # 测试健康检查
    log_info "Testing health check..."
    if curl -s http://localhost:$API_PORT/health | grep -q "healthy"; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi

    # 测试统计信息
    log_info "Testing stats endpoint..."
    if curl -s http://localhost:$API_PORT/stats | grep -q "workers"; then
        log_success "Stats endpoint passed"
    else
        log_warning "Stats endpoint failed (might be starting up)"
    fi

    log_success "Architecture test completed"
}

# 显示帮助信息
show_help() {
    echo "Shared Memory Architecture Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --test     Test the architecture after starting"
    echo "  --clean    Only cleanup old processes"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  WORKERS_PER_GPU    Number of workers per GPU (default: 2)"
    echo "  API_PORT           API port (default: 5001)"
    echo "  LOG_LEVEL          Log level (default: INFO)"
    echo "  MAX_FILE_SIZE      Max file size in MB (default: 50)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Start with default settings"
    echo "  WORKERS_PER_GPU=4 $0                 # Start with 4 workers per GPU"
    echo "  API_PORT=8080 $0                     # Start on port 8080"
    echo "  $0 --test                            # Start and test"
    echo "  $0 --clean                           # Only cleanup"
}

# 主函数
main() {
    echo "========================================"
    echo "  Shared Memory Architecture"
    echo "========================================"
    echo ""

    # 解析命令行参数
    case "${1:-}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --clean)
            cleanup
            exit 0
            ;;
        --test)
            TEST_AFTER_START=true
            ;;
        "")
            TEST_AFTER_START=false
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac

    # 执行步骤
    check_dependencies
    cleanup
    start_architecture

    if [ "$TEST_AFTER_START" = true ]; then
        test_architecture
    fi
}

# 捕获中断信号
trap 'log_info "Interrupted by user"; cleanup; exit 0' INT TERM

# 运行主函数
main "$@"