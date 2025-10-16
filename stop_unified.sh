#!/bin/bash
# 停止所有unified服务

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info "Stopping all unified processes..."

# 查找并终止进程
PIDS=$(pgrep -f "unified_api.py\|load_balancer.py" || true)
if [ ! -z "$PIDS" ]; then
    log_info "Found processes: $PIDS"
    kill $PIDS 2>/dev/null || true
    sleep 3

    # 强制终止仍在运行的进程
    REMAINING=$(pgrep -f "unified_api.py\|load_balancer.py" || true)
    if [ ! -z "$REMAINING" ]; then
        log_error "Force killing remaining processes: $REMAINING"
        kill -9 $REMAINING 2>/dev/null || true
    fi

    log_info "All processes stopped"
else
    log_info "No unified processes found"
fi

# 清理共享内存
log_info "Cleaning up shared memory..."
rm -f /dev/shm/whisper_* 2>/dev/null || true

log_info "Cleanup completed"
