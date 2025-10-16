#!/bin/bash
# 测试统一多GPU架构Docker构建脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_info "=== Testing Unified Multi-GPU Docker Build ==="

# 检查必要文件
log_info "Checking required files..."

required_files=(
    "docker-compose-unified.yml"
    "docker/Dockerfile.unified"
    "unified_api.py"
    "master_process.py"
    "worker_process.py"
    "shared_memory_manager.py"
    "load_balancer.py"
    "start_unified_multi_gpu.sh"
    "deploy_multi_gpu.sh"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    log_error "Missing required files:"
    for file in "${missing_files[@]}"; do
        log_error "  - $file"
    done
    exit 1
fi

log_info "✅ All required files found"

# 检查Docker是否可用
if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Please install Docker."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose not found. Please install Docker Compose."
    exit 1
fi

log_info "✅ Docker and Docker Compose found"

# 检查Docker是否可以访问NVIDIA运行时
if ! docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log_warn "NVIDIA Docker runtime not available. GPU support may be limited."
    log_warn "Please ensure nvidia-docker2 or NVIDIA Container Toolkit is installed."
fi

# 验证脚本权限
log_info "Checking script permissions..."
scripts=("deploy_multi_gpu.sh" "start_unified_multi_gpu.sh")
for script in "${scripts[@]}"; do
    if [ -x "$script" ]; then
        log_info "✅ $script is executable"
    else
        log_warn "$script is not executable, making it executable..."
        chmod +x "$script"
    fi
done

# 验证Docker Compose配置
log_info "Validating Docker Compose configuration..."
if docker-compose -f docker-compose-unified.yml config > /dev/null 2>&1; then
    log_info "✅ Docker Compose configuration is valid"
else
    log_error "❌ Docker Compose configuration is invalid"
    docker-compose -f docker-compose-unified.yml config
    exit 1
fi

# 显示构建计划
log_info "=== Build Plan ==="
log_info "1. Build faster-whisper-unified image using Dockerfile.unified"
log_info "2. Start services using docker-compose-unified.yml"
log_info "3. Services included:"
log_info "   - redis: Message queue"
log_info "   - tus-api-server: API server (port 8000)"
log_info "   - tus-server: TUS upload server (port 1080)"
log_info "   - callback-service: Callback service"
log_info "   - faster-whisper-unified: New unified multi-GPU architecture"
log_info "   - asr-worker: ASR worker (port 8081)"

log_info ""
log_info "Unified architecture features:"
log_info "   - Each GPU runs its own unified_api.py instance"
log_info "   - Compatible with existing load_balancer.py"
log_info "   - Shared memory architecture for performance"
log_info "   - Master-worker process management"
log_info "   - Health checks and monitoring"

log_info ""
log_info "To build and start the new architecture:"
log_info "  docker-compose -f docker-compose-unified.yml build"
log_info "  docker-compose -f docker-compose-unified.yml up -d"

log_info ""
log_info "To check logs:"
log_info "  docker-compose -f docker-compose-unified.yml logs -f faster-whisper-unified"

log_info ""
log_info "To stop services:"
log_info "  docker-compose -f docker-compose-unified.yml down"

log_info ""
log_info "=== Test Complete ==="
log_info "✅ All checks passed. Ready to build unified multi-GPU architecture."