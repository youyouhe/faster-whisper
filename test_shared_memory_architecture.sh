#!/bin/bash
# 测试Shared Memory单端口多进程架构

# Exit on any error
set -e

# 配置
NUM_GPUS=${NUM_GPUS:-4}
START_PORT=${START_PORT:-5002}
TEST_FILE=${TEST_FILE:-""}
API_KEY=${API_KEY:-"your-secret-api-key-here"}

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

# 测试函数
test_health_check() {
    log_header "🏥 Health Check Test"

    local healthy_count=0
    local total_count=0

    for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
        port=$((START_PORT + gpu_id))
        total_count=$((total_count + 1))

        echo "Testing GPU $gpu_id (port $port)..."
        if curl -f "http://localhost:$port/health" > /dev/null 2>&1; then
            log_info "✅ GPU $gpu_id is healthy"
            healthy_count=$((healthy_count + 1))
        else
            log_error "❌ GPU $gpu_id is not responding"
        fi
    done

    echo ""
    log_info "Health check results: $healthy_count/$total_count services healthy"

    if [[ $healthy_count -eq $total_count ]]; then
        log_info "✅ All services are healthy!"
        return 0
    else
        log_error "❌ Some services are not healthy"
        return 1
    fi
}

test_memory_pool() {
    log_header "💾 Shared Memory Pool Test"

    for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
        port=$((START_PORT + gpu_id))

        echo "Testing memory pool for GPU $gpu_id..."
        stats=$(curl -s "http://localhost:$port/stats" 2>/dev/null)

        if [[ -n "$stats" ]]; then
            # 提取内存池统计
            pool_size=$(echo "$stats" | python3 -c "
import sys, json
data = json.load(sys.stdin)
pool = data.get('memory_pool', {})
print(f'Pool Size: {pool.get(\"pool_size_mb\", 0)}MB')
print(f'Free Chunks: {pool.get(\"free_chunks\", 0)}')
print(f'Used Size: {pool.get(\"used_size_mb\", 0):.2f}MB')
print(f'Active Tasks: {pool.get(\"active_tasks\", 0)}')
" 2>/dev/null || echo "Error parsing stats")

            echo "   $pool_size"
            echo "   $pool_size"
        else
            log_error "❌ Failed to get stats for GPU $gpu_id"
        fi
    done
}

test_worker_stats() {
    log_header "👥 Worker Statistics Test"

    for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
        port=$((START_PORT + gpu_id))

        echo "Checking workers for GPU $gpu_id..."
        stats=$(curl -s "http://localhost:$port/stats" 2>/dev/null)

        if [[ -n "$stats" ]]; then
            # 提取worker统计
            echo "$stats" | python3 -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', [])
if workers:
    for i, worker in enumerate(workers):
        print(f'   Worker {worker.get(\"worker_id\", i)}:')
        print(f'     GPU: {worker.get(\"gpu_id\", \"unknown\")}')
        print(f'     Tasks: {worker.get(\"tasks_processed\", 0)}')
        print(f'     Success Rate: {worker.get(\"success_rate\", 0)}%')
        print(f'     Avg Processing: {worker.get(\"average_processing_time\", 0):.2f}s')
        print(f'     Throughput: {worker.get(\"throughput_mb_per_hour\", 0):.2f}MB/h')
        print()
else:
    print('   No worker data available')
" 2>/dev/null || echo "Error parsing worker stats"
        else
            log_error "❌ Failed to get stats for GPU $gpu_id"
        fi
    done
}

test_inference() {
    if [[ -z "$TEST_FILE" ]]; then
        log_warn "⚠️  No test file specified. Skipping inference test."
        log_info "Use: $0 --test-file <audio_file_path>"
        return 0
    fi

    if [[ ! -f "$TEST_FILE" ]]; then
        log_error "❌ Test file not found: $TEST_FILE"
        return 1
    fi

    log_header "🎵 Inference Test"
    log_info "Testing file: $TEST_FILE"

    # 获取文件大小
    file_size=$(stat -f%z "$TEST_FILE" 2>/dev/null || stat -c%s "$TEST_FILE" 2>/dev/null || echo "0")
    file_size_mb=$((file_size / 1024 / 1024))

    echo "File size: ${file_size_mb}MB"

    # 选择一个可用的服务进行测试
    test_port=$START_PORT
    if ! curl -f "http://localhost:$test_port/health" > /dev/null 2>&1; then
        log_error "❌ No healthy services available for inference test"
        return 1
    fi

    echo "Testing inference on port $test_port..."

    # 执行推理请求
    start_time=$(date +%s.%N)

    response=$(curl -s -X POST \
        -H "X-API-Key: $API_KEY" \
        -F "file=@$TEST_FILE" \
        -F "response_format=srt" \
        -F "language=auto" \
        "http://localhost:$test_port/inference" 2>/dev/null)

    end_time=$(date +%s.%N)
    processing_time=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")

    if [[ -n "$response" ]]; then
        # 解析响应
        code=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('code', -1))
    print(data.get('msg', 'No message'))
    if 'data' in data:
        print(len(data['data']))
except:
    print(-1)
" 2>/dev/null)

        if [[ "$code" == "0" ]]; then
            log_info "✅ Inference test successful!"
            log_info "   Processing time: ${processing_time}s"
            log_info "   Response size: $(echo "$response" | wc -c) characters"
            return 0
        else
            log_error "❌ Inference test failed with code: $code"
            return 1
        fi
    else
        log_error "❌ No response from service"
        return 1
    fi
}

test_concurrent_inference() {
    if [[ -z "$TEST_FILE" ]]; then
        log_warn "⚠️  No test file specified. Skipping concurrent inference test."
        return 0
    fi

    log_header "🔄 Concurrent Inference Test"
    log_info "Testing concurrent requests..."

    # 启动并发请求
    local pids=()
    local success_count=0
    local total_count=2

    for i in $(seq 1 $total_count); do
        (
            echo "Starting concurrent request $i..."

            start_time=$(date +%s.%N)
            response=$(curl -s -X POST \
                -H "X-API-Key: $API_KEY" \
                -F "file=@$TEST_FILE" \
                -F "response_format=srt" \
                -F "language=auto" \
                "http://localhost:$START_PORT/inference" 2>/dev/null)

            end_time=$(date +%s.%N)
            processing_time=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")

            if [[ -n "$response" ]]; then
                code=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('code', -1))" 2>/dev/null)
                if [[ "$code" == "0" ]]; then
                    echo "✅ Request $i completed in ${processing_time}s"
                    exit 0
                else
                    echo "❌ Request $i failed with code: $code"
                    exit 1
                fi
            else
                echo "❌ Request $i failed: No response"
                exit 1
            fi
        ) &
        pids+=($!)

        # 短暂延迟避免同时开始
        sleep 0.1
    done

    # 等待所有请求完成
    for pid in "${pids[@]}"; do
        wait $pid
        if [[ $? -eq 0 ]]; then
            success_count=$((success_count + 1))
        fi
    done

    echo ""
    log_info "Concurrent inference results: $success_count/$total_count requests successful"

    if [[ $success_count -eq $total_count ]]; then
        log_info "✅ All concurrent requests succeeded!"
        return 0
    else
        log_error "❌ Some concurrent requests failed"
        return 1
    fi
}

# 主函数
main() {
    log_header "🧪 Shared Memory Architecture Test Suite"

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --test-file)
                TEST_FILE="$2"
                shift 2
                ;;
            --gpu-count)
                NUM_GPUS="$2"
                shift 2
                ;;
            --start-port)
                START_PORT="$2"
                shift 2
                ;;
            --api-key)
                API_KEY="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --test-file FILE     Audio file for inference tests"
                echo "  --gpu-count NUM     Number of GPUs (default: 4)"
                echo "  --start-port PORT   Starting port (default: 5002)"
                echo "  --api-key KEY       API key for authentication"
                echo "  -h, --help           Show this help message"
                echo ""
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # 显示配置
    log_info "Test Configuration:"
    echo "   🎯 GPUs: $NUM_GPUS"
    echo "   🚪 Starting Port: $START_PORT"
    echo "   📄 Test File: ${TEST_FILE:-'Not specified'}"
    echo ""

    # 运行测试
    local test_failed=0

    # 基础健康检查
    if ! test_health_check; then
        test_failed=1
    fi

    # 内存池测试
    if ! test_memory_pool; then
        test_failed=1
    fi

    # Worker统计测试
    if ! test_worker_stats; then
        test_failed=1
    fi

    # 推理测试（如果有测试文件）
    if [[ -n "$TEST_FILE" ]]; then
        if ! test_inference; then
            test_failed=1
        fi

        # 并发推理测试
        if ! test_concurrent_inference; then
            test_failed=1
        fi
    fi

    # 总结
    log_header "📊 Test Summary"
    if [[ $test_failed -eq 0 ]]; then
        log_info "✅ All tests passed successfully!"
        log_info "🎉 Shared Memory architecture is working correctly!"
        return 0
    else
        log_error "❌ Some tests failed!"
        log_info "Please check the logs and configuration."
        return 1
    fi
}

# 执行主函数
main "$@"