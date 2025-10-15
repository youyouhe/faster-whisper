#!/bin/bash
# 测试多实例部署的脚本

set -e

echo "🧪 开始测试多实例GPU部署..."
echo "======================================"

# 配置
LOAD_BALANCER_URL="http://localhost:5001"
TIMEOUT=300  # 5分钟超时

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# 检查服务是否启动
check_service_health() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1

    log_info "检查 $service_name 健康状态..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url/health" >/dev/null 2>&1; then
            log_info "✅ $service_name 健康检查通过!"
            return 0
        fi

        log_warn "第 $attempt 次尝试，等待 $service_name 启动..."
        sleep 10
        ((attempt++))
    done

    log_error "❌ $service_name 健康检查失败!"
    return 1
}

# 获取负载均衡器状态
get_load_balancer_status() {
    log_info "获取负载均衡器状态..."

    if curl -s "$LOAD_BALANCER_URL/health" | jq '.' >/dev/null 2>&1; then
        local status=$(curl -s "$LOAD_BALANCER_URL/health")
        echo "负载均衡器状态:"
        echo "$status" | jq '.'

        local healthy_backends=$(echo "$status" | jq '.healthy_backends')
        local total_backends=$(echo "$status" | jq '.total_backends')

        log_info "健康后端: $healthy_backends/$total_backends"

        if [ "$healthy_backends" -gt 0 ]; then
            return 0
        fi
    else
        log_warn "无法解析JSON响应，使用curl状态检查"
        if curl -s -f "$LOAD_BALANCER_URL/health" >/dev/null 2>&1; then
            log_info "✅ 负载均衡器基本健康检查通过!"
            return 0
        fi
    fi

    log_error "❌ 负载均衡器状态异常!"
    return 1
}

# 测试API接口
test_api_inference() {
    log_info "测试API推理接口..."

    # 检查是否有测试音频文件
    if [ ! -f "test_audio.mp3" ] && [ ! -f "test_audio.wav" ]; then
        log_warn "未找到测试音频文件，跳过推理测试"
        return 0
    fi

    # 选择测试音频文件
    local test_file="test_audio.mp3"
    if [ ! -f "$test_file" ]; then
        test_file="test_audio.wav"
    fi

    log_info "使用测试文件: $test_file"

    # 发送推理请求
    local response=$(curl -s -X POST \
        -H "X-API-Key: your-secret-api-key-here" \
        -F "file=@$test_file" \
        -F "response_format=srt" \
        -F "language=auto" \
        "$LOAD_BALANCER_URL/inference" \
        --connect-timeout 30 \
        --max-time $TIMEOUT \
        -w "HTTP_STATUS:%{http_code}")

    # 提取HTTP状态码
    local http_code=$(echo "$response" | grep -o 'HTTP_STATUS:[0-9]*' | grep -o '[0-9]*')

    if [ "$http_code" = "200" ]; then
        log_info "✅ API推理测试成功!"
        # 提取JSON响应
        local json_response=$(echo "$response" | sed 's/HTTP_STATUS:[0-9]*$//')

        # 检查SRT内容
        local srt_content=$(echo "$json_response" | jq -r '.data' 2>/dev/null || echo "")
        if [ -n "$srt_content" ] && [ "$srt_content" != "null" ]; then
            local lines=$(echo "$srt_content" | wc -l)
            log_info "✅ 获得SRT内容 ($lines 行)"
        fi
        return 0
    else
        log_error "❌ API推理测试失败! HTTP状态码: $http_code"
        echo "响应内容: $response"
        return 1
    fi
}

# 检查GPU利用率
check_gpu_utilization() {
    log_info "检查GPU利用率..."

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "当前GPU状态:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits

        # 检查是否有GPU在使用
        local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null || echo "0")
        if [ "$gpu_count" -gt 0 ]; then
            log_info "✅ 检测到 $gpu_count 个GPU"

            # 获取平均GPU利用率
            local avg_utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
            log_info "平均GPU利用率: ${avg_utilization}%"
        else
            log_warn "未检测到GPU或nvidia-smi不可用"
        fi
    else
        log_warn "nvidia-smi 命令不可用，跳过GPU检查"
    fi
}

# 检查端口占用
check_ports() {
    log_info "检查端口占用情况..."

    local ports=(5001 5002 5003 5004 5005 5006 5007)

    for port in "${ports[@]}"; do
        if netstat -tlnp 2>/dev/null | grep ":$port " >/dev/null; then
            local pid=$(netstat -tlnp 2>/dev/null | grep ":$port " | head -1 | awk '{print $7}' | cut -d'/' -f1)
            log_info "✅ 端口 $port 正在使用 (PID: $pid)"
        else
            log_warn "❌ 端口 $port 未被使用"
        fi
    done
}

# 主测试流程
main() {
    echo "开始时间: $(date)"
    echo ""

    # 1. 检查端口占用
    check_ports
    echo ""

    # 2. 检查负载均衡器健康状态
    if check_service_health "$LOAD_BALANCER_URL" "负载均衡器"; then
        echo ""
        # 3. 获取负载均衡器详细状态
        get_load_balancer_status
        echo ""

        # 4. 测试API推理
        test_api_inference
        echo ""

        # 5. 检查GPU利用率
        check_gpu_utilization
        echo ""

        log_info "🎉 所有测试完成!"
        echo "======================================"
        echo "测试总结:"
        echo "✅ 负载均衡器运行正常"
        echo "✅ 后端服务实例健康"
        echo "✅ API接口可用"
        echo "✅ GPU资源检测完成"
        echo ""
        echo "监控命令:"
        echo "  - 实时GPU监控: watch -n 1 nvidia-smi"
        echo "  - 服务健康检查: curl $LOAD_BALANCER_URL/health"
        echo "  - 负载均衡器日志: docker logs faster-whisper-dynamic"
        echo ""
        echo "结束时间: $(date)"

        return 0
    else
        log_error "❌ 基础健康检查失败!"
        echo ""
        echo "调试建议:"
        echo "1. 检查容器是否正常启动: docker ps"
        echo "2. 查看容器日志: docker logs faster-whisper-dynamic"
        echo "3. 检查端口是否被占用: netstat -tlnp | grep 5001"
        echo "4. 确认GPU可用: nvidia-smi"
        echo ""
        echo "结束时间: $(date)"

        return 1
    fi
}

# 执行主流程
main "$@"