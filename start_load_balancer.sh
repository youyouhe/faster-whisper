#!/bin/bash
# 启动负载均衡器脚本，连接到多GPU多实例架构

# Exit on any error
set -e

echo "🚀 启动Whisper多GPU负载均衡器"
echo "连接到已运行的多GPU实例服务"

# 配置参数
BACKEND_SERVICES="http://localhost:5002,http://localhost:5003,http://localhost:5004"
LB_PORT=5001
MAX_QUEUE_SIZE=100
HEALTH_CHECK_INTERVAL=30
REQUEST_TIMEOUT=1800

echo "📋 配置信息:"
echo "   后端服务: ${BACKEND_SERVICES}"
echo "   负载均衡器端口: $LB_PORT"
echo "   最大队列大小: $MAX_QUEUE_SIZE"
echo "   健康检查间隔: ${HEALTH_CHECK_INTERVAL}s"
echo "   请求超时: ${REQUEST_TIMEOUT}s"

# 检查后端服务是否运行
echo ""
echo "🔍 检查后端服务状态:"
for service in ${BACKEND_SERVICES//,/ }; do
    if curl -s "$service/health" > /dev/null; then
        echo "   ✅ $service: 健康"
    else
        echo "   ❌ $service: 未响应"
        echo "⚠️  请确保所有后端服务都已启动"
        exit 1
    fi
done

# 设置环境变量
export BACKEND_SERVICES="$BACKEND_SERVICES"
export LB_PORT="$LB_PORT"
export MAX_QUEUE_SIZE="$MAX_QUEUE_SIZE"
export HEALTH_CHECK_INTERVAL="$HEALTH_CHECK_INTERVAL"
export REQUEST_TIMEOUT="$REQUEST_TIMEOUT"

# 启动负载均衡器
echo ""
echo "🔄 启动负载均衡器..."
python load_balancer.py > "logs/load_balancer.log" 2>&1 &
lb_pid=$!

echo "✅ 负载均衡器已启动 (PID: $lb_pid, 端口: $LB_PORT)"
echo ""
echo "📝 日志文件: logs/load_balancer.log"
echo ""
echo "🌐 统一访问地址: http://localhost:$LB_PORT"
echo "🔍 健康检查: http://localhost:$LB_PORT/health"
echo "📈 推理接口: http://localhost:$LB_PORT/inference"
echo ""
echo "🎯 架构总览:"
echo "   负载均衡器 (端口 $LB_PORT) → 分发请求到后端GPU服务"
for service in ${BACKEND_SERVICES//,/ }; do
    port=$(echo $service | cut -d':' -f3)
    if [ "$port" = "5002" ]; then
        echo "   → $service (GPU 0 - 2个实例)"
    elif [ "$port" = "5003" ]; then
        echo "   → $service (GPU 1 - 2个实例)"
    elif [ "$port" = "5004" ]; then
        echo "   → $service (GPU 2 - 2个实例)"
    fi
done
echo ""
echo "⚠️  按 Ctrl+C 停止负载均衡器"

# 捕获退出信号
cleanup() {
    echo ""
    echo "🛑 停止负载均衡器..."
    if [[ -n "$lb_pid" ]]; then
        kill "$lb_pid" 2>/dev/null || true
        echo "✅ 负载均衡器已停止"
    fi
    exit 0
}

trap cleanup EXIT INT TERM

# 等待负载均衡器进程
wait $lb_pid