#!/bin/bash
# Multi-GPU Multi-Instance Docker Deployment Script
# 基于docker-compose.multi-gpu-instance.yml的部署方案

# Exit on any error
set -e

echo "🚀 启动多GPU多实例Whisper Docker服务"
echo "基于docker-compose.multi-gpu-instance.yml架构"

# 配置参数
GPU_COUNT=${GPU_COUNT:-3}  # GPU数量
INSTANCES_PER_GPU=${INSTANCES_PER_GPU:-4}  # 每GPU实例数
MODEL_SIZE=${MODEL_SIZE:-large-v3-turbo}  # 模型大小
COMPUTE_TYPE=${COMPUTE_TYPE:-int8}  # 计算类型

echo ""
echo "📋 配置信息:"
echo "   GPU数量: $GPU_COUNT"
echo "   每GPU实例数: $INSTANCES_PER_GPU"
echo "   总实例数: $((GPU_COUNT * INSTANCES_PER_GPU))"
echo "   模型: $MODEL_SIZE"
echo "   计算类型: $COMPUTE_TYPE"

# 检查GPU是否可用
echo ""
echo "🔍 检查GPU状态..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi 未找到，请确保安装了NVIDIA驱动"
    exit 1
fi

# 检查GPU数量
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "✅ 检测到 $AVAILABLE_GPUS 个GPU"

if [ "$AVAILABLE_GPUS" -lt "$GPU_COUNT" ]; then
    echo "⚠️  警告: 检测到 $AVAILABLE_GPUS 个GPU，但配置要求 $GPU_COUNT 个"
    echo "   将使用 $AVAILABLE_GPUS 个GPU"
    GPU_COUNT=$AVAILABLE_GPUS
fi

# 检查Docker和Docker Compose
echo ""
echo "🔍 检查Docker环境..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未找到，请安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未找到，请安装Docker Compose"
    exit 1
fi

# 检查NVIDIA Docker支持
echo "🔍 检查NVIDIA Docker支持..."
if docker info | grep -q "nvidia" || command -v nvidia-docker &> /dev/null; then
    echo "✅ NVIDIA Docker 运行时已配置"
else
    echo "❌ NVIDIA Docker 运行时未配置，请安装nvidia-docker2"
    echo "   安装命令: sudo apt-get install nvidia-docker2"
    echo "   重启Docker: sudo systemctl restart docker"
    exit 1
fi

echo "✅ Docker环境检查通过"

# 创建必要的目录
echo ""
echo "📁 创建必要的目录..."
mkdir -p logs models data srt_results
echo "✅ 目录创建完成"

# 设置环境变量
export API_KEY=${API_KEY:-your-secret-api-key-here}
export INSTANCES_PER_GPU=$INSTANCES_PER_GPU
export MODEL_SIZE=$MODEL_SIZE
export COMPUTE_TYPE=$COMPUTE_TYPE

echo ""
echo "🔧 启动服务..."

# 启动Docker Compose
docker-compose -f docker-compose.multi-gpu-instance.yml up --build -d

echo ""
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo ""
echo "🔍 检查服务状态..."

# 检查负载均衡器（统一入口）
if curl -s http://localhost:5001/health > /dev/null; then
    echo "✅ 负载均衡器 (端口 5001) - 统一入口: 健康"
else
    echo "❌ 负载均衡器 (端口 5001) - 统一入口: 未响应"
fi

# 检查多GPU多实例服务端口
for ((i=0; i<GPU_COUNT; i++)); do
    port=$((5002 + i))  # 5002, 5003, 5004
    if curl -s "http://localhost:$port/health" > /dev/null; then
        echo "✅ 多GPU服务 GPU$i (端口 $port): 可访问"
    else
        echo "❌ 多GPU服务 GPU$i (端口 $port): 未响应"
    fi
done

# 检查其他服务
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ TUS API服务器 (端口 8000): 健康"
else
    echo "❌ TUS API服务器 (端口 8000): 未响应"
fi

if curl -s http://localhost:1080/health > /dev/null; then
    echo "✅ TUS服务器 (端口 1080): 健康"
else
    echo "❌ TUS服务器 (端口 1080): 未响应"
fi

# 显示GPU内存使用情况
echo ""
echo "💾 GPU内存使用情况:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "📊 容器状态:"
docker-compose -f docker-compose.multi-gpu-instance.yml ps

echo ""
echo "✅ 多GPU多实例服务启动完成!"
echo ""
echo "🌐 服务地址:"
echo "   主API (负载均衡器统一入口): http://localhost:5001"
echo "   TUS API服务器: http://localhost:8000"
echo "   TUS文件服务器: http://localhost:1080"
echo ""
echo "🔍 健康检查和监控:"
echo "   负载均衡器状态: http://localhost:5001/health"
echo "   服务统计信息: http://localhost:5001/stats"
echo ""
echo "📝 日志查看:"
echo "   所有服务日志: docker-compose -f docker-compose.multi-gpu-instance.yml logs -f"
echo "   特定服务日志: docker-compose -f docker-compose.multi-gpu-instance.yml logs -f [service-name]"
echo ""
echo "🛑 停止服务:"
echo "   docker-compose -f docker-compose.multi-gpu-instance.yml down"
echo ""
echo "⚠️  注意事项:"
echo "   - 负载均衡器(5001)是唯一对外入口，提供统一的API接口"
echo "   - 单容器支持所有GPU，简化部署和管理"
echo "   - 确保有足够的GPU内存运行 $((GPU_COUNT * INSTANCES_PER_GPU)) 个实例"
echo "   - 首次启动需要下载模型，可能需要较长时间"
echo "   - 负载均衡器自动分发请求到多GPU服务端口"