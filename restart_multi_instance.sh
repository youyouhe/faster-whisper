#!/bin/bash

echo "🔄 重启多实例Whisper服务..."

# 查找并停止当前服务
echo "停止当前服务..."
pkill -f "run_multi_instance_local.py"
pkill -f "multi_instance_api.py"

# 等待进程完全停止
sleep 2

# 检查端口是否释放
if lsof -i :5001 > /dev/null 2>&1; then
    echo "警告: 端口5001仍被占用，强制终止..."
    lsof -ti :5001 | xargs kill -9
    sleep 1
fi

echo "启动新的多实例服务..."
# 启动服务
python run_multi_instance_local.py