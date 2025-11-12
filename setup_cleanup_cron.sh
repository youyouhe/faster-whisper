#!/bin/bash

# 设置定时清理任务的脚本
# 将清理任务添加到crontab中

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup_tasks.py"
LOG_FILE="$SCRIPT_DIR/logs/cleanup.log"

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

echo "设置定时清理任务..."

# 检查清理脚本是否存在
if [ ! -f "$CLEANUP_SCRIPT" ]; then
    echo "错误: 清理脚本不存在 $CLEANUP_SCRIPT"
    exit 1
fi

# 创建crontab任务
# 每天凌晨2点执行清理任务
CRON_JOB="0 2 * * * cd $SCRIPT_DIR && /usr/bin/python3 $CLEANUP_SCRIPT >> $LOG_FILE 2>&1"

# 检查是否已经存在相同的任务
if crontab -l 2>/dev/null | grep -q "$CLEANUP_SCRIPT"; then
    echo "定时清理任务已存在，正在更新..."
    # 删除旧的任务
    crontab -l 2>/dev/null | grep -v "$CLEANUP_SCRIPT" | crontab -
fi

# 添加新的任务
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "定时清理任务设置完成!"
echo "任务将在每天凌晨2点执行"
echo "日志文件: $LOG_FILE"
echo ""
echo "当前crontab任务:"
crontab -l | grep cleanup_tasks