#!/bin/bash

echo "🧪 异步API测试脚本 (curl版本)"
echo "确保服务运行在 http://localhost:5020"
echo

# 检查服务状态
echo "1. 检查服务状态..."
curl -s http://localhost:5020/health | python3 -m json.tool || {
    echo "❌ 服务未运行，请先启动服务"
    exit 1
}

echo

# 寻找测试文件
TEST_FILE=$(ls *.wav 2>/dev/null | head -1)
if [ -z "$TEST_FILE" ]; then
    echo "❌ 未找到测试WAV文件，请将WAV文件放在当前目录"
    exit 1
fi

echo "2. 使用测试文件: $TEST_FILE"

# 提交异步任务
echo
echo "3. 提交异步任务 (SRT格式)..."
SUBMIT_RESULT=$(curl -s -X POST "http://localhost:5020/transcribe_async" \
    -F "file=@$TEST_FILE" \
    -F "language=auto" \
    -F "response_format=srt" \
    -F "callback_url=http://httpbin.org/post" \
    --max-time 30)

echo "提交结果:"
echo "$SUBMIT_RESULT" | python3 -m json.tool

# 提取task_id
TASK_ID=$(echo "$SUBMIT_RESULT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('task_id', ''))" 2>/dev/null)

if [ -z "$TASK_ID" ]; then
    echo "❌ 提交失败，无法获取task_id"
    exit 1
fi

echo
echo "4. 任务ID: $TASK_ID"

# 监控任务状态
echo
echo "5. 监控任务状态..."
MAX_CHECKS=60  # 最多检查5分钟
CHECK_INTERVAL=5

for i in $(seq 1 $MAX_CHECKS); do
    echo -n "检查 $i/$MAX_CHECKS: "

    STATUS_RESULT=$(curl -s "http://localhost:5020/task/$TASK_ID" --max-time 10)

    if echo "$STATUS_RESULT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', ''))" 2>/dev/null | grep -q "completed"; then
        echo "✅ 任务完成!"
        echo "$STATUS_RESULT" | python3 -m json.tool
        break
    elif echo "$STATUS_RESULT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', ''))" 2>/dev/null | grep -q "failed"; then
        echo "❌ 任务失败!"
        echo "$STATUS_RESULT" | python3 -m json.tool
        break
    else
        STATUS=$(echo "$STATUS_RESULT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null)
        echo "状态: $status"
        sleep $CHECK_INTERVAL
    fi

    if [ $i -eq $MAX_CHECKS ]; then
        echo "⏰ 检查超时，任务可能仍在处理中"
    fi
done

# 验证SRT格式结果
echo
echo "6. 验证SRT格式结果..."
if echo "$STATUS_RESULT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', ''))" 2>/dev/null | grep -q "completed"; then
    echo "✅ 任务完成，检查SRT格式..."

    # 提取并显示SRT内容前几行
    SRT_CONTENT=$(echo "$STATUS_RESULT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', {}).get('text', ''))" 2>/dev/null)

    if [ ! -z "$SRT_CONTENT" ]; then
        echo "📝 SRT内容预览 (前8行):"
        echo "$SRT_CONTENT" | head -8
        echo ""
        echo "📊 SRT统计信息:"
        SEGMENT_COUNT=$(echo "$SRT_CONTENT" | grep -c "^[0-9]*$" || echo "0")
        echo "   字幕段落数: $SEGMENT_COUNT"
    else
        echo "❌ 未找到SRT内容"
    fi
fi

echo
echo "7. 测试完成"