#!/bin/bash
# GPU利用率监控脚本 - 用于多实例部署的性能监控

set -e

# 配置参数
LOAD_BALANCER_URL="http://localhost:5001"
MONITOR_INTERVAL=10  # 监控间隔（秒）
LOG_FILE="/tmp/gpu_monitor_$(date +%Y%m%d_%H%M%S).log"
DURATION=3600  # 监控时长（秒），默认1小时

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    local msg="ℹ️  $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $msg" >> "$LOG_FILE"
}

log_warn() {
    local msg="⚠️  $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $msg" >> "$LOG_FILE"
}

log_error() {
    local msg="❌ $1"
    echo -e "${RED}${msg}${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $msg" >> "$LOG_FILE"
}

log_debug() {
    local msg="🔍 $1"
    echo -e "${BLUE}${msg}${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $msg" >> "$LOG_FILE"
}

# 检查工具依赖
check_dependencies() {
    local missing_tools=()

    command -v nvidia-smi >/dev/null 2>&1 || missing_tools+=("nvidia-smi")
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "缺少必要工具: ${missing_tools[*]}"
        log_info "请安装缺少的工具后重试"
        exit 1
    fi
}

# 获取GPU信息
get_gpu_info() {
    local output=""

    if command -v nvidia-smi >/dev/null 2>&1; then
        # 获取GPU基本信息
        local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null || echo "0")

        if [ "$gpu_count" -gt 0 ]; then
            # 获取详细的GPU利用率信息
            output=$(nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
        fi
    fi

    echo "$output"
}

# 获取负载均衡器状态
get_lb_status() {
    local status=""

    if curl -s "$LOAD_BALANCER_URL/health" >/dev/null 2>&1; then
        if command -v jq >/dev/null 2>&1; then
            local response=$(curl -s "$LOAD_BALANCER_URL/health")
            local healthy=$(echo "$response" | jq -r '.healthy_backends // 0')
            local total=$(echo "$response" | jq -r '.total_backends // 0')
            local queue_len=$(echo "$response" | jq -r '.queue_length // 0')

            status="LB: ${healthy}/${total} backends, Queue: ${queue_len}"
        else
            status="LB: OK"
        fi
    else
        status="LB: ERROR"
    fi

    echo "$status"
}

# 分析GPU利用率
analyze_gpu_utilization() {
    local gpu_info="$1"
    local total_gpu=0
    local avg_utilization=0
    local avg_memory=0
    local active_gpus=0

    if [ -n "$gpu_info" ]; then
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                # 解析GPU信息：index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw
                local gpu_util=$(echo "$line" | cut -d',' -f3)
                local mem_util=$(echo "$line" | cut -d',' -f4)

                if [[ "$gpu_util" =~ ^[0-9]+$ ]]; then
                    total_gpu=$((total_gpu + gpu_util))
                    active_gpus=$((active_gpus + 1))
                fi

                if [[ "$mem_util" =~ ^[0-9]+$ ]]; then
                    avg_memory=$((avg_memory + mem_util))
                fi
            fi
        done <<< "$gpu_info"

        if [ "$active_gpus" -gt 0 ]; then
            avg_utilization=$((total_gpu / active_gpus))
            avg_memory=$((avg_memory / active_gpus))
        fi
    fi

    echo "${avg_utilization},${avg_memory},${active_gpus}"
}

# 生成性能报告
generate_report() {
    local log_file="$1"
    local report_file="${log_file%.log}_report.txt"

    log_info "生成性能报告: $report_file"

    cat > "$report_file" << EOF
GPU多实例监控报告
=================
监控时间段: $(head -1 "$log_file" | cut -d' ' -f1-2) - $(tail -1 "$log_file" | cut -d' ' -f1-2)
监控间隔: ${MONITOR_INTERVAL}秒
总监控时长: $DURATION 秒

性能统计:
--------
EOF

    # 统计平均GPU利用率
    if grep "GPU利用率" "$log_file" >/dev/null 2>&1; then
        echo "" >> "$report_file"
        echo "平均GPU利用率分析:" >> "$report_file"
        grep "GPU利用率" "$log_file" | cut -d':' -f2 | sed 's/%//' | awk '{sum+=$1; count++} END {if(count>0) printf "  平均利用率: %.1f%%\n", sum/count; printf "  采样次数: %d\n", count}' >> "$report_file"
    fi

    # 统计最大利用率时间点
    echo "" >> "$report_file"
    echo "利用率峰值:" >> "$report_file"
    grep "GPU利用率" "$log_file" | sort -t':' -k2 -nr | head -3 | while IFS= read -r line; do
        echo "  $line" >> "$report_file"
    done

    # 负载均衡器统计
    echo "" >> "$report_file"
    echo "负载均衡器状态:" >> "$report_file"
    grep "LB:" "$log_file" | awk '{print $3, $4}' | sort | uniq -c | sort -nr | while read count status; do
        echo "  $status: $count 次" >> "$report_file"
    done

    log_info "报告生成完成: $report_file"
}

# 主监控循环
monitor_loop() {
    local start_time=$(date +%s)
    local end_time=$((start_time + DURATION))
    local iteration=0

    log_info "开始GPU监控 (时长: ${DURATION}秒, 间隔: ${MONITOR_INTERVAL}秒)"
    log_info "日志文件: $LOG_FILE"
    echo ""

    # 显示监控表头
    echo -e "${BLUE}时间                    GPU利用率(%)  内存利用率(%)  活动GPU数  负载均衡器状态${NC}"
    echo "--------------------------------------------------------------------------------"

    while [ $(date +%s) -lt $end_time ]; do
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local iteration_display=$((iteration + 1))

        # 获取GPU信息
        local gpu_info=$(get_gpu_info)

        # 获取负载均衡器状态
        local lb_status=$(get_lb_status)

        # 分析GPU利用率
        local analysis=$(analyze_gpu_utilization "$gpu_info")
        local avg_gpu_util=$(echo "$analysis" | cut -d',' -f1)
        local avg_mem_util=$(echo "$analysis" | cut -d',' -f2)
        local active_gpus=$(echo "$analysis" | cut -d',' -f3)

        # 显示实时状态
        printf "%-22s  %-12s  %-12s  %-8s  %s\n" \
            "$timestamp ($iteration_display)" \
            "${avg_gpu_util}%" \
            "${avg_mem_util}%" \
            "$active_gpus" \
            "$lb_status"

        # 记录到日志文件
        log_debug "GPU利用率: ${avg_gpu_util}%, 内存利用率: ${avg_mem_util}%, 活动GPU: $active_gpus, $lb_status"

        # 显示详细GPU信息（每5次迭代显示一次）
        if [ $((iteration % 5)) -eq 0 ] && [ -n "$gpu_info" ]; then
            log_debug "详细GPU信息:"
            while IFS= read -r line; do
                if [ -n "$line" ]; then
                    log_debug "  $line"
                fi
            done <<< "$gpu_info"
        fi

        iteration=$((iteration + 1))
        sleep "$MONITOR_INTERVAL"
    done

    echo ""
    log_info "监控完成，共进行了 $iteration 次采样"

    # 生成报告
    generate_report "$LOG_FILE"
}

# 显示帮助信息
show_help() {
    echo "GPU多实例监控脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -i, --interval SEC     监控间隔（秒），默认10秒"
    echo "  -d, --duration SEC     监控时长（秒），默认3600秒（1小时）"
    echo "  -l, --log-file FILE    日志文件路径"
    echo "  -h, --help             显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                     # 使用默认参数监控1小时"
    echo "  $0 -i 5 -d 1800        # 每5秒监控一次，持续30分钟"
    echo "  $0 -l custom.log       # 指定自定义日志文件"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -l|--log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主函数
main() {
    echo "GPU多实例利用率监控脚本"
    echo "========================"
    echo "监控配置:"
    echo "  监控间隔: ${MONITOR_INTERVAL}秒"
    echo "  监控时长: ${DURATION}秒 ($(echo "$DURATION/3600" | bc -l)小时)"
    echo "  日志文件: $LOG_FILE"
    echo "  负载均衡器: $LOAD_BALANCER_URL"
    echo ""

    # 检查依赖
    log_info "检查工具依赖..."
    check_dependencies

    # 检查服务可用性
    log_info "检查服务可用性..."
    if ! curl -s "$LOAD_BALANCER_URL/health" >/dev/null 2>&1; then
        log_error "无法连接到负载均衡器 ($LOAD_BALANCER_URL)"
        log_info "请确保服务正在运行"
        exit 1
    fi

    log_info "✅ 服务检查通过"
    echo ""

    # 开始监控
    monitor_loop

    log_info "监控完成！详细日志保存在: $LOG_FILE"
}

# 执行主函数
main "$@"