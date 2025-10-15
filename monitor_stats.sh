#!/bin/bash
# 统计监控脚本 - 实时显示多实例统计数据

set -e

# 配置
LOAD_BALANCER_URL="http://localhost:5001"
MONITOR_INTERVAL=10  # 监控间隔（秒）
SHOW_DETAILS=false  # 是否显示详细信息

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

# 检查依赖
check_dependencies() {
    local missing_tools=()

    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "缺少必要工具: ${missing_tools[*]}"
        log_info "请安装缺少的工具后重试"
        exit 1
    fi
}

# 获取统计数据
get_stats() {
    local url="$1"
    local response

    response=$(curl -s "$url/stats" 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        echo "$response"
    else
        echo "null"
    fi
}

# 格式化数字
format_number() {
    local num=$1
    if [ "$num" -gt 1000000 ]; then
        echo "$(echo "scale=1; $num/1000000" | bc 2>/dev/null)M"
    elif [ "$num" -gt 1000 ]; then
        echo "$(echo "scale=1; $num/1000" | bc 2>/dev/null)K"
    else
        echo "$num"
    fi
}

# 显示汇总统计
show_summary_stats() {
    local stats_data="$1"

    if [ "$stats_data" = "null" ] || [ -z "$stats_data" ]; then
        log_error "无法获取统计数据"
        return
    fi

    # 提取关键指标
    local status=$(echo "$stats_data" | jq -r '.load_balancer.status // "unknown"')
    local healthy=$(echo "$stats_data" | jq -r '.load_balancer.healthy_backends // 0')
    local total=$(echo "$stats_data" | jq -r '.load_balancer.total_backends // 0')
    local queue_len=$(echo "$stats_data" | jq -r '.load_balancer.queue_length // 0')
    local active_req=$(echo "$stats_data" | jq -r '.load_balancer.active_requests // 0')

    local total_req=$(echo "$stats_data" | jq -r '.aggregated_stats.total_requests // 0')
    local success_req=$(echo "$stats_data" | jq -r '.aggregated_stats.successful_requests // 0')
    local failed_req=$(echo "$stats_data" | jq -r '.aggregated_stats.failed_requests // 0')
    local success_rate=$(echo "$stats_data" | jq -r '.aggregated_stats.success_rate_percent // 0')

    local total_files=$(echo "$stats_data" | jq -r '.aggregated_stats.total_files_processed // 0')
    local total_size=$(echo "$stats_data" | jq -r '.aggregated_stats.total_file_size_mb // 0')
    local total_chunks=$(echo "$stats_data" | jq -r '.aggregated_stats.total_chunks_processed // 0')
    local avg_file_size=$(echo "$stats_data" | jq -r '.aggregated_stats.average_file_size_mb // 0')
    local avg_process_time=$(echo "$stats_data" | jq -r '.aggregated_stats.average_processing_time_seconds // 0')

    # 显示状态图标
    local status_icon="✅"
    if [ "$status" != "healthy" ]; then
        status_icon="⚠️"
    fi

    # 清屏并显示标题
    clear
    log_header "多实例 Whisper 统计监控"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "负载均衡器: $LOAD_BALANCER_URL"
    echo ""

    # 负载均衡器状态
    echo "🔄 负载均衡器状态"
    echo "   状态: $status_icon $status"
    echo "   健康实例: $healthy/$total"
    echo "   队列长度: $queue_len"
    echo "   活跃请求: $active_req"
    echo ""

    # 请求统计
    echo "📊 请求统计"
    echo "   总请求数: $(format_number $total_req)"
    echo "   成功请求: $(format_number $success_req)"
    echo "   失败请求: $(format_number $failed_req)"
    echo "   成功率: ${success_rate}%"
    echo ""

    # 文件处理统计
    echo "📁 文件处理统计"
    echo "   处理文件数: $(format_number $total_files)"
    echo "   总文件大小: $(echo "scale=1; $total_size/1024" | bc 2>/dev/null || echo "$total_size") GB"
    echo "   总 chunk 数: $(format_number $total_chunks)"
    echo "   平均文件大小: ${avg_file_size} MB"
    echo "   平均处理时间: ${avg_process_time}s"
    echo ""

    # 实例状态概览
    echo "🖥️  实例状态概览"
    echo "$stats_data" | jq -r '.instance_details[]? | "   实例 \(.instance_id // "unknown"): 端口\(.port // 0) GPU\(.gpu_device // "unknown") 状态\(.status // "unknown") 请求\(.request_stats.total_requests // 0) 成功\(.request_stats.successful_requests // 0)"' 2>/dev/null | while IFS= read -r line; do
        if [ -n "$line" ]; then
            echo "   $line"
        fi
    done
    echo ""
}

# 显示详细统计
show_detailed_stats() {
    local stats_data="$1"

    if [ "$stats_data" = "null" ] || [ -z "$stats_data" ]; then
        log_error "无法获取统计数据"
        return
    fi

    clear
    log_header "详细统计数据"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 显示完整的 JSON 数据（格式化）
    echo "$stats_data" | jq '.' 2>/dev/null || echo "$stats_data"
}

# 主监控循环
monitor_loop() {
    local iteration=0

    log_info "开始监控统计信息..."
    log_info "按 Ctrl+C 停止监控"
    log_info "按 'd' 切换详细/简洁模式"
    echo ""

    while true; do
        iteration=$((iteration + 1))

        # 获取统计数据
        stats_data=$(get_stats "$LOAD_BALANCER_URL")

        # 显示统计信息
        if [ "$SHOW_DETAILS" = true ]; then
            show_detailed_stats "$stats_data"
        else
            show_summary_stats "$stats_data"
        fi

        # 显示控制信息
        echo "监控次数: $iteration | 间隔: ${MONITOR_INTERVAL}s | 模式: $([ "$SHOW_DETAILS" = true ] && echo "详细" || echo "简洁") | 按 Ctrl+C 退出"

        # 等待下次更新
        sleep "$MONITOR_INTERVAL"
    done
}

# 显示帮助信息
show_help() {
    echo "统计监控脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -i, --interval SEC     监控间隔（秒），默认10秒"
    echo "  -l, --load-balancer URL 负载均衡器URL，默认http://localhost:5001"
    echo "  -d, --detailed        显示详细统计信息"
    echo "  -h, --help           显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                  # 使用默认参数监控"
    echo "  $0 -i 5              # 每5秒监控一次"
    echo "  $0 -d               # 显示详细统计"
    echo "  $0 -i 30 -l http://lb:5001  # 自定义URL和间隔"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        -l|--load-balancer)
            LOAD_BALANCER_URL="$2"
            shift 2
            ;;
        -d|--detailed)
            SHOW_DETAILS=true
            shift
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
    echo "🔍 统计监控脚本启动"
    echo "===================="
    echo "负载均衡器: $LOAD_BALANCER_URL"
    echo "监控间隔: ${MONITOR_INTERVAL}秒"
    echo "显示模式: $([ "$SHOW_DETAILS" = true ] && echo "详细" || echo "简洁")"
    echo ""

    # 检查依赖
    log_info "检查工具依赖..."
    check_dependencies

    # 检查服务可用性
    log_info "检查负载均衡器可用性..."
    if ! curl -s "$LOAD_BALANCER_URL/health" >/dev/null 2>&1; then
        log_error "无法连接到负载均衡器 ($LOAD_BALANCER_URL)"
        log_info "请确保服务正在运行"
        exit 1
    fi

    # 检查统计接口
    log_info "检查统计接口..."
    if ! curl -s "$LOAD_BALANCER_URL/stats" >/dev/null 2>&1; then
        log_warn "统计接口不可用，将使用健康检查接口"
    fi

    log_info "✅ 服务检查通过，开始监控..."
    echo ""

    # 开始监控
    monitor_loop
}

# 信号处理
trap 'echo ""; log_info "监控停止"; exit 0' INT TERM

# 执行主函数
main "$@"