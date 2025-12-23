#!/bin/bash
# Run Label Correction Analysis Script
# 支持单个噪声率或批量运行多个噪声率

# Default values
NOISE_RATES=""
RETRAIN_BACKBONE=""
BATCH_MODE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print banner
print_banner() {
    print_message "$BLUE" "======================================================================"
    print_message "$BLUE" "  MEDAL-Lite 标签矫正分析"
    print_message "$BLUE" "======================================================================"
    echo ""
}

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --noise-rate RATE      单个噪声率 (例如: 0.30)"
    echo "  -b, --batch RATES          批量噪声率，空格分隔 (例如: \"0.1 0.2 0.3 0.4\")"
    echo "  -a, --all                  运行所有预设噪声率 (10%, 20%, 30%, 40%)"
    echo "  -r, --retrain-backbone     使用随机初始化的backbone"
    echo "  -h, --help                 显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  $0 -n 0.30                 # 运行30%噪声率分析"
    echo "  $0 -b \"0.1 0.2 0.3\"        # 批量运行10%, 20%, 30%"
    echo "  $0 -a                      # 运行所有预设噪声率"
    echo "  $0 -n 0.40 -r              # 40%噪声 + 随机backbone"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--noise-rate)
            NOISE_RATES="$2"
            shift 2
            ;;
        -b|--batch)
            NOISE_RATES="$2"
            BATCH_MODE=true
            shift 2
            ;;
        -a|--all)
            NOISE_RATES="0.1 0.2 0.3 0.4"
            BATCH_MODE=true
            shift
            ;;
        -r|--retrain-backbone)
            RETRAIN_BACKBONE="--retrain_backbone"
            shift
            ;;
        -h|--help)
            print_banner
            print_usage
            exit 0
            ;;
        *)
            print_message "$RED" "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Default to 30% if no noise rate specified
if [ -z "$NOISE_RATES" ]; then
    NOISE_RATES="0.3"
fi

# Print banner
print_banner

# Check if Python script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/MoudleCode/label_correction/analysis/label_correction_analysis.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_message "$RED" "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d "$HOME/anaconda3/envs/RAPIER" ]; then
    print_message "$GREEN" "激活conda环境: RAPIER"
    source "$HOME/anaconda3/bin/activate" RAPIER
    echo ""
elif [ -d "$HOME/anaconda3/envs/medal" ]; then
    print_message "$GREEN" "激活conda环境: medal"
    source "$HOME/anaconda3/bin/activate" medal
    echo ""
fi

# Convert noise rates to array
read -ra RATES_ARRAY <<< "$NOISE_RATES"

# Print configuration
print_message "$YELLOW" "配置信息:"
echo "  噪声率:            ${RATES_ARRAY[*]}"
echo "  批量模式:          $([ "$BATCH_MODE" = true ] && echo "是" || echo "否")"
echo "  重训练Backbone:    $([ -n "$RETRAIN_BACKBONE" ] && echo "是" || echo "否")"
echo "  脚本路径:          $PYTHON_SCRIPT"
echo ""

# Confirm execution
read -p "$(echo -e ${YELLOW}是否继续? [Y/n]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    print_message "$RED" "用户取消"
    exit 0
fi

# Create log directory
LOG_DIR="$SCRIPT_DIR/output/logs"
mkdir -p "$LOG_DIR"

# Track results
TOTAL_RATES=${#RATES_ARRAY[@]}
COMPLETED=0
FAILED=0
declare -a COMPLETED_RATES
declare -a FAILED_RATES

# Run analysis for each noise rate
cd "$SCRIPT_DIR"

for NOISE_RATE in "${RATES_ARRAY[@]}"; do
    NOISE_PCT=$(printf "%.0f" $(echo "$NOISE_RATE * 100" | bc))
    
    echo ""
    print_message "$CYAN" "======================================================================"
    print_message "$CYAN" "  开始分析噪声率: ${NOISE_PCT}%  (${COMPLETED}/${TOTAL_RATES} 已完成)"
    print_message "$CYAN" "======================================================================"
    echo ""
    
    # Run Python script
    python "$PYTHON_SCRIPT" \
        --noise_rate $NOISE_RATE \
        $RETRAIN_BACKBONE
    
    # Check exit status
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_message "$GREEN" "✓ 噪声率 ${NOISE_PCT}% 分析完成"
        COMPLETED=$((COMPLETED + 1))
        COMPLETED_RATES+=("${NOISE_PCT}%")
    else
        print_message "$RED" "✗ 噪声率 ${NOISE_PCT}% 分析失败"
        FAILED=$((FAILED + 1))
        FAILED_RATES+=("${NOISE_PCT}%")
    fi
done

# Print summary
echo ""
print_message "$BLUE" "======================================================================"
print_message "$BLUE" "  批量分析完成"
print_message "$BLUE" "======================================================================"
echo ""

print_message "$YELLOW" "📊 执行摘要:"
echo "  总计:     ${TOTAL_RATES} 个噪声率"
echo "  成功:     ${COMPLETED} 个"
echo "  失败:     ${FAILED} 个"
echo ""

if [ ${#COMPLETED_RATES[@]} -gt 0 ]; then
    print_message "$GREEN" "✓ 成功完成: ${COMPLETED_RATES[*]}"
fi

if [ ${#FAILED_RATES[@]} -gt 0 ]; then
    print_message "$RED" "✗ 执行失败: ${FAILED_RATES[*]}"
fi

echo ""
print_message "$YELLOW" "📁 输出目录:"
echo "  分析结果: output/label_correction/analysis/noise_*pct/"
echo "  日志文件: output/logs/noise_*pct_analysis_*.log"
echo ""

if [ $FAILED -gt 0 ]; then
    exit 1
fi
