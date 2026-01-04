#!/bin/bash
# Run Label Correction Analysis Script
# 支持单个噪声率或批量运行多个噪声率

# 获取脚本所在目录（scripts/experiments/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 切换到项目根目录（python/MEDAL）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_BASE="$PROJECT_ROOT/output"
if [ -n "${MEDAL_DATASET_NAME:-}" ]; then
    OUTPUT_BASE="$PROJECT_ROOT/output/${MEDAL_DATASET_NAME}"
fi

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
PYTHON_SCRIPT="$PROJECT_ROOT/MoudleCode/label_correction/analysis/label_correction_analysis.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_message "$RED" "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check for existing backbone models
echo "检查已有的骨干网络模型..."
BACKBONE_DIR="$OUTPUT_BASE/feature_extraction/models"
USE_EXISTING_BACKBONE="false"
BACKBONE_PATH=""

if [ -d "$BACKBONE_DIR" ]; then
    # 查找所有backbone文件
    BACKBONE_FILES=($(ls -t "$BACKBONE_DIR"/backbone_*.pth 2>/dev/null))
    
    if [ ${#BACKBONE_FILES[@]} -gt 0 ]; then
        print_message "$GREEN" "✓ 发现 ${#BACKBONE_FILES[@]} 个已训练的骨干网络"
        echo ""
        echo "是否使用已有的骨干网络? (y/n, 默认y)"
        echo "  - 选择 y: 使用已有backbone进行特征提取"
        echo "  - 选择 n: 使用随机初始化的backbone（不推荐）"
        echo ""
        echo -n "请输入选择: "
        read -r use_existing_backbone
        use_existing_backbone=${use_existing_backbone:-y}
        
        if [ "$use_existing_backbone" = "y" ] || [ "$use_existing_backbone" = "Y" ]; then
            echo ""
            echo "可用的骨干网络模型:"
            echo "----------------------------------------"
            for i in "${!BACKBONE_FILES[@]}"; do
                filename=$(basename "${BACKBONE_FILES[$i]}")
                filesize=$(du -h "${BACKBONE_FILES[$i]}" | cut -f1)
                filetime=$(stat -c %y "${BACKBONE_FILES[$i]}" 2>/dev/null | cut -d'.' -f1 || stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "${BACKBONE_FILES[$i]}")
                echo "  $((i+1))) $filename"
                echo "      大小: $filesize | 时间: $filetime"
            done
            echo "----------------------------------------"
            echo ""
            echo -n "请选择要使用的模型 (1-${#BACKBONE_FILES[@]}, 默认1): "
            read -r backbone_choice
            backbone_choice=${backbone_choice:-1}
            
            # 验证输入
            if ! [[ "$backbone_choice" =~ ^[0-9]+$ ]] || [ "$backbone_choice" -lt 1 ] || [ "$backbone_choice" -gt ${#BACKBONE_FILES[@]} ]; then
                echo "无效选择，使用第一个模型"
                backbone_choice=1
            fi
            
            BACKBONE_PATH="${BACKBONE_FILES[$((backbone_choice-1))]}"
            SELECTED_BACKBONE_NAME=$(basename "$BACKBONE_PATH")
            USE_EXISTING_BACKBONE="true"
            
            print_message "$GREEN" "✓ 已选择: $SELECTED_BACKBONE_NAME"
            echo ""
        else
            print_message "$YELLOW" "⚠ 将使用随机初始化的backbone（特征质量可能较差）"
            RETRAIN_BACKBONE="--retrain_backbone"
            echo ""
        fi
    else
        print_message "$YELLOW" "⚠ 未找到已训练的骨干网络"
        echo "将使用随机初始化的backbone（特征质量可能较差）"
        echo "建议先运行: bash scripts/run_experiment.sh 选择模式1训练骨干网络"
        RETRAIN_BACKBONE="--retrain_backbone"
        echo ""
    fi
else
    print_message "$YELLOW" "⚠ 骨干网络目录不存在"
    echo "将使用随机初始化的backbone（特征质量可能较差）"
    RETRAIN_BACKBONE="--retrain_backbone"
    echo ""
fi

# Check if virtual environment exists and activate it
# If user already activated a conda env (e.g. MEDAL), do not switch env implicitly.
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    print_message "$GREEN" "检测到当前已激活 conda 环境: ${CONDA_DEFAULT_ENV}"
    print_message "$GREEN" "将继续使用当前环境，不再自动切换到 RAPIER/medal"
    echo ""
else
    if [ -d "$HOME/anaconda3/envs/RAPIER" ]; then
        print_message "$GREEN" "激活conda环境: RAPIER"
        source "$HOME/anaconda3/bin/activate" RAPIER
        echo ""
    elif [ -d "$HOME/anaconda3/envs/medal" ]; then
        print_message "$GREEN" "激活conda环境: medal"
        source "$HOME/anaconda3/bin/activate" medal
        echo ""
    fi
fi

# Convert noise rates to array
read -ra RATES_ARRAY <<< "$NOISE_RATES"

# Print configuration
print_message "$YELLOW" "配置信息:"
echo "  噪声率:            ${RATES_ARRAY[*]}"
echo "  批量模式:          $([ "$BATCH_MODE" = true ] && echo "是" || echo "否")"
if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
    echo "  骨干网络:          $SELECTED_BACKBONE_NAME"
else
    echo "  骨干网络:          随机初始化"
fi
echo "  脚本路径:          $PYTHON_SCRIPT"
echo "  工作目录:          $PROJECT_ROOT"
echo ""

# Confirm execution
read -p "$(echo -e ${YELLOW}是否继续? [Y/n]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    print_message "$RED" "用户取消"
    exit 0
fi

# Create log directory
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"

# Track results
TOTAL_RATES=${#RATES_ARRAY[@]}
COMPLETED=0
FAILED=0
declare -a COMPLETED_RATES
declare -a FAILED_RATES

# Run analysis for each noise rate
# 确保在项目根目录运行
cd "$PROJECT_ROOT"

for NOISE_RATE in "${RATES_ARRAY[@]}"; do
    NOISE_PCT=$(printf "%.0f" $(echo "$NOISE_RATE * 100" | bc))
    
    echo ""
    print_message "$CYAN" "======================================================================"
    print_message "$CYAN" "  开始分析噪声率: ${NOISE_PCT}%  (${COMPLETED}/${TOTAL_RATES} 已完成)"
    print_message "$CYAN" "======================================================================"
    echo ""
    
    # 构建命令参数
    CMD_ARGS="--noise_rate $NOISE_RATE"
    
    # 添加骨干网络参数
    if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
        CMD_ARGS="$CMD_ARGS --backbone_path $BACKBONE_PATH"
    elif [ -n "$RETRAIN_BACKBONE" ]; then
        CMD_ARGS="$CMD_ARGS $RETRAIN_BACKBONE"
    fi
    
    # Run Python script
    python "$PYTHON_SCRIPT" $CMD_ARGS
    
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
if [ -n "${MEDAL_DATASET_NAME:-}" ]; then
  echo "  分析结果: output/${MEDAL_DATASET_NAME}/label_correction/analysis/noise_*pct/"
  echo "  日志文件: output/${MEDAL_DATASET_NAME}/logs/noise_*pct_analysis_*.log"
else
  echo "  分析结果: output/label_correction/analysis/noise_*pct/"
  echo "  日志文件: output/logs/noise_*pct_analysis_*.log"
fi
echo ""

if [ $FAILED -gt 0 ]; then
    exit 1
fi
