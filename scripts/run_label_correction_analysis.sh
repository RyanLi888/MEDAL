#!/bin/bash
# ============================================================
# MEDAL-Lite 标签矫正分析脚本 (重构版)
# ============================================================
# 说明：
# - 支持单个或批量噪声率分析
# - 使用 config.py 中的配置
# - 自动选择已有backbone或训练新的
# ============================================================

set -euo pipefail

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/output"
if [ -n "${MEDAL_DATASET_NAME:-}" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/output/${MEDAL_DATASET_NAME}"
fi

LOG_DIR="${OUTPUT_DIR}/logs"
BACKBONE_DIR="${OUTPUT_DIR}/feature_extraction/models"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 默认值
NOISE_RATES=""
USE_EXISTING_BACKBONE="false"
BACKBONE_PATH=""

# ============================================================
# 帮助信息
# ============================================================

print_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -n, --noise-rate RATE    单个噪声率 (例如: 0.30)"
    echo "  -b, --batch RATES        批量噪声率 (例如: \"0.1 0.2 0.3\")"
    echo "  -a, --all                运行所有预设噪声率 (10%, 20%, 30%, 40%)"
    echo "  -r, --retrain-backbone   使用随机初始化的backbone"
    echo "  -h, --help               显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 -n 0.30               # 运行30%噪声率"
    echo "  $0 -b \"0.1 0.2 0.3\"      # 批量运行"
    echo "  $0 -a                    # 运行所有预设噪声率"
    echo ""
}

# ============================================================
# 参数解析
# ============================================================

RETRAIN_BACKBONE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--noise-rate)
            NOISE_RATES="$2"
            shift 2
            ;;
        -b|--batch)
            NOISE_RATES="$2"
            shift 2
            ;;
        -a|--all)
            NOISE_RATES="0.1 0.2 0.3 0.4"
            shift
            ;;
        -r|--retrain-backbone)
            RETRAIN_BACKBONE="--retrain_backbone"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# 默认30%
if [ -z "$NOISE_RATES" ]; then
    NOISE_RATES="0.3"
fi

# ============================================================
# Banner
# ============================================================

echo -e "${BLUE}=========================================="
echo "MEDAL-Lite 标签矫正分析"
echo -e "==========================================${NC}"
echo ""

# ============================================================
# 检查backbone
# ============================================================

if [ -z "$RETRAIN_BACKBONE" ] && [ -d "$BACKBONE_DIR" ]; then
    BACKBONE_FILES=($(ls -t "$BACKBONE_DIR"/backbone_*.pth 2>/dev/null))
    
    if [ ${#BACKBONE_FILES[@]} -gt 0 ]; then
        echo -e "${GREEN}✓ 发现 ${#BACKBONE_FILES[@]} 个已训练的骨干网络${NC}"
        echo -n "是否使用已有backbone? (y/n, 默认y): "
        read -r use_existing
        use_existing=${use_existing:-y}
        
        if [ "$use_existing" = "y" ]; then
            echo ""
            echo "可用的骨干网络:"
            for i in "${!BACKBONE_FILES[@]}"; do
                echo "  $((i+1))) $(basename "${BACKBONE_FILES[$i]}")"
            done
            
            echo -n "选择模型 (1-${#BACKBONE_FILES[@]}, 默认1): "
            read -r choice
            choice=${choice:-1}
            
            BACKBONE_PATH="${BACKBONE_FILES[$((choice-1))]}"
            USE_EXISTING_BACKBONE="true"
            echo -e "${GREEN}✓ 已选择: $(basename "$BACKBONE_PATH")${NC}"
        else
            RETRAIN_BACKBONE="--retrain_backbone"
        fi
    else
        echo -e "${YELLOW}⚠ 未找到已训练的骨干网络${NC}"
        RETRAIN_BACKBONE="--retrain_backbone"
    fi
else
    echo -e "${YELLOW}⚠ 将使用随机初始化的backbone${NC}"
fi

echo ""

# ============================================================
# 运行分析
# ============================================================

PYTHON_SCRIPT="$PROJECT_ROOT/MoudleCode/label_correction/analysis/label_correction_analysis.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}❌ 未找到脚本: $PYTHON_SCRIPT${NC}"
    exit 1
fi

read -ra RATES_ARRAY <<< "$NOISE_RATES"
TOTAL=${#RATES_ARRAY[@]}
COMPLETED=0
FAILED=0

echo -e "${YELLOW}配置:${NC}"
echo "  噪声率: ${RATES_ARRAY[*]}"
if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
    echo "  骨干网络: $(basename "$BACKBONE_PATH")"
else
    echo "  骨干网络: 随机初始化"
fi
echo ""

for NOISE_RATE in "${RATES_ARRAY[@]}"; do
    NOISE_PCT=$(printf "%.0f" $(echo "$NOISE_RATE * 100" | bc))
    
    echo -e "${BLUE}=========================================="
    echo "分析噪声率: ${NOISE_PCT}% (${COMPLETED}/${TOTAL})"
    echo -e "==========================================${NC}"
    
    CMD_ARGS="--noise_rate $NOISE_RATE"
    
    if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
        CMD_ARGS="$CMD_ARGS --backbone_path $BACKBONE_PATH"
    elif [ -n "$RETRAIN_BACKBONE" ]; then
        CMD_ARGS="$CMD_ARGS $RETRAIN_BACKBONE"
    fi
    
    python "$PYTHON_SCRIPT" $CMD_ARGS
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 噪声率 ${NOISE_PCT}% 完成${NC}"
        COMPLETED=$((COMPLETED + 1))
    else
        echo -e "${RED}✗ 噪声率 ${NOISE_PCT}% 失败${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# ============================================================
# 摘要
# ============================================================

echo -e "${BLUE}=========================================="
echo "批量分析完成"
echo -e "==========================================${NC}"
echo "  总计: ${TOTAL}"
echo "  成功: ${COMPLETED}"
echo "  失败: ${FAILED}"
echo ""
echo -e "${YELLOW}输出目录: ${OUTPUT_DIR}/label_correction/analysis/${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    exit 1
fi
