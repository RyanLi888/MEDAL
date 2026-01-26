#!/bin/bash
# ============================================================
# MEDAL-Lite 骨干网络对比学习扫描脚本 (重构版)
# ============================================================
# 说明：
# - 训练 SimMTM + InfoNCE
# - 自动评估骨干网络的特征质量
# - 使用 config.py 中的配置
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
MODEL_DIR="${OUTPUT_DIR}/feature_extraction/models"
EVAL_ROOT="${OUTPUT_DIR}/backbone_eval"
DATA_ROOT="${OUTPUT_DIR}/preprocessed"

mkdir -p "$LOG_DIR" "$MODEL_DIR" "$EVAL_ROOT"

# 时间戳
TS=$(date +"%Y%m%d_%H%M%S")

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ============================================================
# GPU选择
# ============================================================

select_free_gpu() {
    python -c "
import torch
import subprocess
import sys

if not torch.cuda.is_available():
    print('0')
    sys.exit(0)

try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True
    )
    
    gpu_info = []
    for line in result.stdout.strip().split('\n'):
        idx, free = line.strip().split(',')
        gpu_info.append((int(idx.strip()), int(free.strip())))
    
    if not gpu_info:
        print('0')
        sys.exit(0)
    
    # 选择空闲显存最多的GPU
    gpu_info.sort(key=lambda x: -x[1])
    print(gpu_info[0][0])
    
except Exception:
    print('0')
" 2>/dev/null
}

SELECTED_GPU=$(select_free_gpu)
export CUDA_VISIBLE_DEVICES=$SELECTED_GPU
echo -e "${GREEN}✓ 使用 GPU $SELECTED_GPU${NC}"
echo ""

# ============================================================
# 检查环境
# ============================================================

if ! command -v python >/dev/null 2>&1; then
    echo -e "${RED}❌ 未找到 python${NC}"
    exit 1
fi

if ! python -c "import torch" >/dev/null 2>&1; then
    echo -e "${RED}❌ 未找到 torch${NC}"
    exit 1
fi

if [ ! -f "$DATA_ROOT/train_X.npy" ]; then
    echo -e "${RED}❌ 未找到预处理数据: $DATA_ROOT/train_X.npy${NC}"
    echo "请先运行数据预处理"
    exit 1
fi

# ============================================================
# 训练和评估
# ============================================================

echo -e "${BLUE}=========================================="
echo "骨干网络对比学习训练"
echo "=========================================="
echo "方法: InfoNCE"
echo "输出: $MODEL_DIR"
echo "评估: $EVAL_ROOT"
echo -e "==========================================${NC}"
echo ""

method="infonce"

echo -e "${BLUE}=========================================="
echo "训练: SimMTM + InfoNCE"
echo -e "==========================================${NC}"

LOG_FILE="$LOG_DIR/backbone_${method}_${TS}.log"

# 训练
MEDAL_CONTRASTIVE_METHOD="$method" \
    python scripts/training/train.py --start_stage 1 --end_stage 1 2>&1 | tee "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}❌ 训练失败${NC}"
    exit 1
fi

# 保存模型
PRETRAINED="$MODEL_DIR/backbone_pretrained.pth"
SAVED_BACKBONE="$MODEL_DIR/backbone_SimMTM_${method^^}_${TS}.pth"

if [ -f "$PRETRAINED" ]; then
    cp -f "$PRETRAINED" "$SAVED_BACKBONE"
    echo -e "${GREEN}✓ 保存: $SAVED_BACKBONE${NC}"
fi

# 评估
echo -e "${BLUE}=========================================="
echo "评估: InfoNCE"
echo -e "==========================================${NC}"

EVAL_DIR="$EVAL_ROOT/${method}_${TS}"
mkdir -p "$EVAL_DIR"

python scripts/evaluate_backbone.py \
    --backbone "$SAVED_BACKBONE" \
    --data_root "$DATA_ROOT" \
    --output "$EVAL_DIR" 2>&1 | tee "$EVAL_DIR/evaluation.log"

echo -e "${GREEN}✓ 评估完成: $EVAL_DIR${NC}"
echo ""

echo -e "${GREEN}=========================================="
echo "全部完成"
echo "模型: $MODEL_DIR"
echo "评估: $EVAL_ROOT"
echo -e "==========================================${NC}"
