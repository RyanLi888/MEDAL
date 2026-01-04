#!/bin/bash
# Run backbone evaluation (t-SNE + KNN purity) for an existing Stage1 backbone

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_BASE="output"
if [ -n "${MEDAL_DATASET_NAME:-}" ]; then
  OUTPUT_BASE="output/${MEDAL_DATASET_NAME}"
fi

MODEL_DIR="$OUTPUT_BASE/feature_extraction/models"
DATA_ROOT="$OUTPUT_BASE/preprocessed"
EVAL_ROOT="$OUTPUT_BASE/backbone_eval"

if ! command -v python >/dev/null 2>&1; then
  echo "❌ 未找到 python"
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "❌ 找不到骨干网络目录: $MODEL_DIR"
  echo "请先运行 Stage 1 训练骨干网络"
  exit 1
fi

if [ ! -f "$DATA_ROOT/train_X.npy" ] || [ ! -f "$DATA_ROOT/train_y.npy" ]; then
  echo "❌ 找不到预处理数据: $DATA_ROOT/train_X.npy 或 $DATA_ROOT/train_y.npy"
  echo "请先运行: python scripts/utils/preprocess.py"
  exit 1
fi

BACKBONE_FILES=( $(ls -t "$MODEL_DIR"/*.pth 2>/dev/null || true) )
if [ ${#BACKBONE_FILES[@]} -eq 0 ]; then
  echo "❌ 未找到任何 backbone .pth 文件: $MODEL_DIR"
  exit 1
fi

echo "可用骨干网络模型:"
echo "----------------------------------------"
for i in "${!BACKBONE_FILES[@]}"; do
  echo "  $((i+1))) $(basename "${BACKBONE_FILES[$i]}")"
done
echo "----------------------------------------"
echo ""
echo -n "请选择要评估的模型 (1-${#BACKBONE_FILES[@]}, 默认1): "
read -r choice
choice=${choice:-1}

if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#BACKBONE_FILES[@]} ]; then
  echo "无效选择，使用第一个模型"
  choice=1
fi

BACKBONE_PATH="${BACKBONE_FILES[$((choice-1))]}"
TS=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="$EVAL_ROOT/interactive_${TS}"
mkdir -p "$OUT_DIR"

echo ""
echo "============================================================"
echo "评估骨干网络"
echo "  backbone: $BACKBONE_PATH"
echo "  data_root: $DATA_ROOT"
echo "  output:   $OUT_DIR"
echo "============================================================"
echo ""

DEVICE_ARG="cpu"
if python -c "import torch; print('1' if torch.cuda.is_available() else '0')" >/dev/null 2>&1; then
  HAS_CUDA=$(python -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>/dev/null || echo "0")
  if [ "$HAS_CUDA" = "1" ]; then
    DEVICE_ARG="cuda"
  fi
fi

python scripts/evaluate_backbone.py \
  --backbone "$BACKBONE_PATH" \
  --data_root "$DATA_ROOT" \
  --output "$OUT_DIR" \
  --device "$DEVICE_ARG" 2>&1 | tee "$OUT_DIR/evaluation_console.log"

echo ""
echo "✓ 完成"
echo "  - 输出目录: $OUT_DIR"
