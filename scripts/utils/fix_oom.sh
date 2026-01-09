#!/bin/bash
# 快速修复 OOM 问题
# 使用方法: bash scripts/utils/fix_oom.sh

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "修复 CUDA OOM 问题"
echo "=========================================="
echo ""

# 1. 检查当前批次大小
current_batch=$(grep "PRETRAIN_BATCH_SIZE = " MoudleCode/utils/config.py | head -1)
echo "当前配置: $current_batch"

# 2. 清理 GPU 缓存
echo ""
echo "清理 GPU 缓存..."
python -c "import torch; torch.cuda.empty_cache(); print('✓ GPU 缓存已清理')" 2>/dev/null || echo "⚠️  torch 未安装，跳过 GPU 清理"

# 3. 显示修复建议
echo ""
echo "=========================================="
echo "修复方案"
echo "=========================================="
echo ""
echo "✓ 批次大小已降低到 32 (在 config.py 中)"
echo ""
echo "建议的运行命令:"
echo ""
echo "  # 方案 1: 使用内存优化 (推荐)"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
echo "    python scripts/training/train.py --start_stage 1 --end_stage 1"
echo ""
echo "  # 方案 2: 使用混合精度训练 (更快)"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
echo "    MEDAL_USE_AMP=1 \\"
echo "    python scripts/training/train.py --start_stage 1 --end_stage 1"
echo ""
echo "  # 方案 3: 进一步降低批次大小 (如果仍然 OOM)"
echo "  MEDAL_PRETRAIN_BATCH_SIZE=16 \\"
echo "    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
echo "    python scripts/training/train.py --start_stage 1 --end_stage 1"
echo ""
echo "=========================================="
echo "内存监控"
echo "=========================================="
echo ""
echo "在另一个终端运行以下命令监控 GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
