#!/bin/bash
set -euo pipefail

# 临时脚本：依次训练 SimMTM + {InfoNCE, SimSiam, NNCLR} 三个骨干网络，并对每个骨干网络运行评估

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Reuse a stable timestamp across daemon re-exec to avoid log/pid mismatch.
TS="${MEDAL_SWEEP_TS:-$(date +"%Y%m%d_%H%M%S")}" 
LOG_DIR="output/logs"
MODEL_DIR="output/feature_extraction/models"
EVAL_ROOT="output/backbone_eval"
DATA_ROOT="output/preprocessed"
DEVICE="cuda"

mkdir -p "$LOG_DIR" "$MODEL_DIR" "$EVAL_ROOT"

# -------------------- 动态选择空闲GPU --------------------
select_free_gpu() {
    python -c "
import torch
import subprocess
import sys

if not torch.cuda.is_available():
    print('CPU', file=sys.stderr)
    sys.exit(1)

try:
    # 获取显存使用情况
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.memory', 
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True
    )
    
    gpu_info = []
    for line in result.stdout.strip().split('\n'):
        parts = line.strip().split(',')
        if len(parts) >= 4:
            idx = int(parts[0].strip())
            used = int(parts[1].strip())
            total = int(parts[2].strip())
            util = int(parts[3].strip())
            free = total - used
            gpu_info.append((idx, free, util, used, total))
    
    if not gpu_info:
        print('0')
        sys.exit(0)
    
    # 按空闲显存排序（降序），然后按占用率排序（升序）
    gpu_info.sort(key=lambda x: (-x[1], x[2]))
    
    best_gpu = gpu_info[0]
    print(f'{best_gpu[0]}', file=sys.stderr)
    print(f'选择GPU {best_gpu[0]}: 空闲 {best_gpu[1]/1024:.2f}GB / 总量 {best_gpu[4]/1024:.2f}GB (占用率: {best_gpu[2]}%)', file=sys.stderr)
    print(best_gpu[0])
    
except Exception as e:
    print(f'GPU检测失败: {e}', file=sys.stderr)
    print('0')
" 2>&1
}

# 选择最空闲的GPU（在后台进程中再次确认）
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  SELECTED_GPU=$(select_free_gpu | tail -n 1)
  if [ -n "$SELECTED_GPU" ] && [ "$SELECTED_GPU" != "CPU" ]; then
      export CUDA_VISIBLE_DEVICES=$SELECTED_GPU
      echo "✓ 自动选择GPU: $SELECTED_GPU"
  else
      echo "⚠ 未检测到GPU或选择失败，将使用默认设置"
  fi
else
  echo "✓ 使用预设GPU: $CUDA_VISIBLE_DEVICES"
fi
echo ""

# -------------------- Background mode --------------------
# Default: when you run this script in foreground, it will re-exec itself via nohup
# and continue in the background, writing a master log.
# Disable by setting: MEDAL_SWEEP_NO_DAEMON=1

MASTER_LOG="$LOG_DIR/backbone_sweep_${TS}.log"
PID_FILE="$LOG_DIR/backbone_sweep_${TS}.pid"

if [ "${MEDAL_SWEEP_NO_DAEMON:-0}" != "1" ] && [ -z "${MEDAL_SWEEP_DAEMONIZED:-}" ]; then
  # 在后台启动前先选择GPU
  SELECTED_GPU_TEMP=$(select_free_gpu | tail -n 1)
  
  echo "[daemon] Launching in background..."
  echo "[daemon] master log: $MASTER_LOG"
  echo "[daemon] pid file:   $PID_FILE"
  
  if [ -n "$SELECTED_GPU_TEMP" ] && [ "$SELECTED_GPU_TEMP" != "CPU" ]; then
    echo "[daemon] Selected GPU: $SELECTED_GPU_TEMP"
    nohup env MEDAL_SWEEP_DAEMONIZED=1 MEDAL_SWEEP_TS="$TS" CUDA_VISIBLE_DEVICES=$SELECTED_GPU_TEMP \
      bash "$SCRIPT_PATH" > "$MASTER_LOG" 2>&1 &
  else
    echo "[daemon] No GPU selected, using default"
    nohup env MEDAL_SWEEP_DAEMONIZED=1 MEDAL_SWEEP_TS="$TS" \
      bash "$SCRIPT_PATH" > "$MASTER_LOG" 2>&1 &
  fi
  
  echo $! > "$PID_FILE"
  echo "[daemon] PID: $(cat "$PID_FILE")"
  exit 0
fi

# Basic environment checks
if ! command -v python >/dev/null 2>&1; then
  echo "❌ 未找到 python，请先激活你的环境（如 conda/mamba/venv）"
  exit 1
fi

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "❌ 当前 python 环境缺少 torch（或未激活正确环境）"
  echo "  - 请先激活包含 torch 的环境后再运行该脚本"
  echo "  - 例如：conda activate RAPIER"
  exit 1
fi

if [ ! -f "$DATA_ROOT/train_X.npy" ] || [ ! -f "$DATA_ROOT/train_y.npy" ]; then
  echo "❌ 找不到预处理数据: $DATA_ROOT/train_X.npy 或 $DATA_ROOT/train_y.npy"
  echo "请先运行 preprocess.py 生成 output/preprocessed/ 下的 train_X/train_y"
  exit 1
fi

declare -a METHODS=("infonce" "simsiam" "nnclr")

TOTAL=${#METHODS[@]}
IDX=0

for method in "${METHODS[@]}"; do
  IDX=$((IDX+1))
  
  # 在每个训练任务前重新选择最空闲的GPU
  echo "============================================================"
  echo "[$IDX/$TOTAL] 检查GPU状态..."
  echo "============================================================"
  
  TASK_GPU=$(select_free_gpu | tail -n 1)
  if [ -n "$TASK_GPU" ] && [ "$TASK_GPU" != "CPU" ]; then
    export CUDA_VISIBLE_DEVICES=$TASK_GPU
    echo "✓ 为本次训练选择GPU: $TASK_GPU"
  else
    echo "⚠ GPU选择失败，使用当前设置: ${CUDA_VISIBLE_DEVICES:-default}"
  fi
  echo ""
  
  echo "============================================================"
  echo "[$IDX/$TOTAL] 训练 Stage1: SimMTM + ${method^^}"
  echo "============================================================"

  LOG_FILE="$LOG_DIR/backbone_${method}_${TS}.log"

  # Point live.log to the current method log so users can always `tail -f output/logs/live.log`.
  ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/live.log"

  # 训练（前台运行；如需后台可自行用 nohup 包裹）
  # 如果OOM，尝试清理缓存后重试一次
  RETRY_COUNT=0
  MAX_RETRIES=1
  
  # 添加心跳日志
  echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始训练 ${method^^}" | tee -a "$LOG_FILE"
  
  while [ $RETRY_COUNT -le $MAX_RETRIES ]; do
    # 使用trap捕获信号
    trap 'echo "$(date) - 训练被中断" | tee -a "$LOG_FILE"; exit 130' INT TERM
    
    MEDAL_CONTRASTIVE_METHOD="$method" \
      PYTHONUNBUFFERED=1 python -u scripts/training/train.py --start_stage 1 --end_stage 1 2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 训练结束，退出码: $EXIT_CODE" | tee -a "$LOG_FILE"
    
    if [ $EXIT_CODE -eq 0 ]; then
      echo "✓ 训练成功完成"
      break
    elif [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 143 ]; then
      echo "⚠ 训练被用户中断或系统信号终止"
      exit $EXIT_CODE
    else
      # 检查是否是OOM错误
      if grep -q "CUDA out of memory" "$LOG_FILE" && [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        RETRY_COUNT=$((RETRY_COUNT+1))
        echo "❌ 检测到显存不足错误，尝试清理并重试 ($RETRY_COUNT/$MAX_RETRIES)..."
        
        # 清理GPU缓存
        python -c "import torch; torch.cuda.empty_cache(); print('GPU缓存已清理')" 2>/dev/null || true
        sleep 5
        
        # 重新选择GPU
        TASK_GPU=$(select_free_gpu | tail -n 1)
        if [ -n "$TASK_GPU" ] && [ "$TASK_GPU" != "CPU" ]; then
          export CUDA_VISIBLE_DEVICES=$TASK_GPU
          echo "✓ 重新选择GPU: $TASK_GPU"
        fi
        
        echo "重新开始训练..."
        continue
      else
        echo "❌ 训练失败 (退出码: $EXIT_CODE)"
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
          echo "已达到最大重试次数，跳过此方法"
        fi
        break
      fi
    fi
  done
  
  if [ $EXIT_CODE -ne 0 ]; then
    echo "⚠ 跳过 ${method^^} 的评估"
    continue
  fi

  # 找到本轮输出的 backbone 文件（train.py 会写入 backbone_pretrained.pth + backbone_SimMTM_<METHOD>_<n>.pth）
  PRETRAINED="$MODEL_DIR/backbone_pretrained.pth"

  if [ ! -f "$PRETRAINED" ]; then
    echo "❌ 未找到 $PRETRAINED"
    exit 1
  fi

  # 优先取最新的 backbone_SimMTM_<METHOD>_*.pth
  CANDIDATE=$(ls -t "$MODEL_DIR"/backbone_SimMTM_${method^^}_*.pth 2>/dev/null | head -n 1 || true)
  if [ -z "$CANDIDATE" ]; then
    echo "⚠ 未找到 backbone_SimMTM_${method^^}_*.pth，将退回使用 backbone_pretrained.pth"
    CANDIDATE="$PRETRAINED"
  fi

  # 为了避免被下一轮覆盖，复制一份带时间戳的固定文件名
  SAVED_BACKBONE="$MODEL_DIR/backbone_SimMTM_${method^^}_${TS}.pth"
  cp -f "$CANDIDATE" "$SAVED_BACKBONE"

  echo "✓ 保存骨干网络: $SAVED_BACKBONE"

  echo "============================================================"
  echo "[$IDX/$TOTAL] 评估骨干网络: ${method^^}"
  echo "============================================================"

  EVAL_DIR="$EVAL_ROOT/${method}_${TS}"
  mkdir -p "$EVAL_DIR"

  python evaluate_backbone.py \
    --backbone "$SAVED_BACKBONE" \
    --data_root "$DATA_ROOT" \
    --output "$EVAL_DIR" \
    --device "$DEVICE" 2>&1 | tee "$EVAL_DIR/evaluation_console.log"

  echo "✓ 评估输出目录: $EVAL_DIR"
  echo "  - tsne: $EVAL_DIR/tsne_visualization.png"
  echo "  - log:  $EVAL_DIR/evaluation_report.log"
  echo "  - json: $EVAL_DIR/evaluation_report.json"

done

echo "============================================================"
echo "全部完成"
echo "骨干模型目录: $MODEL_DIR"
echo "评估结果目录: $EVAL_ROOT"
echo "============================================================"
