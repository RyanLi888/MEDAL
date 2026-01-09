#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ACTION="${1:-run_bg}"  # run_bg | collect | run_fg
GPU_ID="${GPU_ID:-7}"
BACKBONE_PATH="${BACKBONE_PATH:-$PROJECT_ROOT/output/feature_extraction/models/backbone_SimMTM_INFONCE_32d_500.pth}"
SEEDS_STR="${SEEDS:-42 43 44}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/output/opt/clean_only_${TS}}"
META_TSV="$RUN_DIR/runs.tsv"
SUMMARY_CSV="$RUN_DIR/summary.csv"

mkdir -p "$RUN_DIR"

resolve_python() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
      echo "$PYTHON_BIN"
      return 0
    fi
  fi

  if [ -x "$HOME/anaconda3/bin/python" ]; then
    if "$HOME/anaconda3/bin/python" -c "import torch" >/dev/null 2>&1; then
      echo "$HOME/anaconda3/bin/python"
      return 0
    fi
  fi

  echo "$PYTHON_BIN"
}

PYTHON_BIN_RESOLVED="$(resolve_python)"

collect_one() {
  local name="$1"
  local log="$2"

  local threshold accuracy precision recall f1 f1_macro auc
  threshold=$(grep -E "测试集最优阈值:" -m 1 "$log" | sed -E 's/.*测试集最优阈值: ([0-9.]+).*/\1/' || true)
  accuracy=$(grep -E "^Accuracy:" -m 1 "$log" | awk '{print $2}' || true)
  precision=$(grep -E "^Precision \(pos=1\):" -m 1 "$log" | awk '{print $3}' || true)
  recall=$(grep -E "^Recall    \(pos=1\):" -m 1 "$log" | awk '{print $3}' || true)
  f1=$(grep -E "^F1 \(pos=1\):" -m 1 "$log" | awk '{print $3}' || true)
  f1_macro=$(grep -E "^F1-Macro:" -m 1 "$log" | awk '{print $2}' || true)
  auc=$(grep -E "^AUC:" -m 1 "$log" | awk '{print $2}' || true)

  echo "${name},${accuracy},${precision},${recall},${f1},${f1_macro},${auc},${threshold},${log}" >> "$SUMMARY_CSV"
}

run_one() {
  local exp_name="$1"
  local seed="$2"
  shift 2

  local name="${exp_name}_seed${seed}"
  local log="$RUN_DIR/${name}.log"

  echo "  🚀 启动: $name"
  echo "     日志: $log"

  (
    cd "$PROJECT_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN_RESOLVED" scripts/training/train_clean_only_then_test.py \
      --use_ground_truth \
      --backbone_path "$BACKBONE_PATH" \
      --seed "$seed" \
      --run_tag "$name" \
      "$@" \
      > "$log" 2>&1
  )

  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "${name}	OK	${log}" >> "$META_TSV"
  else
    echo "${name}	FAIL(${exit_code})	${log}" >> "$META_TSV"
  fi
}

run_suite_sync() {
  : > "$META_TSV"

  echo "run_dir: $RUN_DIR"
  echo "gpu_id: $GPU_ID"
  echo "seeds: $SEEDS_STR"
  echo "python: $PYTHON_BIN_RESOLVED"
  echo "backbone: $BACKBONE_PATH"

  # A small, step-by-step sweep. You can extend this matrix later.
  for seed in $SEEDS_STR; do
    run_one "baseline" "$seed" \
      --focal_alpha 0.5 --focal_gamma 2.0 \
      --finetune_scope projection --finetune_lr 1e-5 --finetune_warmup_epochs 30

    run_one "alpha065" "$seed" \
      --focal_alpha 0.65 --focal_gamma 2.0 \
      --finetune_scope projection --finetune_lr 1e-5 --finetune_warmup_epochs 30

    run_one "gamma15" "$seed" \
      --focal_alpha 0.5 --focal_gamma 1.5 \
      --finetune_scope projection --finetune_lr 1e-5 --finetune_warmup_epochs 30

    run_one "alpha065_gamma15" "$seed" \
      --focal_alpha 0.65 --focal_gamma 1.5 \
      --finetune_scope projection --finetune_lr 1e-5 --finetune_warmup_epochs 30

    run_one "warmup50" "$seed" \
      --focal_alpha 0.5 --focal_gamma 2.0 \
      --finetune_scope projection --finetune_lr 1e-5 --finetune_warmup_epochs 50

    run_one "projlr2e5" "$seed" \
      --focal_alpha 0.5 --focal_gamma 2.0 \
      --finetune_scope projection --finetune_lr 2e-5 --finetune_warmup_epochs 30

    run_one "epochs300" "$seed" \
      --focal_alpha 0.5 --focal_gamma 2.0 \
      --finetune_scope projection --finetune_lr 1e-5 --finetune_warmup_epochs 30 \
      --finetune_epochs 300
  done
}

collect() {
  if [ ! -f "$META_TSV" ]; then
    echo "meta not found: $META_TSV" >&2
    exit 1
  fi

  echo "name,accuracy,precision,recall,f1,f1_macro,auc,threshold,log" > "$SUMMARY_CSV"
  tail -n +2 "$META_TSV" | while IFS=$'\t' read -r name status log; do
    if [ -f "$log" ]; then
      collect_one "$name" "$log"
    else
      echo "${name},,,,,,,,${log}" >> "$SUMMARY_CSV"
    fi
  done

  echo "summary: $SUMMARY_CSV"
}

case "$ACTION" in
  run_fg)
    run_suite_sync
    ;;
  run_bg)
    (
      run_suite_sync
    ) > "$RUN_DIR/driver.log" 2>&1 &
    echo "started"
    echo "run_dir: $RUN_DIR"
    echo "pid: $!"
    echo "driver_log: $RUN_DIR/driver.log"
    ;;
  collect)
    collect
    ;;
  *)
    echo "Usage: $0 {run_bg|run_fg|collect}" >&2
    exit 2
    ;;
esac
