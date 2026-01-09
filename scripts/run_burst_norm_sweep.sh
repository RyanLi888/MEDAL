#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/output/sweeps/burst_norm_${TS}}"
PREPROCESSED_SRC="${PREPROCESSED_SRC:-$PROJECT_ROOT/output/preprocessed}"

mkdir -p "$RUN_DIR"

echo "run_dir: $RUN_DIR" | tee "$RUN_DIR/driver.log"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

resolve_python() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "$PYTHON_BIN"
  else
    echo "python"
  fi
}

PYTHON_BIN_RESOLVED="$(resolve_python)"

link_preprocessed() {
  local output_root="$1"
  if [ -d "$PREPROCESSED_SRC" ]; then
    mkdir -p "$output_root"
    if [ ! -e "$output_root/preprocessed" ]; then
      ln -s "$PREPROCESSED_SRC" "$output_root/preprocessed" 2>/dev/null || true
    fi
  fi
}

run_one() {
  local name="$1"
  local burst_norm="$2"
  local dataset_name="$3"
  local burst_weight="$4"

  local log="$RUN_DIR/${name}.log"
  echo "[$(timestamp)] START ${name} seed=${SEED} burst_norm=${burst_norm}" | tee -a "$RUN_DIR/driver.log"
  echo "  log: $log" | tee -a "$RUN_DIR/driver.log"

  (
    cd "$PROJECT_ROOT"

    export MEDAL_DATASET_NAME="$dataset_name"
    export MEDAL_SEED="$SEED"
    export MEDAL_BURSTSIZE_NORMALIZE="$burst_norm"
    if [ -n "$burst_weight" ]; then
      export MEDAL_PRETRAIN_BURST_WEIGHT="$burst_weight"
    fi

    exp_output_root="$PROJECT_ROOT/output/$dataset_name"
    link_preprocessed "$exp_output_root"

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN_RESOLVED" scripts/training/train.py \
      --noise_rate 0.0 --start_stage 1 --end_stage 1 \
      >> "$log" 2>&1

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN_RESOLVED" scripts/training/train_clean_only_then_test.py \
      --use_ground_truth \
      --seed "$SEED" \
      --run_tag "$name" \
      >> "$log" 2>&1
  )

  echo "[$(timestamp)] DONE ${name}" | tee -a "$RUN_DIR/driver.log"
}

collect_one() {
  local name="$1"
  local log="$2"

  local acc="" prec="" rec="" f1="" f1m="" auc="" thr=""

  acc=$(grep -E "^Accuracy:" -m 1 "$log" | awk '{print $2}' || true)
  prec=$(grep -E "^Precision \(pos=1\):" -m 1 "$log" | awk '{print $3}' || true)
  rec=$(grep -E "^Recall" -m 1 "$log" | awk '{print $NF}' || true)
  f1=$(grep -E "^F1 \(pos=1\):" -m 1 "$log" | awk '{print $3}' || true)
  f1m=$(grep -E "^F1-Macro:" -m 1 "$log" | awk '{print $2}' || true)
  auc=$(grep -E "^AUC:" -m 1 "$log" | awk '{print $2}' || true)
  thr=$(grep -E "best_threshold" -m 1 "$log" | sed -E 's/.*best_threshold[:= ]+([0-9.]+).*/\1/' || true)

  echo "${name},${acc},${prec},${rec},${f1},${f1m},${auc},${thr},${log}" >> "$SUMMARY_CSV"
}

SUMMARY_CSV="$RUN_DIR/summary.csv"

echo "name,accuracy,precision,recall,f1,f1_macro,auc,threshold,log" > "$SUMMARY_CSV"

run_one "burst_nonorm" "0" "sweep_burst_nonorm" ""
run_one "burst_norm" "1" "sweep_burst_norm" ""
run_one "burst_norm_w14" "1" "sweep_burst_norm_w14" "14.2"

collect_one "burst_nonorm" "$RUN_DIR/burst_nonorm.log"
collect_one "burst_norm" "$RUN_DIR/burst_norm.log"
collect_one "burst_norm_w14" "$RUN_DIR/burst_norm_w14.log"

echo "summary: $SUMMARY_CSV" | tee -a "$RUN_DIR/driver.log"
