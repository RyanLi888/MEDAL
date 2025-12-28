#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  hard_case_analysis.sh <predictions_npz> [out_dir] [low] [high]

Inputs:
  predictions_npz: path to *_predictions.npz saved by scripts/testing/test.py
                  expected keys: y_true (N,), y_prob (N,2), features (N,D) optional

Args:
  out_dir: output directory (default: ./output/hard_case_report)
  low:     lower prob bound for hard region (default: 0.35)
  high:    upper prob bound for hard region (default: 0.45)

Outputs (in out_dir):
  - hard_samples.csv
  - feature_diff.csv
  - report.md
USAGE
  exit 0
fi

NPZ_PATH="$1"
OUT_DIR="${2:-./output/hard_case_report}"
LOW="${3:-0.35}"
HIGH="${4:-0.45}"

if [[ ! -f "$NPZ_PATH" ]]; then
  echo "[ERR] predictions_npz not found: $NPZ_PATH" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

python3 - "$NPZ_PATH" "$OUT_DIR" "$LOW" "$HIGH" <<'PY'
import os
import json
import numpy as np
import sys

npz_path = sys.argv[1]
out_dir = sys.argv[2]
low = float(sys.argv[3])
high = float(sys.argv[4])

data = np.load(npz_path, allow_pickle=True)

y_true = data['y_true'].astype(int)
y_prob = data['y_prob']

if y_prob.ndim == 2 and y_prob.shape[1] == 2:
    p = y_prob[:, 1].astype(float)
else:
    p = np.asarray(y_prob, dtype=float).reshape(-1)

features = data['features'] if 'features' in data.files else None

hard = (p >= low) & (p <= high)
fn_like = hard & (y_true == 1)
fp_like = hard & (y_true == 0)

rows = []
for idx in np.where(hard)[0]:
    yt = int(y_true[idx])
    prob = float(p[idx])
    if yt == 1:
        group = 'hard_pos'
    else:
        group = 'hard_neg'
    rows.append((idx, yt, prob, group))

hard_csv = os.path.join(out_dir, 'hard_samples.csv')
with open(hard_csv, 'w', encoding='utf-8') as f:
    f.write('index,y_true,prob,group\n')
    for idx, yt, prob, group in rows:
        f.write(f"{idx},{yt},{prob:.6f},{group}\n")

report_path = os.path.join(out_dir, 'report.md')
feat_csv = os.path.join(out_dir, 'feature_diff.csv')

n = int(len(y_true))
nh = int(hard.sum())
nh_pos = int(fn_like.sum())
nh_neg = int(fp_like.sum())

p_stats = {
    'all': {
        'mean': float(np.mean(p)),
        'std': float(np.std(p)),
        'p50': float(np.percentile(p, 50)),
        'p90': float(np.percentile(p, 90)),
        'p99': float(np.percentile(p, 99)),
    },
    'hard': {
        'mean': float(np.mean(p[hard])) if nh else float('nan'),
        'std': float(np.std(p[hard])) if nh else float('nan'),
        'p50': float(np.percentile(p[hard], 50)) if nh else float('nan'),
    },
    'hard_pos': {
        'mean': float(np.mean(p[fn_like])) if nh_pos else float('nan'),
        'std': float(np.std(p[fn_like])) if nh_pos else float('nan'),
        'p50': float(np.percentile(p[fn_like], 50)) if nh_pos else float('nan'),
    },
    'hard_neg': {
        'mean': float(np.mean(p[fp_like])) if nh_neg else float('nan'),
        'std': float(np.std(p[fp_like])) if nh_neg else float('nan'),
        'p50': float(np.percentile(p[fp_like], 50)) if nh_neg else float('nan'),
    },
}

feature_section = ""

top_lines = []
if features is not None and nh_pos > 0 and nh_neg > 0:
    X = np.asarray(features)
    if X.ndim != 2:
        X = X.reshape((X.shape[0], -1))

    A = X[fn_like]
    B = X[fp_like]

    mu_a = A.mean(axis=0)
    mu_b = B.mean(axis=0)
    std_a = A.std(axis=0)
    std_b = B.std(axis=0)
    pooled = np.sqrt(0.5 * (std_a**2 + std_b**2) + 1e-12)
    effect = (mu_a - mu_b) / pooled

    order = np.argsort(-np.abs(effect))
    k = min(30, order.size)
    top = order[:k]

    with open(feat_csv, 'w', encoding='utf-8') as f:
        f.write('dim,mean_hard_pos,mean_hard_neg,std_hard_pos,std_hard_neg,effect_size\n')
        for d in top:
            f.write(f"{int(d)},{mu_a[d]:.6f},{mu_b[d]:.6f},{std_a[d]:.6f},{std_b[d]:.6f},{effect[d]:.6f}\n")

    feature_section = "\n## Hard region feature contrast (hard_pos vs hard_neg)\n\n"
    feature_section += f"Computed on backbone features: D={X.shape[1]}. Ranked by |effect_size|.\n\n"
    feature_section += "Top dimensions (dim: effect_size, mean_pos, mean_neg):\n\n"
    for d in top[:10]:
        top_lines.append(f"- dim {int(d)}: effect={effect[d]:.3f}, mean_pos={mu_a[d]:.3f}, mean_neg={mu_b[d]:.3f}")
    feature_section += "\n".join(top_lines) + "\n"
else:
    with open(feat_csv, 'w', encoding='utf-8') as f:
        f.write('dim,mean_hard_pos,mean_hard_neg,std_hard_pos,std_hard_neg,effect_size\n')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"# Hard-sample analysis report\n\n")
    f.write(f"Input: `{os.path.abspath(npz_path)}`\n\n")
    f.write(f"Hard region: prob in [{low:.4f}, {high:.4f}]\n\n")

    f.write("## Counts\n\n")
    f.write(f"- Total samples: {n}\n")
    f.write(f"- Hard samples: {nh}\n")
    f.write(f"- Hard positives (y=1): {nh_pos}\n")
    f.write(f"- Hard negatives (y=0): {nh_neg}\n\n")

    f.write("## Probability stats\n\n")
    f.write("Group | mean | std | p50 |\n")
    f.write("---|---:|---:|---:|\n")
    for k in ['all','hard','hard_pos','hard_neg']:
        s = p_stats[k]
        f.write(f"{k} | {s['mean']:.6f} | {s['std']:.6f} | {s['p50']:.6f} |\n")
    f.write("\n")

    f.write("## Artifacts\n\n")
    f.write(f"- hard sample list: `{os.path.basename(hard_csv)}`\n")
    f.write(f"- feature diff: `{os.path.basename(feat_csv)}`\n")

    f.write(feature_section)

print("[OK] Wrote:")
print(" -", hard_csv)
print(" -", feat_csv)
print(" -", report_path)
PY

echo "[DONE] Report generated in: $OUT_DIR"
