#!/bin/bash
# 对比实验脚本：SimMTM vs SimMTM+InfoNCE
# 依次训练两个骨干网络并进行性能对比

# 获取脚本所在目录（scripts/experiments/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 切换到项目根目录（python/MEDAL）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "对比实验: SimMTM vs SimMTM+InfoNCE"
echo -e "==========================================${NC}"
echo ""

# 创建对比实验目录
COMPARISON_DIR="output/comparison_experiments"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COMPARISON_RUN_DIR="$COMPARISON_DIR/run_$TIMESTAMP"
mkdir -p "$COMPARISON_RUN_DIR"

echo "对比实验将保存到: $COMPARISON_RUN_DIR"
echo ""
echo "说明:"
echo "  1) 训练 SimMTM-only backbone"
echo "  2) 训练 SimMTM+InfoNCE backbone"
echo "  3) 使用两个backbone分别进行测试"
echo "  4) 生成性能对比报告"
echo ""
echo -n "是否继续? (y/n, 默认y): "
read -r confirm
confirm=${confirm:-y}

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消"
    exit 0
fi

# 备份配置文件
CONFIG_FILE="MoudleCode/utils/config.py"
cp "$CONFIG_FILE" "$CONFIG_FILE.backup_comparison"

# ============================================================
# 1. 训练SimMTM-only backbone
# ============================================================
echo ""
echo -e "${GREEN}=========================================="
echo "[1/4] 训练 SimMTM-only backbone"
echo -e "==========================================${NC}"
echo "禁用实例对比学习..."

# 禁用实例对比学习
sed -i 's/USE_INSTANCE_CONTRASTIVE = True/USE_INSTANCE_CONTRASTIVE = False/' "$CONFIG_FILE"

echo "开始训练 SimMTM-only..."
echo "日志: $COMPARISON_RUN_DIR/simmtm_train.log"
echo ""

python scripts/training/train.py --noise_rate 0.0 --start_stage 1 --end_stage 1 > "$COMPARISON_RUN_DIR/simmtm_train.log" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ SimMTM-only 训练完成${NC}"
    # 复制backbone到对比实验目录
    SIMMTM_BACKBONE=$(ls output/feature_extraction/models/backbone_SimMTM_*.pth 2>/dev/null | head -1)
    if [ -f "$SIMMTM_BACKBONE" ]; then
        cp "$SIMMTM_BACKBONE" "$COMPARISON_RUN_DIR/"
        echo "  Backbone已保存: $(basename $SIMMTM_BACKBONE)"
    else
        cp output/feature_extraction/models/backbone_pretrained.pth "$COMPARISON_RUN_DIR/backbone_SimMTM.pth"
        echo "  Backbone已保存: backbone_SimMTM.pth"
    fi
else
    echo -e "${RED}✗ SimMTM-only 训练失败${NC}"
    echo "  查看日志: $COMPARISON_RUN_DIR/simmtm_train.log"
fi
echo ""

# ============================================================
# 2. 训练SimMTM+InfoNCE backbone
# ============================================================
echo -e "${GREEN}=========================================="
echo "[2/4] 训练 SimMTM+InfoNCE backbone"
echo -e "==========================================${NC}"
echo "启用实例对比学习..."

# 恢复并启用实例对比学习
cp "$CONFIG_FILE.backup_comparison" "$CONFIG_FILE"
sed -i 's/USE_INSTANCE_CONTRASTIVE = False/USE_INSTANCE_CONTRASTIVE = True/' "$CONFIG_FILE"

echo "开始训练 SimMTM+InfoNCE..."
echo "日志: $COMPARISON_RUN_DIR/simclr_train.log"
echo ""

python scripts/training/train.py --noise_rate 0.0 --start_stage 1 --end_stage 1 > "$COMPARISON_RUN_DIR/simclr_train.log" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ SimMTM+InfoNCE 训练完成${NC}"
    # 复制backbone到对比实验目录
    SIMCLR_BACKBONE=$(ls output/feature_extraction/models/backbone_SimCLR_*.pth 2>/dev/null | head -1)
    if [ -f "$SIMCLR_BACKBONE" ]; then
        cp "$SIMCLR_BACKBONE" "$COMPARISON_RUN_DIR/"
        echo "  Backbone已保存: $(basename $SIMCLR_BACKBONE)"
    else
        cp output/feature_extraction/models/backbone_pretrained.pth "$COMPARISON_RUN_DIR/backbone_SimCLR.pth"
        echo "  Backbone已保存: backbone_SimCLR.pth"
    fi
else
    echo -e "${RED}✗ SimMTM+InfoNCE 训练失败${NC}"
    echo "  查看日志: $COMPARISON_RUN_DIR/simclr_train.log"
fi
echo ""

# 恢复原始配置
mv "$CONFIG_FILE.backup_comparison" "$CONFIG_FILE"

# ============================================================
# 3. 使用SimMTM backbone进行测试
# ============================================================
echo -e "${GREEN}=========================================="
echo "[3/4] 测试 SimMTM-only backbone"
echo -e "==========================================${NC}"

SIMMTM_BACKBONE=$(ls "$COMPARISON_RUN_DIR"/backbone_SimMTM*.pth 2>/dev/null | head -1)
if [ -f "$SIMMTM_BACKBONE" ]; then
    echo "使用backbone: $(basename $SIMMTM_BACKBONE)"
    echo "日志: $COMPARISON_RUN_DIR/simmtm_test.log"
    echo ""
    
    python scripts/training/train_clean_only_then_test.py --use_ground_truth --backbone_path "$SIMMTM_BACKBONE" > "$COMPARISON_RUN_DIR/simmtm_test.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ SimMTM-only 测试完成${NC}"
    else
        echo -e "${RED}✗ SimMTM-only 测试失败${NC}"
    fi
else
    echo -e "${RED}✗ 未找到SimMTM backbone${NC}"
fi
echo ""

# ============================================================
# 4. 使用SimMTM+InfoNCE backbone进行测试
# ============================================================
echo -e "${GREEN}=========================================="
echo "[4/4] 测试 SimMTM+InfoNCE backbone"
echo -e "==========================================${NC}"

SIMCLR_BACKBONE=$(ls "$COMPARISON_RUN_DIR"/backbone_SimCLR*.pth 2>/dev/null | head -1)
if [ -f "$SIMCLR_BACKBONE" ]; then
    echo "使用backbone: $(basename $SIMCLR_BACKBONE)"
    echo "日志: $COMPARISON_RUN_DIR/simclr_test.log"
    echo ""
    
    python scripts/training/train_clean_only_then_test.py --use_ground_truth --backbone_path "$SIMCLR_BACKBONE" > "$COMPARISON_RUN_DIR/simclr_test.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ SimMTM+InfoNCE 测试完成${NC}"
    else
        echo -e "${RED}✗ SimMTM+InfoNCE 测试失败${NC}"
    fi
else
    echo -e "${RED}✗ 未找到SimCLR backbone${NC}"
fi
echo ""

# ============================================================
# 5. 生成对比报告
# ============================================================
echo -e "${GREEN}=========================================="
echo "[5/5] 生成对比报告"
echo -e "==========================================${NC}"

python - << 'PYTHON_EOF'
import re
import json
import sys
from pathlib import Path

comparison_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

def extract_metrics(log_file):
    """从日志文件中提取性能指标"""
    if not log_file.exists():
        return None
    
    content = log_file.read_text()
    metrics = {}
    
    # 提取F1, Precision, Recall, AUC
    patterns = {
        'accuracy': r'Accuracy:\s+([\d.]+)',
        'precision': r'Precision.*?:\s+([\d.]+)',
        'recall': r'Recall.*?:\s+([\d.]+)',
        'f1': r'F1.*?:\s+([\d.]+)',
        'auc': r'AUC:\s+([\d.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
    
    # 提取混淆矩阵
    cm_pattern = r'\[\[(\d+)\s+(\d+)\]\s+\[(\d+)\s+(\d+)\]\]'
    cm_match = re.search(cm_pattern, content)
    if cm_match:
        metrics['confusion_matrix'] = {
            'TN': int(cm_match.group(1)),
            'FP': int(cm_match.group(2)),
            'FN': int(cm_match.group(3)),
            'TP': int(cm_match.group(4))
        }
    
    return metrics

# 提取两个实验的指标
simmtm_metrics = extract_metrics(comparison_dir / "simmtm_test.log")
simclr_metrics = extract_metrics(comparison_dir / "simclr_test.log")

# 生成对比报告
report = []
report.append("=" * 70)
report.append("对比实验报告: SimMTM vs SimMTM+InfoNCE")
report.append("=" * 70)
report.append("")

if simmtm_metrics and simclr_metrics:
    report.append("性能对比:")
    report.append("-" * 70)
    report.append(f"{'指标':<15} | {'SimMTM':<12} | {'SimMTM+InfoNCE':<15} | {'变化':<12}")
    report.append("-" * 70)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        if metric in simmtm_metrics and metric in simclr_metrics:
            v1 = simmtm_metrics[metric]
            v2 = simclr_metrics[metric]
            diff = v2 - v1
            diff_pct = (diff / v1 * 100) if v1 > 0 else 0
            sign = "+" if diff > 0 else ""
            marker = "✓" if diff > 0 else ("✗" if diff < -0.01 else "≈")
            report.append(f"{metric.upper():<15} | {v1:<12.4f} | {v2:<15.4f} | {sign}{diff_pct:>6.2f}% {marker}")
    
    report.append("")
    report.append("混淆矩阵对比:")
    report.append("-" * 70)
    
    if 'confusion_matrix' in simmtm_metrics and 'confusion_matrix' in simclr_metrics:
        cm1 = simmtm_metrics['confusion_matrix']
        cm2 = simclr_metrics['confusion_matrix']
        
        report.append("SimMTM:")
        report.append(f"  [[{cm1['TN']:>5} {cm1['FP']:>5}]")
        report.append(f"   [{cm1['FN']:>5} {cm1['TP']:>5}]]")
        report.append("")
        report.append("SimMTM+InfoNCE:")
        report.append(f"  [[{cm2['TN']:>5} {cm2['FP']:>5}]")
        report.append(f"   [{cm2['FN']:>5} {cm2['TP']:>5}]]")
        report.append("")
        report.append("变化:")
        report.append(f"  TP: {cm1['TP']} → {cm2['TP']} ({cm2['TP']-cm1['TP']:+d})")
        report.append(f"  FN: {cm1['FN']} → {cm2['FN']} ({cm2['FN']-cm1['FN']:+d})")
        report.append(f"  FP: {cm1['FP']} → {cm2['FP']} ({cm2['FP']-cm1['FP']:+d})")
        report.append(f"  TN: {cm1['TN']} → {cm2['TN']} ({cm2['TN']-cm1['TN']:+d})")
else:
    report.append("⚠ 无法提取性能指标，请检查日志文件")
    report.append(f"  SimMTM日志: {comparison_dir / 'simmtm_test.log'}")
    report.append(f"  SimCLR日志: {comparison_dir / 'simclr_test.log'}")

report.append("")
report.append("=" * 70)
report.append("文件位置:")
report.append(f"  - SimMTM训练日志: {comparison_dir}/simmtm_train.log")
report.append(f"  - SimMTM测试日志: {comparison_dir}/simmtm_test.log")
report.append(f"  - SimCLR训练日志: {comparison_dir}/simclr_train.log")
report.append(f"  - SimCLR测试日志: {comparison_dir}/simclr_test.log")
report.append("=" * 70)

# 保存报告
report_text = "\n".join(report)
(comparison_dir / "comparison_report.txt").write_text(report_text)
print(report_text)

# 保存JSON格式
if simmtm_metrics and simclr_metrics:
    json_data = {
        'simmtm': simmtm_metrics,
        'simclr': simclr_metrics
    }
    (comparison_dir / "comparison_metrics.json").write_text(json.dumps(json_data, indent=2))
    print(f"\n✓ JSON数据已保存: {comparison_dir}/comparison_metrics.json")

PYTHON_EOF "$COMPARISON_RUN_DIR"

echo ""
echo -e "${GREEN}=========================================="
echo "✓ 对比实验完成"
echo -e "==========================================${NC}"
echo ""
echo "查看报告:"
echo "  cat $COMPARISON_RUN_DIR/comparison_report.txt"
echo ""
echo "查看详细日志:"
echo "  SimMTM训练: tail -f $COMPARISON_RUN_DIR/simmtm_train.log"
echo "  SimMTM测试: tail -f $COMPARISON_RUN_DIR/simmtm_test.log"
echo "  SimCLR训练: tail -f $COMPARISON_RUN_DIR/simclr_train.log"
echo "  SimCLR测试: tail -f $COMPARISON_RUN_DIR/simclr_test.log"
echo ""
