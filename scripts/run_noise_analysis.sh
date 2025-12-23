#!/bin/bash
# 标签矫正噪声分析脚本
# 用法: ./run_noise_analysis.sh [噪声率1] [噪声率2] ...
# 示例: ./run_noise_analysis.sh 10 20 30 40
# 如果不指定参数，默认运行 10%, 20%, 30%, 40%

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate RAPIER

# 默认噪声率
DEFAULT_NOISE_RATES=(10 20 30 40)

# 如果有参数，使用参数指定的噪声率
if [ $# -gt 0 ]; then
    NOISE_RATES=("$@")
else
    NOISE_RATES=("${DEFAULT_NOISE_RATES[@]}")
fi

echo "========================================"
echo "MEDAL-Lite 标签矫正噪声分析"
echo "========================================"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "噪声率: ${NOISE_RATES[*]}%"
echo ""

# 运行每个噪声率的分析
for noise_pct in "${NOISE_RATES[@]}"; do
    noise_rate=$(echo "scale=2; $noise_pct / 100" | bc)
    echo "========================================"
    echo ">>> 开始分析噪声率 ${noise_pct}%..."
    echo "========================================"
    
    python MoudleCode/label_correction/analysis/label_correction_analysis.py \
        --noise_rate "$noise_rate"
    
    echo ""
    echo "<<< 噪声率 ${noise_pct}% 分析完成"
    echo ""
done

echo "========================================"
echo "🎉 所有噪声率分析完成!"
echo "========================================"
echo ""
echo "输出目录:"
for noise_pct in "${NOISE_RATES[@]}"; do
    echo "  - output/label_correction/analysis/noise_${noise_pct}pct/"
done
echo ""
echo "日志目录: output/logs/"
