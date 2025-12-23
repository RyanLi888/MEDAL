#!/bin/bash
#
# ML投票标签矫正实验运行脚本
# ============================
#
# 功能: 使用多个机器学习模型对低质量样本进行投票矫正
#
# 前置条件:
#   1. 已运行 train.py 完成 Stage 1-2（生成特征和矫正结果）
#   2. 存在以下文件:
#      - output/feature_extraction/models/train_features.npy
#      - output/label_correction/models/correction_results.npz
#
# 使用方法:
#   ./run.sh                    # 使用默认参数运行
#   ./run.sh --hq_ratio 0.6     # 自定义高质量样本比例
#   ./run.sh --balance_train    # 启用类别平衡训练
#

# 切换到MEDAL项目根目录
cd "$(dirname "$0")/../.." || exit 1

python - <<'PY'
try:
    import torch  # noqa
except Exception:
    raise SystemExit("❌ 错误: 当前Python环境缺少torch。请先激活包含torch的环境（如: conda activate RAPIER）再运行run.sh")
PY

# 默认参数
FEATURES="./output/feature_extraction/models/train_features.npy"
FEATURES_NPY=""
CORRECTION="./output/label_correction/models/correction_results.npz"
OUTPUT_DIR="./experiments/ml_vote_label_correction/output"
HQ_RATIO=0.5
VOTE_K=4
VOTE_RULE="hard"
BAND_LOW=-1
BAND_HIGH=8
VOTE_SCORE="count"
PROB_THRESHOLD=0.5
PROB_LOW=0.3
PROB_HIGH=0.7
MLP_HIDDEN=128
MLP_EPOCHS=200
MLP_LR=0.001
MLP_WEIGHT_DECAY=0.0001
HC_IMPL="experimental"
SKIP_PLOTS=""
SEED=42
USE_BASE="y_corrected"
BALANCE=""
MODE="from_preprocessed"
DATA_SPLIT="train"
NOISE_RATE=0.0
TRAIN_ON="keep"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --data_split)
            DATA_SPLIT="$2"
            shift 2
            ;;
        --noise_rate)
            NOISE_RATE="$2"
            shift 2
            ;;
        --train_on)
            TRAIN_ON="$2"
            shift 2
            ;;
        --vote_rule)
            VOTE_RULE="$2"
            shift 2
            ;;
        --band_low)
            BAND_LOW="$2"
            shift 2
            ;;
        --band_high)
            BAND_HIGH="$2"
            shift 2
            ;;
        --vote_score)
            VOTE_SCORE="$2"
            shift 2
            ;;
        --prob_threshold)
            PROB_THRESHOLD="$2"
            shift 2
            ;;
        --prob_low)
            PROB_LOW="$2"
            shift 2
            ;;
        --prob_high)
            PROB_HIGH="$2"
            shift 2
            ;;
        --mlp_hidden)
            MLP_HIDDEN="$2"
            shift 2
            ;;
        --mlp_epochs)
            MLP_EPOCHS="$2"
            shift 2
            ;;
        --mlp_lr)
            MLP_LR="$2"
            shift 2
            ;;
        --mlp_weight_decay)
            MLP_WEIGHT_DECAY="$2"
            shift 2
            ;;
        --hc_impl)
            HC_IMPL="$2"
            shift 2
            ;;
        --skip_plots)
            SKIP_PLOTS="--skip_plots"
            shift
            ;;
        --features_npy)
            FEATURES_NPY="$2"
            shift 2
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --correction)
            CORRECTION="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --hq_ratio)
            HQ_RATIO="$2"
            shift 2
            ;;
        --vote_k)
            VOTE_K="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --use_base)
            USE_BASE="$2"
            shift 2
            ;;
        --balance_train)
            BALANCE="--balance_train"
            shift
            ;;
        -h|--help)
            echo "ML投票标签矫正实验"
            echo ""
            echo "用法: ./run.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --mode STR          运行模式 legacy|from_preprocessed (默认: from_preprocessed)"
            echo "  --data_split STR    预处理数据切分 train|test (默认: train)"
            echo "  --noise_rate FLOAT  注入噪声率(用于可控评估) (默认: 0.0)"
            echo "  --train_on STR      训练集口径 keep|keep_flip|nondrop (默认: keep)"
            echo "  --vote_rule STR     投票覆盖策略 hard|band|none (默认: hard)"
            echo "  --band_low INT      band策略: vote_sum<=band_low 判为正常覆盖 (默认: -1)"
            echo "  --band_high INT     band策略: vote_sum>=band_high 判为恶意覆盖 (默认: 8)"
            echo "  --vote_score STR    投票打分 count|mean_proba|stacked (默认: count)"
            echo "  --prob_threshold F  概率硬覆盖阈值 (默认: 0.5)"
            echo "  --prob_low F        band策略(概率): <=prob_low 覆盖为正常 (默认: 0.3)"
            echo "  --prob_high F       band策略(概率): >=prob_high 覆盖为恶意 (默认: 0.7)"
            echo "  --mlp_hidden INT    MLP hidden size (默认: 128)"
            echo "  --mlp_epochs INT    MLP epochs (默认: 200)"
            echo "  --mlp_lr F          MLP learning rate (默认: 0.001)"
            echo "  --mlp_weight_decay F MLP weight decay (默认: 0.0001)"
            echo "  --hc_impl STR       HybridCourt实现 experimental|original (默认: experimental)"
            echo "  --skip_plots        跳过绘图与t-SNE(用于自动调参加速)"
            echo "  --features_npy PATH  (可选) 直接提供特征npy，跳过backbone特征提取 (默认: 空)"
            echo "  --features PATH      特征文件路径 (默认: output/feature_extraction/models/train_features.npy)"
            echo "  --correction PATH    矫正结果文件路径 (默认: output/label_correction/models/correction_results.npz)"
            echo "  --output_dir PATH    输出目录 (默认: experiments/ml_vote_label_correction/output)"
            echo "  --hq_ratio FLOAT     高质量样本比例 (默认: 0.5)"
            echo "  --vote_k INT         投票阈值 (默认: 4)"
            echo "  --seed INT           随机种子 (默认: 42)"
            echo "  --use_base STR       基础标签 y_corrected|y_noisy (默认: y_corrected)"
            echo "  --balance_train      启用类别平衡训练"
            echo "  -h, --help           显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查前置文件
echo "========================================"
echo "ML投票标签矫正实验"
echo "========================================"
echo ""

if [ "$MODE" = "legacy" ]; then
    if [ ! -f "$FEATURES" ]; then
        echo "❌ 错误: 特征文件不存在: $FEATURES"
        echo "   请先运行 train.py 完成 Stage 1-2"
        exit 1
    fi
    if [ ! -f "$CORRECTION" ]; then
        echo "❌ 错误: 矫正结果文件不存在: $CORRECTION"
        echo "   请先运行 train.py 完成 Stage 1-2"
        exit 1
    fi
else
    if [ -n "$FEATURES_NPY" ] && [ ! -f "$FEATURES_NPY" ]; then
        echo "❌ 错误: features_npy 不存在: $FEATURES_NPY"
        exit 1
    fi
fi

if [ "$MODE" = "legacy" ]; then
    echo "✓ 特征文件: $FEATURES"
else
    if [ -n "$FEATURES_NPY" ]; then
        echo "✓ 特征文件: $FEATURES_NPY"
    else
        echo "✓ 特征文件: (backbone提取)"
    fi
fi
if [ "$MODE" = "legacy" ]; then
    echo "✓ 矫正结果: $CORRECTION"
else
    echo "✓ 模式: $MODE (将基于预处理数据重新跑HybridCourt得到最新Keep/动作)"
fi
echo "✓ 输出目录: $OUTPUT_DIR"
echo ""
echo "参数配置:"
echo "  模式: $MODE"
echo "  数据切分: $DATA_SPLIT"
echo "  注入噪声率: $NOISE_RATE"
echo "  训练口径: $TRAIN_ON"
echo "  高质量样本比例: $HQ_RATIO"
echo "  投票阈值: $VOTE_K"
echo "  投票规则: $VOTE_RULE (band_low=$BAND_LOW, band_high=$BAND_HIGH)"
echo "  投票打分: $VOTE_SCORE (prob_th=$PROB_THRESHOLD, prob_low=$PROB_LOW, prob_high=$PROB_HIGH)"
echo "  MLP超参: hidden=$MLP_HIDDEN epochs=$MLP_EPOCHS lr=$MLP_LR wd=$MLP_WEIGHT_DECAY"
echo "  HybridCourt实现: $HC_IMPL"
echo "  跳过绘图: ${SKIP_PLOTS:-否}"
echo "  随机种子: $SEED"
echo "  基础标签: $USE_BASE"
echo "  类别平衡: ${BALANCE:-否}"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行实验
echo "开始运行..."
echo "----------------------------------------"

if [ "$MODE" = "legacy" ]; then
    PYTHONPATH=/home/lx/python/MEDAL:$PYTHONPATH python experiments/ml_vote_label_correction/run_ml_vote_label_correction.py \
        --mode legacy \
        --features "$FEATURES" \
        --correction "$CORRECTION" \
        --output_dir "$OUTPUT_DIR" \
        --hq_ratio "$HQ_RATIO" \
        --vote_k "$VOTE_K" \
        --vote_rule "$VOTE_RULE" \
        --band_low "$BAND_LOW" \
        --band_high "$BAND_HIGH" \
        --vote_score "$VOTE_SCORE" \
        --prob_threshold "$PROB_THRESHOLD" \
        --prob_low "$PROB_LOW" \
        --prob_high "$PROB_HIGH" \
        --mlp_hidden "$MLP_HIDDEN" \
        --mlp_epochs "$MLP_EPOCHS" \
        --mlp_lr "$MLP_LR" \
        --mlp_weight_decay "$MLP_WEIGHT_DECAY" \
        --hc_impl "$HC_IMPL" \
        $SKIP_PLOTS \
        --seed "$SEED" \
        --use_base "$USE_BASE" \
        --train_on "$TRAIN_ON" \
        $BALANCE
else
    EXTRA_FEATURES_ARG=""
    if [ -n "$FEATURES_NPY" ]; then
        EXTRA_FEATURES_ARG="--features_npy $FEATURES_NPY"
    fi

    PYTHONPATH=/home/lx/python/MEDAL:$PYTHONPATH python experiments/ml_vote_label_correction/run_ml_vote_label_correction.py \
        --mode from_preprocessed \
        --data_split "$DATA_SPLIT" \
        --noise_rate "$NOISE_RATE" \
        $EXTRA_FEATURES_ARG \
        --output_dir "$OUTPUT_DIR" \
        --vote_k "$VOTE_K" \
        --vote_rule "$VOTE_RULE" \
        --band_low "$BAND_LOW" \
        --band_high "$BAND_HIGH" \
        --vote_score "$VOTE_SCORE" \
        --prob_threshold "$PROB_THRESHOLD" \
        --prob_low "$PROB_LOW" \
        --prob_high "$PROB_HIGH" \
        --mlp_hidden "$MLP_HIDDEN" \
        --mlp_epochs "$MLP_EPOCHS" \
        --mlp_lr "$MLP_LR" \
        --mlp_weight_decay "$MLP_WEIGHT_DECAY" \
        --hc_impl "$HC_IMPL" \
        $SKIP_PLOTS \
        --seed "$SEED" \
        --train_on "$TRAIN_ON" \
        $BALANCE
fi

EXIT_CODE=$?

echo "----------------------------------------"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ 实验完成!"
    echo ""
    echo "输出文件:"
    echo "  $OUTPUT_DIR/summary.json          - 实验摘要"
    echo "  $OUTPUT_DIR/y_corrected_ml.npy    - ML矫正后的标签"
    echo "  $OUTPUT_DIR/quality_score.npy     - 样本质量分数"
    echo "  $OUTPUT_DIR/low_quality_votes.npz - 低质量样本投票详情"
else
    echo ""
    echo "❌ 实验失败 (退出码: $EXIT_CODE)"
fi

exit $EXIT_CODE
