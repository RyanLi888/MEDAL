#!/bin/bash
# ============================================================
# MEDAL-Lite 统一运行脚本 (重构版 v2)
# ============================================================
# 说明：
# - 使用 config.py 中的最优配置
# - 复用主流程代码，减少重复
# - 支持前台/后台运行、GPU自动选择
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
BACKBONE_DIR="${OUTPUT_DIR}/feature_extraction/models"
mkdir -p "$LOG_DIR" "$BACKBONE_DIR"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ============================================================
# 工具函数
# ============================================================

print_banner() {
    echo -e "${BLUE}=========================================="
    echo "MEDAL-Lite 加密恶意流量检测系统 (v2)"
    echo -e "==========================================${NC}"
    echo ""
}

check_python() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}错误: 未找到 Python 环境${NC}"
        exit 1
    fi
    echo "Python 版本: $(python --version)"
}

show_system_info() {
    echo -e "${BLUE}系统资源状态:${NC}"
    
    # CPU 占用
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 2>/dev/null || echo "N/A")
    echo "  CPU 占用: ${cpu_usage}%"
    
    # 内存占用
    local mem_info=$(free -h 2>/dev/null | grep Mem | awk '{print $3 "/" $2}' || echo "N/A")
    echo "  内存占用: ${mem_info}"
    
    echo ""
}

select_gpu() {
    local gpu_available=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null)
    
    if [ "$gpu_available" != "yes" ]; then
        echo -e "${YELLOW}⚠ 未检测到GPU，将使用CPU${NC}"
        return
    fi
    
    local gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    
    if [ "$gpu_count" -eq 0 ]; then
        echo -e "${YELLOW}⚠ 未检测到可用GPU${NC}"
        return
    fi
    
    echo -e "${GREEN}✓ 检测到 $gpu_count 个GPU${NC}"
    
    # 显示GPU信息，包括显存使用情况
    python -c "
import torch
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                idx, name, total, used, free, util = parts[:6]
                print(f'  {idx}) {name}')
                print(f'      显存: {used}MB / {total}MB (空闲 {free}MB) | GPU利用率: {util}%')
    else:
        raise Exception('nvidia-smi failed')
except:
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  {i}) {name} ({total:.1f} GB)')
" 2>/dev/null
    
    echo ""
    if [ "$gpu_count" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
        echo -e "${GREEN}✓ 使用 GPU 0${NC}"
    else
        echo -n "请选择GPU (0-$((gpu_count-1)), 默认0): "
        read -r selected_gpu
        selected_gpu=${selected_gpu:-0}
        export CUDA_VISIBLE_DEVICES=$selected_gpu
        echo -e "${GREEN}✓ 使用 GPU $selected_gpu${NC}"
    fi
}

select_backbone() {
    local allow_existing=$1
    
    export USE_EXISTING_BACKBONE="false"
    export BACKBONE_PATH=""
    
    if [ "$allow_existing" != "true" ]; then
        return
    fi
    
    if [ ! -d "$BACKBONE_DIR" ]; then
        echo -e "${YELLOW}⚠ 未找到骨干网络目录${NC}"
        return
    fi
    
    local backbone_files=($(ls -t "$BACKBONE_DIR"/backbone_*.pth 2>/dev/null))
    
    if [ ${#backbone_files[@]} -eq 0 ]; then
        echo -e "${YELLOW}⚠ 未找到已训练的骨干网络${NC}"
        return
    fi
    
    echo -e "${GREEN}✓ 发现 ${#backbone_files[@]} 个已训练的骨干网络${NC}"
    echo -n "是否使用已有backbone? (y/n, 默认y): "
    read -r use_existing
    use_existing=${use_existing:-y}
    
    if [ "$use_existing" != "y" ]; then
        return
    fi
    
    echo ""
    echo "可用的骨干网络:"
    for i in "${!backbone_files[@]}"; do
        echo "  $((i+1))) $(basename "${backbone_files[$i]}")"
    done
    
    echo -n "选择模型 (1-${#backbone_files[@]}, 默认1): "
    read -r choice
    choice=${choice:-1}
    
    BACKBONE_PATH="${backbone_files[$((choice-1))]}"
    USE_EXISTING_BACKBONE="true"
    echo -e "${GREEN}✓ 已选择: $(basename "$BACKBONE_PATH")${NC}"
}

# ============================================================
# 主程序
# ============================================================

print_banner
check_python
show_system_info
select_gpu

echo ""
echo "请选择运行模式:"
echo "1) 完整流程 (训练 + 测试) - 使用config最优配置"
echo "2) 仅训练 - 使用config最优配置"
echo "3) 仅测试"
echo "4) 骨干网络训练 (Stage 1 only)"
echo "5) 骨干网络评估 (对比不同方法)"
echo "6) 干净数据训练+测试 (消融实验)"
echo "7) 数据增强训练+测试 (消融实验)"
echo "8) 完整流程去除数据增强 (特征提取+标签矫正+分类训练)"
echo ""
echo -n "请输入选择 (1-8): "
read -r choice

case $choice in
    1)
        select_backbone true
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python scripts/training/all_train_test.py --start_stage 2 --backbone_path $BACKBONE_PATH"
            MODE="完整流程 (使用已有backbone)"
        else
            CMD="python scripts/training/all_train_test.py"
            MODE="完整流程 (训练新backbone)"
        fi
        LOG_PREFIX="all_train_test"
        ;;
    2)
        select_backbone true
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python scripts/training/train.py --start_stage 2 --backbone_path $BACKBONE_PATH"
            MODE="仅训练 (使用已有backbone)"
        else
            CMD="python scripts/training/train.py"
            MODE="仅训练 (训练新backbone)"
        fi
        LOG_PREFIX="train"
        ;;
    3)
        CMD="python scripts/testing/test.py"
        MODE="仅测试"
        LOG_PREFIX="test"
        ;;
    4)
        echo ""
        echo "骨干网络训练方式:"
        echo "1) SimMTM (掩码重建)"
        echo "2) SimMTM + InfoNCE (推荐)"
        echo ""
        echo -n "请选择 (1-2, 默认2): "
        read -r pretrain_mode
        pretrain_mode=${pretrain_mode:-2}
        
        if [ "$pretrain_mode" = "1" ]; then
            CMD="python scripts/training/train.py --start_stage 1 --end_stage 1"
            MODE="骨干网络训练 (SimMTM)"
        else
            CMD="python scripts/training/train.py --start_stage 1 --end_stage 1"
            MODE="骨干网络训练 (SimMTM + InfoNCE)"
        fi
        LOG_PREFIX="backbone_train"
        ;;
    5)
        echo ""
        echo "骨干网络评估 - 对比不同对比学习方法"
        echo "将依次训练并评估: InfoNCE, NNCLR, SimSiam"
        echo ""
        CMD="python scripts/evaluate_backbone.py"
        MODE="骨干网络评估 (对比不同方法)"
        LOG_PREFIX="backbone_eval"
        ;;
    6)
        select_backbone true
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python scripts/training/train_clean_only_then_test.py --use_ground_truth --backbone_path $BACKBONE_PATH"
        else
            CMD="python scripts/training/train_clean_only_then_test.py --use_ground_truth"
        fi
        MODE="干净数据训练+测试 (消融实验)"
        LOG_PREFIX="clean_train_test"
        ;;
    7)
        select_backbone true
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python scripts/training/train_augmented_then_test.py --backbone_path $BACKBONE_PATH"
        else
            CMD="python scripts/training/train_augmented_then_test.py --retrain_backbone"
        fi
        MODE="数据增强训练+测试 (消融实验)"
        LOG_PREFIX="ablation_data_augmentation"
        ;;
    8)
        select_backbone true
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python scripts/training/train_no_augmentation_then_test.py --backbone_path $BACKBONE_PATH"
        else
            CMD="python scripts/training/train_no_augmentation_then_test.py --retrain_backbone"
        fi
        MODE="完整流程去除数据增强 (特征提取+标签矫正+分类训练)"
        LOG_PREFIX="no_augmentation_train_test"
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

# 运行方式
echo ""
echo "运行方式:"
echo "1) 前台运行 (实时查看)"
echo "2) 后台运行 (推荐)"
echo ""
echo -n "请选择 (1-2, 默认2): "
read -r run_mode
run_mode=${run_mode:-2}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${LOG_PREFIX}_${TIMESTAMP}.log"

# 创建实时日志链接
ln -sf "$LOG_FILE" "$LOG_DIR/live.log"

case $run_mode in
    1)
        echo -e "${GREEN}启动前台运行...${NC}"
        {
            echo "=========================================="
            echo "MEDAL-Lite 训练任务"
            echo "=========================================="
            echo "开始时间: $(date)"
            echo "运行模式: $MODE"
            echo "命令: $CMD"
            echo "=========================================="
            echo ""
        } | tee "$LOG_FILE"
        
        bash -c "$CMD" 2>&1 | tee -a "$LOG_FILE"
        ;;
    2)
        echo -e "${GREEN}启动后台运行...${NC}"
        {
            echo "=========================================="
            echo "MEDAL-Lite 训练任务"
            echo "=========================================="
            echo "开始时间: $(date)"
            echo "运行模式: $MODE"
            echo "命令: $CMD"
            echo "=========================================="
            echo ""
        } > "$LOG_FILE"
        
        nohup bash -c "$CMD" >> "$LOG_FILE" 2>&1 &
        PID=$!
        
        echo -e "${GREEN}✓ 已在后台启动 (PID: $PID)${NC}"
        echo "日志文件: $LOG_FILE"
        echo "实时日志: tail -f $LOG_DIR/live.log"
        ;;
esac

echo ""
echo -e "${BLUE}输出目录: $OUTPUT_DIR${NC}"
