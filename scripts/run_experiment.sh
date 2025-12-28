#!/bin/bash
# MEDAL-Lite 统一运行脚本
# 支持前台/后台运行、实时日志、GPU选择

# 获取脚本所在目录（scripts/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 切换到项目根目录（python/MEDAL）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "MEDAL-Lite 加密恶意流量检测系统"
echo -e "==========================================${NC}"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python 环境${NC}"
    exit 1
fi

echo "检测到 Python 版本: $(python --version)"
echo ""

# 检查依赖
echo "检查依赖包..."
python -c "import torch; import numpy; import sklearn; import scapy.all" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}警告: 部分依赖包未安装${NC}"
    echo "是否现在安装? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        pip install -r requirements.txt
    else
        echo "请手动安装依赖: pip install -r requirements.txt"
        exit 1
    fi
fi

echo -e "${GREEN}依赖检查完成 ✓${NC}"
echo ""

# 检查并选择GPU
echo "检查GPU..."
gpu_available=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null)

if [ "$gpu_available" == "yes" ]; then
    echo -e "${GREEN}✓ 检测到可用的GPU${NC}"
    echo ""
    
    # 获取GPU数量和信息
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    
    if [ "$gpu_count" -gt 0 ]; then
        echo "可用GPU列表:"
        python -c "
import torch
import subprocess
import re

def get_gpu_info():
    try:
        # 获取显存使用率
        result_util = subprocess.run(['nvidia-smi', '--query-gpu=utilization.memory', '--format=csv,noheader,nounits'], 
                                    capture_output=True, text=True, check=True)
        utils = [int(x.strip()) for x in result_util.stdout.strip().split('\n')]
        
        # 获取显存使用情况 (已用/总量)
        result_mem = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True, check=True)
        mem_info = []
        for line in result_mem.stdout.strip().split('\n'):
            used, total = line.strip().split(',')
            mem_info.append((int(used.strip()), int(total.strip())))
        
        return utils, mem_info
    except:
        count = torch.cuda.device_count()
        return [0] * count, [(0, 0)] * count

utils, mem_info = get_gpu_info()
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    util = utils[i] if i < len(utils) else 0
    
    if i < len(mem_info):
        used_mb, total_mb = mem_info[i]
        used_gb = used_mb / 1024
        total_gb = total_mb / 1024
        free_gb = total_gb - used_gb
        print(f'  {i}) {name} ({total_mem:.1f} GB)')
        print(f'      显存: {used_gb:.2f}/{total_gb:.2f} GB (已用/总量) | 空闲: {free_gb:.2f} GB | 占用率: {util}%')
    else:
        print(f'  {i}) {name} ({total_mem:.1f} GB) [占用率: {util}%]')
" 2>/dev/null
        
        echo ""
        if [ "$gpu_count" -eq 1 ]; then
            echo "仅有1个GPU，将自动使用 GPU 0"
            export CUDA_VISIBLE_DEVICES=0
            selected_gpu=0
        else
            echo -n "请选择要使用的GPU (0-$((gpu_count-1)), 默认0): "
            read -r selected_gpu
            selected_gpu=${selected_gpu:-0}
            
            # 验证输入
            if ! [[ "$selected_gpu" =~ ^[0-9]+$ ]] || [ "$selected_gpu" -ge "$gpu_count" ]; then
                echo "无效选择，使用默认 GPU 0"
                selected_gpu=0
            fi
            
            export CUDA_VISIBLE_DEVICES=$selected_gpu
        fi
        
        echo -e "${GREEN}✓ 将使用 GPU $selected_gpu${NC}"
        
        # 显示选中GPU的详细信息
        python -c "
import torch
i = $selected_gpu
name = torch.cuda.get_device_name(i)
props = torch.cuda.get_device_properties(i)
mem = props.total_memory / 1024**3
compute = f'{props.major}.{props.minor}'
print(f'  GPU名称: {name}')
print(f'  显存: {mem:.1f} GB')
print(f'  计算能力: {compute}')
" 2>/dev/null
        echo ""
    else
        echo -e "${YELLOW}⚠ 未检测到可用GPU，将使用CPU (训练速度较慢)${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}⚠ PyTorch未检测到GPU，将使用CPU (训练速度较慢)${NC}"
    echo "  提示: 可运行 'python check_gpu.py' 查看详细信息"
    echo ""
fi

# 检查数据集
echo "检查数据集..."
if [ ! -d "Datasets/T1_train/benign" ] || [ ! -d "Datasets/T1_train/malicious" ]; then
    echo -e "${RED}警告: 未找到训练数据集${NC}"
    echo "请确保数据集位于:"
    echo "  - Datasets/T1_train/benign/"
    echo "  - Datasets/T1_train/malicious/"
    echo "  - Datasets/T2_test/benign/"
    echo "  - Datasets/T2_test/malicious/"
    exit 1
fi

echo -e "${GREEN}数据集检查完成 ✓${NC}"
echo ""

# 函数: 检查并选择骨干网络
# 参数: $1 - 是否允许选择已有backbone (true/false)
# 返回: 设置全局变量 USE_EXISTING_BACKBONE, BACKBONE_PATH, SELECTED_BACKBONE_NAME
check_and_select_backbone() {
    local allow_existing=$1
    
    # 初始化默认值
    export USE_EXISTING_BACKBONE="false"
    export BACKBONE_PATH=""
    export SELECTED_BACKBONE_NAME=""
    export START_FROM_STAGE=1
    
    # 如果不允许使用已有backbone，直接返回
    if [ "$allow_existing" != "true" ]; then
        return
    fi
    
    echo ""
    echo "检查已有的骨干网络模型..."
    BACKBONE_DIR="output/feature_extraction/models"
    
    if [ ! -d "$BACKBONE_DIR" ]; then
        echo -e "${YELLOW}⚠ 骨干网络目录不存在，将训练新的骨干网络${NC}"
        return
    fi
    
    # 查找所有backbone文件
    BACKBONE_FILES=($(ls -t "$BACKBONE_DIR"/backbone_*.pth 2>/dev/null))
    
    if [ ${#BACKBONE_FILES[@]} -eq 0 ]; then
        echo -e "${YELLOW}⚠ 未找到已训练的骨干网络，将训练新的骨干网络${NC}"
        return
    fi
    
    # 发现已有backbone，询问用户
    echo -e "${GREEN}✓ 发现 ${#BACKBONE_FILES[@]} 个已训练的骨干网络${NC}"
    echo ""
    echo "是否使用已有的骨干网络? (y/n, 默认n)"
    echo "  - 选择 y: 跳过Stage 1，直接使用已有backbone"
    echo "  - 选择 n: 重新训练新的backbone"
    echo ""
    echo -n "请输入选择: "
    read -r use_existing_backbone
    use_existing_backbone=${use_existing_backbone:-n}
    
    if [ "$use_existing_backbone" != "y" ] && [ "$use_existing_backbone" != "Y" ]; then
        echo ""
        echo "将重新训练新的骨干网络"
        return
    fi
    
    # 用户选择使用已有backbone
    echo ""
    echo "可用的骨干网络模型:"
    echo "----------------------------------------"
    for i in "${!BACKBONE_FILES[@]}"; do
        filename=$(basename "${BACKBONE_FILES[$i]}")
        filesize=$(du -h "${BACKBONE_FILES[$i]}" | cut -f1)
        filetime=$(stat -c %y "${BACKBONE_FILES[$i]}" 2>/dev/null | cut -d'.' -f1 || stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "${BACKBONE_FILES[$i]}")
        echo "  $((i+1))) $filename"
        echo "      大小: $filesize | 时间: $filetime"
    done
    echo "----------------------------------------"
    echo ""
    echo -n "请选择要使用的模型 (1-${#BACKBONE_FILES[@]}, 默认1): "
    read -r backbone_choice
    backbone_choice=${backbone_choice:-1}
    
    # 验证输入
    if ! [[ "$backbone_choice" =~ ^[0-9]+$ ]] || [ "$backbone_choice" -lt 1 ] || [ "$backbone_choice" -gt ${#BACKBONE_FILES[@]} ]; then
        echo "无效选择，使用第一个模型"
        backbone_choice=1
    fi
    
    SELECTED_BACKBONE="${BACKBONE_FILES[$((backbone_choice-1))]}"
    export SELECTED_BACKBONE_NAME=$(basename "$SELECTED_BACKBONE")
    export BACKBONE_PATH="$SELECTED_BACKBONE"
    export USE_EXISTING_BACKBONE="true"
    export START_FROM_STAGE=2
    
    echo -e "${GREEN}✓ 已选择: $SELECTED_BACKBONE_NAME${NC}"
    echo "将从 Stage 2 开始运行（跳过骨干网络训练）"
}

# 选择运行模式
echo "请选择运行模式:"
echo "1) 完整流程 (训练 + 测试)"
echo "2) 仅训练"
echo "3) 仅测试"
echo "4) 干净数据训练 (使用骨干网络提取干净训练集特征，训练分类器并测试)"
echo "5) 骨干网络评估 (评估已训练骨干网络的特征空间质量)"
echo "6) 从指定阶段开始 (训练/测试)"
echo "7) 消融实验 (特征提取 / 数据增强 / 标签矫正)"
echo "8) 骨干网络训练 (仅训练骨干网络，可选对比学习)"
echo ""
echo -n "请输入选择 (1-8): "
read -r choice

# 构建命令
case $choice in
    1)
        # 完整流程：询问是否使用已有backbone
        check_and_select_backbone true
        
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python scripts/training/all_train_test.py --start_stage $START_FROM_STAGE --backbone_path $BACKBONE_PATH"
            MODE="完整流程 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
        else
            CMD="python scripts/training/all_train_test.py"
            MODE="完整流程 (训练新backbone)"
        fi
        LOG_PREFIX="all_train_test"
        ;;
    2)
        # 仅训练：询问是否使用已有backbone
        check_and_select_backbone true
        
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python scripts/training/train.py --start_stage $START_FROM_STAGE --backbone_path $BACKBONE_PATH"
            MODE="仅训练 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
        else
            CMD="python scripts/training/train.py"
            MODE="仅训练 (训练新backbone)"
        fi
        LOG_PREFIX="train"
        ;;
    3)
        # 仅测试：不涉及backbone，不需要询问
        CMD="python scripts/testing/test.py"
        MODE="仅测试"
        LOG_PREFIX="test"
        ;;
    4)
        # 干净数据训练模式：需要backbone
        echo ""
        echo "干净数据训练模式"
        echo ""
        echo "说明: 使用骨干网络提取干净训练集（无噪声）的特征，训练分类器并测试"
        echo "  - 使用真实标签（无噪声注入）"
        echo "  - 跳过标签矫正和数据增强"
        echo "  - 直接训练 Stage 3 分类器"
        echo "  - 使用相同骨干网络进行测试"
        
        check_and_select_backbone true
        
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            echo ""
            echo "将使用已选择的骨干网络: $SELECTED_BACKBONE_NAME"
            CMD="python scripts/training/train_clean_only_then_test.py --use_ground_truth --backbone_path $BACKBONE_PATH"
            MODE="干净数据训练+测试 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
        else
            echo ""
            echo "将先训练新的骨干网络（Stage 1），然后用干净数据训练分类器"
            CMD="python scripts/training/train.py --noise_rate 0.0 --start_stage 1 --end_stage 1 && python scripts/training/train_clean_only_then_test.py --use_ground_truth"
            MODE="干净数据训练+测试 (训练新backbone)"
        fi
        LOG_PREFIX="clean_train_test"
        echo ""
        ;;
    5)
        # 骨干网络评估模式
        echo ""
        echo "骨干网络评估模式"
        echo ""
        echo "说明: 评估已训练骨干网络的特征空间质量"
        echo "  - 使用真实标签评估特征可分性"
        echo "  - 生成 t-SNE 可视化"
        echo "  - KNN 纯净度测试"
        echo "  - 决策建议（是否需要 SupCon 微调）"
        echo ""
        
        # 直接调用骨干网络评估脚本
        bash "$SCRIPT_DIR/run_backbone_eval.sh"
        exit 0
        ;;
    6)
        # 从指定阶段开始
        echo ""
        echo "从指定阶段开始模式"
        echo ""
        echo "训练阶段说明:"
        echo "  Stage 1: 骨干网络预训练 (SimMTM)"
        echo "  Stage 2: 标签矫正和数据增强"
        echo "  Stage 3: 分类器微调"
        echo "  test: 模型测试"
        echo ""
        
        echo -n "请选择起始阶段 (1/2/3/test, 默认1): "
        read -r start_stage
        start_stage=${start_stage:-1}
        
        # 根据起始阶段决定是否需要backbone
        if [ "$start_stage" = "1" ]; then
            # 从Stage 1开始，不需要已有backbone
            check_and_select_backbone false
        elif [ "$start_stage" = "test" ] || [ "$start_stage" = "Test" ] || [ "$start_stage" = "TEST" ]; then
            # 仅测试，不需要backbone选择
            check_and_select_backbone false
        else
            # 从Stage 2或3开始，询问是否使用已有backbone
            check_and_select_backbone true
        fi
        
        # 构建命令参数
        if [ "$start_stage" = "test" ] || [ "$start_stage" = "Test" ] || [ "$start_stage" = "TEST" ]; then
            # 仅测试模式
            CMD="python scripts/testing/test.py"
            MODE="仅测试"
            LOG_PREFIX="test"
        else
            # 训练模式，询问是否包含测试
            echo ""
            echo -n "是否包含测试? (y/n, 默认y): "
            read -r include_test
            include_test=${include_test:-y}
            
            # 根据backbone选择和起始阶段构建命令
            STAGE_ARG="--start_stage $start_stage"
            BACKBONE_ARG=""
            
            if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
                BACKBONE_ARG="--backbone_path $BACKBONE_PATH"
            fi
            
            if [ "$include_test" = "y" ] || [ "$include_test" = "Y" ]; then
                CMD="python scripts/training/all_train_test.py $STAGE_ARG $BACKBONE_ARG"
                if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
                    MODE="从Stage $start_stage开始 (含测试, 使用已有backbone: $SELECTED_BACKBONE_NAME)"
                else
                    MODE="从Stage $start_stage开始 (含测试)"
                fi
                LOG_PREFIX="all_train_test_stage${start_stage}"
            else
                CMD="python scripts/training/train.py $STAGE_ARG $BACKBONE_ARG"
                if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
                    MODE="从Stage $start_stage开始 (仅训练, 使用已有backbone: $SELECTED_BACKBONE_NAME)"
                else
                    MODE="从Stage $start_stage开始 (仅训练)"
                fi
                LOG_PREFIX="train_stage${start_stage}"
            fi
        fi
        ;;
    7)
        # 消融实验
        echo ""
        echo "消融实验模式"
        echo ""
        echo "请选择消融实验类型:"
        echo "1) 特征提取: 先训练特征提取器(Stage1)，用真实标签(权重1/无噪声)训练分类器，然后测试"
        echo "2) 数据增强: 用真实标签(权重1/无噪声)提取特征后直接增强(TabDDPM)，用真实+增强训练分类器，然后测试"
        echo "3) 标签矫正: 使用30%噪声提取特征->标签矫正->用矫正后的干净数据训练分类器，然后测试"
        echo ""
        echo -n "请输入选择 (1-3): "
        read -r ab_choice

        case $ab_choice in
            1)
                # 消融实验模式1：特征提取 - 可选择训练新backbone或使用已有backbone
                echo ""
                echo "[消融-特征提取]"
                echo "说明: 评估特征提取器（骨干网络）的质量"
                echo "  - 选项1: 训练新的骨干网络（Stage 1），然后用真实标签训练分类器并测试"
                echo "  - 选项2: 使用已有骨干网络，直接用真实标签训练分类器并测试"
                echo ""
                
                check_and_select_backbone true
                
                if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
                    echo ""
                    echo "将使用已选择的骨干网络: $SELECTED_BACKBONE_NAME"
                    echo "跳过 Stage 1，直接用真实标签训练分类器并测试"
                    BACKBONE_ARG="--backbone_path $BACKBONE_PATH"
                    CMD="python scripts/training/train_clean_only_then_test.py --use_ground_truth $BACKBONE_ARG"
                    MODE="消融-特征提取 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
                else
                    echo ""
                    echo "将训练新的骨干网络（Stage 1），然后用真实标签训练分类器并测试"
                    CMD="python scripts/training/train.py --noise_rate 0.0 --start_stage 1 --end_stage 1 && python scripts/training/train_clean_only_then_test.py --use_ground_truth"
                    MODE="消融-特征提取 (训练新backbone)"
                fi
                LOG_PREFIX="ablation_feature_extraction"
                ;;
            2)
                # 消融实验模式2：数据增强 - 需要backbone
                echo ""
                echo "[消融-数据增强]"
                echo "说明: 使用真实标签(无噪声/权重=1)，直接增强(TabDDPM)，再训练分类器并测试"
                
                check_and_select_backbone true
                
                if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
                    echo ""
                    echo "将使用已选择的骨干网络: $SELECTED_BACKBONE_NAME"
                    echo "跳过Stage 1，从Stage 2开始"
                    BACKBONE_ARG="--backbone_path $BACKBONE_PATH --start_stage 2"
                    CMD="python scripts/training/train.py --noise_rate 0.0 --end_stage 3 --stage2_mode clean_augment_only $BACKBONE_ARG && python scripts/testing/test.py"
                    MODE="消融-数据增强 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
                else
                    echo ""
                    echo "将训练新的骨干网络"
                    CMD="python scripts/training/train.py --noise_rate 0.0 --start_stage 1 --end_stage 3 --stage2_mode clean_augment_only && python scripts/testing/test.py"
                    MODE="消融-数据增强 (训练新backbone)"
                fi
                LOG_PREFIX="ablation_data_augmentation"
                ;;
            3)
                # 消融实验模式3：标签矫正 - 需要backbone
                echo ""
                echo "[消融-标签矫正]"
                echo "说明: 使用30%噪声进行标签矫正分析，然后用矫正结果训练分类器并测试"
                
                check_and_select_backbone true
                
                CORR_NPZ="output/label_correction/analysis/noise_30pct/correction_results.npz"
                
                if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
                    echo ""
                    echo "将使用已选择的骨干网络: $SELECTED_BACKBONE_NAME"
                    BACKBONE_ARG="--backbone_path $BACKBONE_PATH"
                    CMD="python MoudleCode/label_correction/analysis/label_correction_analysis.py --noise_rate 0.30 $BACKBONE_ARG && python scripts/training/train_clean_only_then_test.py --correction_npz $CORR_NPZ $BACKBONE_ARG"
                    MODE="消融-标签矫正 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
                else
                    echo ""
                    echo "将训练新的骨干网络"
                    CMD="python MoudleCode/label_correction/analysis/label_correction_analysis.py --noise_rate 0.30 && python scripts/training/train_clean_only_then_test.py --correction_npz $CORR_NPZ"
                    MODE="消融-标签矫正 (训练新backbone)"
                fi
                LOG_PREFIX="ablation_label_correction"
                ;;
            *)
                echo -e "${RED}无效选择${NC}"
                exit 1
                ;;
        esac
        echo ""
        ;;
    8)
        # 骨干网络训练模式 - 目的是训练新backbone，不需要已有backbone
        echo ""
        echo "骨干网络训练模式"
        echo ""
        echo "说明: 仅训练骨干网络（Stage 1），可选择是否使用对比学习"
        echo ""
        echo "请选择训练方式:"
        echo "1) SimMTM (掩码时序建模)"
        echo "2) SimMTM + InfoNCE"
        echo "3) SimMTM + SimSiam"
        echo "4) SimMTM + NNCLR"
        echo ""
        echo -n "请输入选择 (1-4, 默认1): "
        read -r pretrain_mode
        pretrain_mode=${pretrain_mode:-1}
        
        check_and_select_backbone false
        
        case $pretrain_mode in
            1)
                echo ""
                echo "使用 SimMTM 训练骨干网络"
                CMD="python scripts/training/train.py --start_stage 1 --end_stage 1"
                MODE="骨干网络训练 (SimMTM)"
                LOG_PREFIX="backbone_simmtm"
                ;;
            2)
                echo ""
                echo "使用 SimMTM + InfoNCE 训练骨干网络"
                echo "注意: 对比学习权重/温度由配置文件控制 (MoudleCode/utils/config.py)"
                echo "  - USE_INSTANCE_CONTRASTIVE / INFONCE_LAMBDA / INFONCE_TEMPERATURE"
                CMD="MEDAL_CONTRASTIVE_METHOD=infonce python scripts/training/train.py --start_stage 1 --end_stage 1"
                MODE="骨干网络训练 (SimMTM + InfoNCE)"
                LOG_PREFIX="backbone_hybrid"
                ;;
            3)
                echo ""
                echo "使用 SimMTM + SimSiam 训练骨干网络"
                echo "注意: 对比学习权重由配置文件控制 (MoudleCode/utils/config.py)"
                echo "  - USE_INSTANCE_CONTRASTIVE / INFONCE_LAMBDA"
                CMD="MEDAL_CONTRASTIVE_METHOD=simsiam python scripts/training/train.py --start_stage 1 --end_stage 1"
                MODE="骨干网络训练 (SimMTM + SimSiam)"
                LOG_PREFIX="backbone_simsiam"
                ;;
            4)
                echo ""
                echo "使用 SimMTM + NNCLR 训练骨干网络"
                echo "注意: 对比学习权重/温度由配置文件控制 (MoudleCode/utils/config.py)"
                echo "  - USE_INSTANCE_CONTRASTIVE / INFONCE_LAMBDA / INFONCE_TEMPERATURE / NNCLR_QUEUE_SIZE"
                CMD="MEDAL_CONTRASTIVE_METHOD=nnclr python scripts/training/train.py --start_stage 1 --end_stage 1"
                MODE="骨干网络训练 (SimMTM + NNCLR)"
                LOG_PREFIX="backbone_nnclr"
                ;;
            *)
                echo -e "${RED}无效选择${NC}"
                exit 1
                ;;
        esac
        
        echo ""
        echo "训练完成后，可以使用模式5评估骨干网络质量"
        echo ""
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

# 选择运行方式
echo ""
echo "请选择运行方式:"
echo "1) 前台运行 (实时查看输出，Ctrl+C可终止)"
echo "2) 后台运行 (推荐长时间训练，日志保存到文件)"
echo ""
echo -n "请输入选择 (1-2, 默认1): "
read -r run_mode
run_mode=${run_mode:-1}

# 准备日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/${LOG_PREFIX}_${TIMESTAMP}.log"
LIVE_LOG="$LOG_DIR/live.log"

case $run_mode in
    1)
        # 前台运行，同时保存日志
        echo ""
        echo -e "${GREEN}启动前台训练...${NC}"
        echo "按 Ctrl+C 可随时终止"
        echo "日志同时保存到: $MAIN_LOG"
    echo ""
        sleep 1
        
        # 写入运行信息
        {
            echo "=========================================="
            echo "MEDAL-Lite 训练任务"
            echo "=========================================="
            echo "开始时间: $(date)"
            echo "运行模式: $MODE"
            echo "使用GPU: ${selected_gpu:-CPU}"
            echo "命令: $CMD"
    echo "=========================================="
            echo ""
        } | tee "$MAIN_LOG"
        
        # 使用 tee 同时输出到终端和日志文件
        bash -c "$CMD" 2>&1 | tee -a "$MAIN_LOG"
        
        EXIT_CODE=${PIPESTATUS[0]}
        
        echo ""
        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}=========================================="
    echo "运行完成! ✓"
            echo -e "==========================================${NC}"
        else
            echo -e "${YELLOW}=========================================="
            echo "运行异常退出 (退出码: $EXIT_CODE)"
            echo -e "==========================================${NC}"
        fi
        echo "日志已保存到: $MAIN_LOG"
        ;;
        
    2)
        # 后台运行
        echo ""
        echo -e "${GREEN}启动后台训练...${NC}"
        echo ""
        
        # 检查是否有正在运行的训练
        if [ -f "output/train.pid" ]; then
            OLD_PID=$(cat output/train.pid)
            if ps -p $OLD_PID > /dev/null 2>&1; then
                echo -e "${YELLOW}已有训练进程在运行 (PID: $OLD_PID)${NC}"
                echo -n "是否停止当前训练并启动新的? (y/n): "
                read -r response
                if [[ "$response" == "y" ]]; then
                    kill $OLD_PID 2>/dev/null
                    sleep 2
                    echo "已停止旧进程"
                else
                    echo "退出"
                    exit 0
                fi
            fi
        fi
        
        # 写入运行信息到日志
        {
            echo "=========================================="
            echo "MEDAL-Lite 训练任务"
            echo "=========================================="
            echo "开始时间: $(date)"
            echo "运行模式: $MODE"
            echo "使用GPU: ${selected_gpu:-CPU}"
            echo "命令: $CMD"
    echo "=========================================="
            echo ""
        } > "$MAIN_LOG"
        
        # 创建实时日志链接
        ln -sf "$(basename $MAIN_LOG)" "$LIVE_LOG"
        
        # 后台运行并保存日志
        nohup bash -c "$CMD" >> "$MAIN_LOG" 2>&1 &
        PID=$!
        echo $PID > output/train.pid
        
        echo -e "${GREEN}✓ 训练已在后台启动${NC}"
        echo ""
        echo "进程 PID: $PID"
        echo "日志文件: $MAIN_LOG"
        echo ""
        echo -e "${BLUE}常用命令:${NC}"
        echo "  查看实时日志: tail -f $MAIN_LOG"
        echo "  查看最新日志: tail -f $LIVE_LOG"
        echo "  监控训练进度: watch -n 5 'tail -n 30 $MAIN_LOG | grep -E \"Epoch|Loss|Accuracy\"'"
        echo "  停止训练: kill $PID"
        echo "  检查进程: ps aux | grep $PID"
        echo ""
        
        # 等待1秒并显示初始输出
        sleep 2
        if [ -f "$MAIN_LOG" ] && [ -s "$MAIN_LOG" ]; then
            echo -e "${YELLOW}最新日志输出:${NC}"
            echo "----------------------------------------"
            tail -n 20 "$MAIN_LOG"
            echo "----------------------------------------"
            echo ""
        fi
        
        echo -e "${GREEN}训练正在后台运行中...${NC}"
        echo "使用上述命令查看实时进度"
        ;;
        
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

# 显示输出位置
if [ $? -eq 0 ] || [ "$run_mode" == "2" ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "输出位置"
    echo -e "==========================================${NC}"
    echo "  - 特征提取: output/feature_extraction/"
    echo "  - 标签矫正: output/label_correction/"
    echo "  - 数据增强: output/data_augmentation/"
    echo "  - 分类器:   output/classification/"
    echo "  - 测试结果: output/result/"
    echo "  - 训练日志: output/logs/"
    echo ""
fi
