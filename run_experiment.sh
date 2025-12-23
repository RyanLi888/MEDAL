#!/bin/bash
# MEDAL-Lite 统一运行脚本
# 支持前台/后台运行、实时日志、GPU选择

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        utils = [int(x.strip()) for x in result.stdout.strip().split('\n')]
        return utils
    except:
        return [0] * torch.cuda.device_count()

utils = get_gpu_utilization()
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    util = utils[i] if i < len(utils) else 0
    print(f'  {i}) {name} ({mem:.1f} GB) [占用率: {util}%]')
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

# 选择运行模式
echo "请选择运行模式:"
echo "1) 完整流程 (训练 + 测试)"
echo "2) 仅训练"
echo "3) 仅测试"
echo "4) 仅Stage 2 (标签矫正 + 数据增强)"
echo "5) 仅干净数据训练+测试 (不使用增强数据)"
echo "6) 从指定阶段开始 (训练/测试)"
echo "7) 消融实验 (特征提取 / 数据增强 / 标签矫正)"
echo ""
echo -n "请输入选择 (1-7): "
read -r choice

# 构建命令
case $choice in
    1)
        CMD="python all_train_test.py"
        MODE="完整流程"
        LOG_PREFIX="all_train_test"
        ;;
    2)
        CMD="python train.py"
        MODE="仅训练"
        LOG_PREFIX="train"
        ;;
    3)
        CMD="python test.py"
        MODE="仅测试"
        LOG_PREFIX="test"
        ;;
    4)
        echo ""
        echo "仅Stage 2 模式 (标签矫正 + 数据增强)"
        echo ""
        echo "说明: 将执行 train.py 的 Stage 2，并在完成后退出（不会训练分类器）"
        echo "  - 输出: output/feature_extraction/, output/label_correction/, output/data_augmentation/"
        echo ""
        echo -n "噪声率 (0.0-1.0, 默认0.30): "
        read -r noise_rate
        noise_rate=${noise_rate:-0.30}

        echo -n "是否使用随机骨干网络（不加载预训练）? (y/n, 默认n): "
        read -r retrain_backbone
        retrain_backbone=${retrain_backbone:-n}

        RETRAIN_ARG=""
        if [ "$retrain_backbone" = "y" ] || [ "$retrain_backbone" = "Y" ]; then
            RETRAIN_ARG="--retrain_backbone"
        fi

        CMD="python train.py --noise_rate $noise_rate --start_stage 2 --end_stage 2 $RETRAIN_ARG"
        MODE="仅Stage 2 (标签矫正 + 数据增强)"
        LOG_PREFIX="train_stage2"
        ;;
    5)
        echo ""
        echo "仅干净数据训练+测试 (不使用增强数据)"
        echo ""
        echo "说明: 将使用矫正后的干净数据训练分类器（不做TabDDPM增强），然后运行测试"
        echo "  - 依赖: output/label_correction/analysis/correction_results.npz (默认路径，可自定义)"
        echo ""

        CLEAN_ONLY_SCRIPT="train_clean_only_then_test.py"
        if [ ! -f "$CLEAN_ONLY_SCRIPT" ]; then
            echo -e "${RED}错误: 未找到脚本 $CLEAN_ONLY_SCRIPT${NC}"
            exit 1
        fi

        echo -n "correction_results.npz 路径 (直接回车使用默认): "
        read -r correction_npz

        echo -n "是否使用随机骨干网络（不加载预训练）? (y/n, 默认n): "
        read -r retrain_backbone
        retrain_backbone=${retrain_backbone:-n}

        RETRAIN_ARG=""
        if [ "$retrain_backbone" = "y" ] || [ "$retrain_backbone" = "Y" ]; then
            RETRAIN_ARG="--retrain_backbone"
        fi

        CORR_ARG=""
        if [ -n "$correction_npz" ]; then
            CORR_ARG="--correction_npz $correction_npz"
        fi

        CMD="python $CLEAN_ONLY_SCRIPT $CORR_ARG $RETRAIN_ARG"
        MODE="仅干净数据训练+测试 (不使用增强数据)"
        LOG_PREFIX="clean_only_train_test"
        ;;
    6)
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
        
        # 构建命令参数
        STAGE_ARG="--start_stage $start_stage"
        
        if [ "$start_stage" = "test" ] || [ "$start_stage" = "Test" ] || [ "$start_stage" = "TEST" ]; then
            # 仅测试模式
            CMD="python test.py"
            MODE="仅测试"
            LOG_PREFIX="test"
        else
            # 训练模式，询问是否包含测试
            echo -n "是否包含测试? (y/n, 默认y): "
            read -r include_test
            include_test=${include_test:-y}
            
            if [ "$include_test" = "y" ] || [ "$include_test" = "Y" ]; then
                CMD="python all_train_test.py $STAGE_ARG"
                MODE="从Stage $start_stage开始 (含测试)"
                LOG_PREFIX="all_train_test_stage${start_stage}"
            else
                CMD="python train.py $STAGE_ARG"
                MODE="从Stage $start_stage开始 (仅训练)"
                LOG_PREFIX="train_stage${start_stage}"
            fi
        fi
        ;;
    7)
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

        echo -n "是否使用随机骨干网络（不加载预训练）? (y/n, 默认n): "
        read -r retrain_backbone
        retrain_backbone=${retrain_backbone:-n}

        RETRAIN_ARG=""
        if [ "$retrain_backbone" = "y" ] || [ "$retrain_backbone" = "Y" ]; then
            RETRAIN_ARG="--retrain_backbone"
        fi

        case $ab_choice in
            1)
                echo ""
                echo "[消融-特征提取]"
                echo "说明: 将先运行 Stage 1 预训练骨干，然后用真实标签(权重=1)训练分类器并测试"
                echo ""
                CMD="python train.py --noise_rate 0.0 --start_stage 1 --end_stage 1 && python train_clean_only_then_test.py --use_ground_truth $RETRAIN_ARG"
                MODE="消融-特征提取"
                LOG_PREFIX="ablation_feature_extraction"
                ;;
            2)
                echo ""
                echo "[消融-数据增强]"
                echo "说明: 使用真实标签(无噪声/权重=1)，直接增强(TabDDPM)，再训练分类器并测试"
                echo ""
                CMD="python train.py --noise_rate 0.0 --start_stage 1 --end_stage 3 --stage2_mode clean_augment_only $RETRAIN_ARG && python test.py"
                MODE="消融-数据增强"
                LOG_PREFIX="ablation_data_augmentation"
                ;;
            3)
                echo ""
                echo "[消融-标签矫正]"
                echo "说明: 使用30%噪声进行标签矫正分析，然后用矫正结果训练分类器并测试"
                echo ""
                CORR_NPZ="output/label_correction/analysis/noise_30pct/correction_results.npz"
                CMD="python MoudleCode/label_correction/analysis/label_correction_analysis.py --noise_rate 0.30 $RETRAIN_ARG && python train_clean_only_then_test.py --correction_npz $CORR_NPZ $RETRAIN_ARG"
                MODE="消融-标签矫正"
                LOG_PREFIX="ablation_label_correction"
                ;;
            *)
                echo -e "${RED}无效选择${NC}"
                exit 1
                ;;
        esac
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
