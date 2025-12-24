#!/bin/bash

# 骨干网络评估快速运行脚本

echo "========================================"
echo "骨干网络「上帝视角」评估"
echo "========================================"

# 获取脚本所在目录的父目录（MEDAL根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 如果提供了命令行参数，直接使用
if [ -n "$1" ]; then
    BACKBONE_PATH="$1"
else
    # 交互式选择骨干网络
    echo ""
    echo "请选择要评估的骨干网络:"
    echo ""
    
    # 查找所有可用的骨干网络模型
    BACKBONES=()
    idx=1
    
    # 查找 feature_extraction 目录下的模型
    if [ -d "output/feature_extraction/models" ]; then
        for model in output/feature_extraction/models/backbone_*.pth; do
            if [ -f "$model" ]; then
                echo "  [$idx] $model"
                BACKBONES+=("$model")
                ((idx++))
            fi
        done
    fi
    
    # 查找 comparison_experiments 目录下的模型
    if [ -d "output/comparison_experiments" ]; then
        for model in output/comparison_experiments/*/backbone_*.pth; do
            if [ -f "$model" ]; then
                echo "  [$idx] $model"
                BACKBONES+=("$model")
                ((idx++))
            fi
        done
    fi
    
    # 查找其他可能的位置
    for model in output/*/models/backbone*.pth; do
        if [ -f "$model" ]; then
            # 避免重复
            if [[ ! " ${BACKBONES[@]} " =~ " ${model} " ]]; then
                echo "  [$idx] $model"
                BACKBONES+=("$model")
                ((idx++))
            fi
        fi
    done
    
    if [ ${#BACKBONES[@]} -eq 0 ]; then
        echo "❌ 错误: 未找到任何骨干网络模型文件"
        echo "请先运行特征提取训练，或手动指定模型路径:"
        echo "  $0 <模型路径>"
        exit 1
    fi
    
    echo ""
    read -p "请输入选项 [1-${#BACKBONES[@]}]: " choice
    
    # 验证输入
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#BACKBONES[@]} ]; then
        echo "❌ 无效的选择"
        exit 1
    fi
    
    BACKBONE_PATH="${BACKBONES[$((choice-1))]}"
fi

# 其他配置参数
DATA_ROOT="output/preprocessed"  # 使用预处理后的数据
OUTPUT_DIR="output/backbone_eval"
DEVICE="cuda"

# 检查文件是否存在
if [ ! -f "$BACKBONE_PATH" ]; then
    echo "❌ 错误: 找不到骨干网络权重文件: $BACKBONE_PATH"
    exit 1
fi

if [ ! -f "$DATA_ROOT/train_X.npy" ] || [ ! -f "$DATA_ROOT/train_y.npy" ]; then
    echo "❌ 错误: 找不到预处理数据文件"
    echo "  需要: $DATA_ROOT/train_X.npy"
    echo "  需要: $DATA_ROOT/train_y.npy"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo ""
echo "配置信息:"
echo "  骨干网络: $BACKBONE_PATH"
echo "  数据集: $DATA_ROOT"
echo "  输出目录: $OUTPUT_DIR"
echo "  设备: $DEVICE"
echo ""

# 运行评估
python evaluate_backbone.py \
    --backbone "$BACKBONE_PATH" \
    --data_root "$DATA_ROOT" \
    --output "$OUTPUT_DIR" \
    --device "$DEVICE"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 评估完成！"
    echo "========================================"
    echo ""
    echo "查看结果:"
    echo "  - 可视化图: $OUTPUT_DIR/tsne_visualization.png"
    echo "  - 日志报告: $OUTPUT_DIR/evaluation_report.log"
    echo "  - JSON报告: $OUTPUT_DIR/evaluation_report.json"
    echo ""
    
    # 显示日志内容
    if [ -f "$OUTPUT_DIR/evaluation_report.log" ]; then
        echo "========================================"
        echo "评估报告摘要"
        echo "========================================"
        cat "$OUTPUT_DIR/evaluation_report.log"
    fi
    
    # 如果有 jq 工具，显示决策建议
    if command -v jq &> /dev/null; then
        echo "决策建议:"
        jq -r '.decision | "  等级: \(.grade)\n  准确率: \(.main_accuracy * 100 | round)%\n  需要 SupCon: \(.need_supcon)\n  \(.action)"' "$OUTPUT_DIR/evaluation_report.json"
    fi
else
    echo ""
    echo "❌ 评估失败，请检查错误信息"
    exit 1
fi
