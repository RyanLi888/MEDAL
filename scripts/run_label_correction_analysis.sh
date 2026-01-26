#!/bin/bash
# ============================================================
# MEDAL-Lite 标签矫正分析脚本 (重构版)
# ============================================================
# 说明：
# - 支持单个或批量噪声率分析
# - 使用 config.py 中的配置
# - 自动选择已有backbone或训练新的
# - 支持前台/后台运行
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
    echo "MEDAL-Lite 标签矫正分析"
    echo -e "==========================================${NC}"
    echo ""
}

select_backbone() {
    export USE_EXISTING_BACKBONE="false"
    export BACKBONE_PATH=""
    export RETRAIN_BACKBONE=""
    
    if [ ! -d "$BACKBONE_DIR" ]; then
        echo -e "${YELLOW}⚠ 未找到骨干网络目录，将使用随机初始化${NC}"
        RETRAIN_BACKBONE="--retrain_backbone"
        return
    fi
    
    local backbone_files=($(ls -t "$BACKBONE_DIR"/backbone_*.pth 2>/dev/null || true))
    
    if [ ${#backbone_files[@]} -eq 0 ]; then
        echo -e "${YELLOW}⚠ 未找到已训练的骨干网络，将使用随机初始化${NC}"
        RETRAIN_BACKBONE="--retrain_backbone"
        return
    fi
    
    echo -e "${GREEN}✓ 发现 ${#backbone_files[@]} 个已训练的骨干网络${NC}"
    echo -n "是否使用已有backbone? (y/n, 默认y): "
    read -r use_existing
    use_existing=${use_existing:-y}
    
    if [ "$use_existing" != "y" ]; then
        RETRAIN_BACKBONE="--retrain_backbone"
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

select_noise_rates() {
    echo ""
    echo "请选择噪声率:"
    echo "1) 单个噪声率 30%"
    echo "2) 单个噪声率 (自定义)"
    echo "3) 批量噪声率 (10%, 20%, 30%, 40%)"
    echo "4) 批量噪声率 (自定义)"
    echo ""
    echo -n "请选择 (1-4, 默认3): "
    read -r noise_choice
    noise_choice=${noise_choice:-3}
    
    case $noise_choice in
        1)
            NOISE_RATES="0.3"
            ;;
        2)
            echo -n "请输入噪声率 (例如 0.25): "
            read -r custom_rate
            NOISE_RATES="${custom_rate:-0.3}"
            ;;
        3)
            NOISE_RATES="0.1 0.2 0.3 0.4"
            ;;
        4)
            echo -n "请输入噪声率列表 (空格分隔, 例如 0.1 0.2 0.3): "
            read -r custom_rates
            NOISE_RATES="${custom_rates:-0.3}"
            ;;
        *)
            NOISE_RATES="0.1 0.2 0.3 0.4"
            ;;
    esac
    
    echo -e "${GREEN}✓ 噪声率: $NOISE_RATES${NC}"
}

select_run_mode() {
    echo ""
    echo "运行方式:"
    echo "1) 前台运行 (实时查看)"
    echo "2) 后台运行 (推荐)"
    echo ""
    echo -n "请选择 (1-2, 默认2): "
    read -r run_mode
    RUN_MODE=${run_mode:-2}
}

# ============================================================
# 主程序
# ============================================================

print_banner
select_backbone
select_noise_rates
select_run_mode

# 构建命令参数
PYTHON_SCRIPT="$PROJECT_ROOT/MoudleCode/label_correction/analysis/label_correction_analysis.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}❌ 未找到脚本: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# 准备主日志（汇总日志）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/label_correction_batch_${TIMESTAMP}.log"
ln -sf "$LOG_FILE" "$LOG_DIR/label_correction_live.log"

echo ""
echo -e "${YELLOW}配置:${NC}"
echo "  噪声率: $NOISE_RATES"
if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
    echo "  骨干网络: $(basename "$BACKBONE_PATH")"
else
    echo "  骨干网络: 随机初始化"
fi
echo "  主日志: $LOG_FILE (汇总信息)"
echo "  详细日志: 每个噪声率单独保存"
echo ""

# 构建运行脚本
read -ra RATES_ARRAY <<< "$NOISE_RATES"

run_analysis() {
    # 清晰的标题
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              MEDAL-Lite 标签矫正分析 (批量)                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  噪声率: $NOISE_RATES"
    if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
        echo "  骨干网络: $(basename "$BACKBONE_PATH")"
    else
        echo "  骨干网络: 随机初始化"
    fi
    echo ""
    
    # 重新解析噪声率数组
    read -ra rates_array <<< "$NOISE_RATES"
    local total=${#rates_array[@]}
    local completed=0
    local failed=0
    
    # 存储每个噪声率的结果摘要
    declare -A results_summary
    
    for noise_rate in "${rates_array[@]}"; do
        local noise_pct=$(printf "%.0f" $(awk "BEGIN {printf \"%.0f\", $noise_rate * 100}"))
        
        echo "┌────────────────────────────────────────────────────────────────┐"
        echo "│  噪声率: ${noise_pct}%  ($((completed+1))/${total})"
        echo "└────────────────────────────────────────────────────────────────┘"
        echo ""
        
        local cmd_args="--noise_rate $noise_rate"
        
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            cmd_args="$cmd_args --backbone_path $BACKBONE_PATH"
        elif [ -n "$RETRAIN_BACKBONE" ]; then
            cmd_args="$cmd_args $RETRAIN_BACKBONE"
        fi
        
        # 运行Python脚本并捕获输出
        local temp_output=$(mktemp)
        if python "$PYTHON_SCRIPT" $cmd_args 2>&1 | tee "$temp_output"; then
            # 提取摘要信息
            local summary_line=$(grep "^SUMMARY:" "$temp_output" | tail -1)
            if [ -n "$summary_line" ]; then
                results_summary["${noise_pct}"]="$summary_line"
            fi
            
            echo ""
            echo "  ✓ 噪声率 ${noise_pct}% 完成"
            completed=$((completed + 1))
        else
            echo ""
            echo "  ✗ 噪声率 ${noise_pct}% 失败"
            failed=$((failed + 1))
        fi
        rm -f "$temp_output"
        echo ""
    done
    
    # 显示汇总结果
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                        批量分析完成                            ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  总计: ${total}  |  成功: ${completed}  |  失败: ${failed}"
    echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 显示每个噪声率的关键指标
    if [ ${#results_summary[@]} -gt 0 ]; then
        echo "┌────────────────────────────────────────────────────────────────┐"
        echo "│  结果摘要                                                       │"
        echo "├────────────────────────────────────────────────────────────────┤"
        printf "│  %-8s │ %-12s │ %-12s │ %-10s │ %-8s │\n" "噪声率" "原始纯度" "最终纯度" "提升" "翻转数"
        echo "├────────────────────────────────────────────────────────────────┤"
        
        for noise_rate in "${rates_array[@]}"; do
            local noise_pct=$(printf "%.0f" $(awk "BEGIN {printf \"%.0f\", $noise_rate * 100}"))
            local summary="${results_summary["${noise_pct}"]}"
            if [ -n "$summary" ]; then
                # 解析摘要行
                local orig_purity=$(echo "$summary" | grep -oP 'original_purity=\K[0-9.]+')
                local final_purity=$(echo "$summary" | grep -oP 'final_purity=\K[0-9.]+')
                local improvement=$(echo "$summary" | grep -oP 'improvement=\K[0-9.+-]+')
                local flip_count=$(echo "$summary" | grep -oP 'flip_count=\K[0-9]+')
                
                printf "│  %-8s │ %-12s │ %-12s │ %-10s │ %-8s │\n" \
                    "${noise_pct}%" "${orig_purity}%" "${final_purity}%" "+${improvement}%" "${flip_count}"
            else
                printf "│  %-8s │ %-12s │ %-12s │ %-10s │ %-8s │\n" \
                    "${noise_pct}%" "-" "-" "-" "-"
            fi
        done
        echo "└────────────────────────────────────────────────────────────────┘"
        echo ""
    fi
    
    echo "  输出目录: ${OUTPUT_DIR}/label_correction/analysis/"
    echo ""
    
    if [ $failed -gt 0 ]; then
        return 1
    fi
}

export -f run_analysis
export PYTHON_SCRIPT NOISE_RATES USE_EXISTING_BACKBONE BACKBONE_PATH RETRAIN_BACKBONE OUTPUT_DIR

case $RUN_MODE in
    1)
        echo -e "${GREEN}启动前台运行...${NC}"
        run_analysis 2>&1 | tee "$LOG_FILE"
        ;;
    2)
        echo -e "${GREEN}启动后台运行...${NC}"
        nohup bash -c 'run_analysis' > "$LOG_FILE" 2>&1 &
        PID=$!
        
        echo "$PID" > "$LOG_DIR/label_correction.pid"
        echo -e "${GREEN}✓ 已在后台启动 (PID: $PID)${NC}"
        echo "主日志: $LOG_FILE (汇总信息)"
        echo "实时查看: tail -f $LOG_DIR/label_correction_live.log"
        echo ""
        echo "各噪声率的详细日志将保存到:"
        read -ra RATES_ARRAY <<< "$NOISE_RATES"
        for noise_rate in "${RATES_ARRAY[@]}"; do
            noise_pct=$(printf "%.0f" $(awk "BEGIN {printf \"%.0f\", $noise_rate * 100}"))
            echo "  ${noise_pct}%: ${LOG_DIR}/noise_${noise_pct}pct_analysis_*.log"
        done
        echo ""
        echo "停止进程: kill $PID"
        ;;
esac

echo ""
echo -e "${BLUE}输出目录: $OUTPUT_DIR${NC}"
