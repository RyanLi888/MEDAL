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

# 检查已有的backbone模型
echo "检查已有的骨干网络模型..."
BACKBONE_DIR="output/feature_extraction/models"
if [ -d "$BACKBONE_DIR" ]; then
    # 查找所有backbone文件
    BACKBONE_FILES=($(ls -t "$BACKBONE_DIR"/backbone_*.pth 2>/dev/null))
    
    if [ ${#BACKBONE_FILES[@]} -gt 0 ]; then
        echo -e "${GREEN}✓ 发现 ${#BACKBONE_FILES[@]} 个已训练的骨干网络${NC}"
        echo ""
        echo "是否使用已有的骨干网络? (y/n, 默认n)"
        echo "  - 选择 y: 跳过Stage 1，直接使用已有backbone"
        echo "  - 选择 n: 重新训练新的backbone"
        echo ""
        echo -n "请输入选择: "
        read -r use_existing_backbone
        use_existing_backbone=${use_existing_backbone:-n}
        
        if [ "$use_existing_backbone" = "y" ] || [ "$use_existing_backbone" = "Y" ]; then
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
            SELECTED_BACKBONE_NAME=$(basename "$SELECTED_BACKBONE")
            
            echo -e "${GREEN}✓ 已选择: $SELECTED_BACKBONE_NAME${NC}"
            echo ""
            
            # 设置环境变量，后续脚本会使用
            export USE_EXISTING_BACKBONE="true"
            export BACKBONE_PATH="$SELECTED_BACKBONE"
            export START_FROM_STAGE=2
            
            echo "将从 Stage 2 开始运行（跳过骨干网络训练）"
            echo ""
        else
            export USE_EXISTING_BACKBONE="false"
            export START_FROM_STAGE=1
            echo ""
            echo "将重新训练新的骨干网络"
            echo ""
        fi
    else
        echo -e "${YELLOW}⚠ 未找到已训练的骨干网络${NC}"
        echo "将从 Stage 1 开始训练"
        echo ""
        export USE_EXISTING_BACKBONE="false"
        export START_FROM_STAGE=1
    fi
else
    echo -e "${YELLOW}⚠ 骨干网络目录不存在${NC}"
    echo "将从 Stage 1 开始训练"
    echo ""
    export USE_EXISTING_BACKBONE="false"
    export START_FROM_STAGE=1
fi

# 选择运行模式
echo "请选择运行模式:"
echo "1) 完整流程 (训练 + 测试)"
echo "2) 仅训练"
echo "3) 仅测试"
echo "4) 仅Stage 2 (标签矫正 + 数据增强)"
echo "5) 特征提取分析 (生成特征分布图和分析报告)"
echo "6) 从指定阶段开始 (训练/测试)"
echo "7) 消融实验 (特征提取 / 数据增强 / 标签矫正)"
echo "8) 对比实验 (SimMTM vs SimMTM+InfoNCE)"
echo ""
echo -n "请输入选择 (1-8): "
read -r choice

# 构建命令
case $choice in
    1)
        # 完整流程：使用开始时选择的backbone配置
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python all_train_test.py --start_stage $START_FROM_STAGE --backbone_path $BACKBONE_PATH"
            MODE="完整流程 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
        else
            CMD="python all_train_test.py"
            MODE="完整流程 (训练新backbone)"
        fi
        LOG_PREFIX="all_train_test"
        ;;
    2)
        # 仅训练：使用开始时选择的backbone配置
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python train.py --start_stage $START_FROM_STAGE --backbone_path $BACKBONE_PATH"
            MODE="仅训练 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
        else
            CMD="python train.py"
            MODE="仅训练 (训练新backbone)"
        fi
        LOG_PREFIX="train"
        ;;
    3)
        # 仅测试：不涉及backbone
        CMD="python test.py"
        MODE="仅测试"
        LOG_PREFIX="test"
        ;;
    4)
        # 仅Stage 2：使用开始时选择的backbone配置
        echo ""
        echo "仅Stage 2 模式 (标签矫正 + 数据增强)"
        echo ""
        echo "说明: 将执行 train.py 的 Stage 2，并在完成后退出（不会训练分类器）"
        echo "  - 输出: output/feature_extraction/, output/label_correction/, output/data_augmentation/"
        echo ""
        
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            echo "将使用已选择的骨干网络: $SELECTED_BACKBONE_NAME"
        else
            echo "将先训练新的骨干网络（Stage 1），然后执行Stage 2"
        fi
        echo ""
        
        echo -n "噪声率 (0.0-1.0, 默认0.30): "
        read -r noise_rate
        noise_rate=${noise_rate:-0.30}

        # 根据开始时的选择构建命令
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python train.py --noise_rate $noise_rate --start_stage 2 --end_stage 2 --backbone_path $BACKBONE_PATH"
            MODE="仅Stage 2 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
        else
            CMD="python train.py --noise_rate $noise_rate --start_stage 1 --end_stage 2"
            MODE="仅Stage 2 (训练新backbone)"
        fi
        LOG_PREFIX="train_stage2"
        ;;
    5)
        # 特征提取分析模式
        echo ""
        echo "特征提取分析模式"
        echo ""
        echo "说明: 使用骨干网络提取特征，生成特征分布图和详细分析报告"
        echo "  - 输出特征分布可视化 (t-SNE/PCA)"
        echo "  - 输出特征质量分析报告"
        echo "  - 输出特征可分性评估"
        echo ""
        
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            echo "将使用已选择的骨干网络: $SELECTED_BACKBONE_NAME"
            BACKBONE_TO_USE="$BACKBONE_PATH"
        else
            echo "将先训练新的骨干网络（Stage 1），然后进行特征分析"
            BACKBONE_TO_USE=""
        fi
        echo ""
        
        # 创建特征分析脚本
        FEATURE_ANALYSIS_SCRIPT="feature_analysis.py"
        
        cat > "$FEATURE_ANALYSIS_SCRIPT" << 'EOF'
"""
特征提取分析脚本
分析骨干网络提取的特征质量
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, silhouette_score, classification_report
from sklearn.decomposition import PCA

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger
from MoudleCode.utils.visualization import plot_feature_space
from MoudleCode.preprocessing.pcap_parser import load_dataset
from MoudleCode.feature_extraction.backbone import MicroBiMambaBackbone

try:
    from preprocess import check_preprocessed_exists, load_preprocessed
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="特征提取分析")
    parser.add_argument("--backbone_path", type=str, default=None, help="骨干网络路径")
    parser.add_argument("--train_backbone", action="store_true", help="是否训练新的骨干网络")
    args = parser.parse_args()
    
    # Setup
    set_seed(config.SEED)
    config.create_dirs()
    
    # 创建分析输出目录
    analysis_dir = os.path.join(config.OUTPUT_ROOT, "feature_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "reports"), exist_ok=True)
    
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, "logs"), name='feature_analysis')
    
    logger.info("="*70)
    logger.info("🔬 特征提取分析模式")
    logger.info("="*70)
    logger.info(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Load dataset
    logger.info("📦 加载训练数据集...")
    if PREPROCESS_AVAILABLE and check_preprocessed_exists('train'):
        X_train, y_train, _ = load_preprocessed('train')
        logger.info(f"  从预处理文件加载: {X_train.shape[0]} 个样本")
    else:
        X_train, y_train, _ = load_dataset(
            benign_dir=config.BENIGN_TRAIN,
            malicious_dir=config.MALICIOUS_TRAIN,
            sequence_length=config.SEQUENCE_LENGTH
        )
    
    logger.info(f"✓ 数据加载完成: {X_train.shape}")
    logger.info(f"  正常样本: {(y_train==0).sum()}, 恶意样本: {(y_train==1).sum()}")
    logger.info("")
    
    # Load or train backbone
    backbone = MicroBiMambaBackbone(config)
    
    if args.train_backbone or args.backbone_path is None:
        logger.info("🔧 训练新的骨干网络...")
        from torch.utils.data import TensorDataset, DataLoader
        import torch.optim as optim
        from MoudleCode.feature_extraction.backbone import SimMTMLoss
        
        backbone.train()
        backbone.to(config.DEVICE)
        
        dataset = TensorDataset(torch.FloatTensor(X_train))
        train_loader = DataLoader(dataset, batch_size=config.PRETRAIN_BATCH_SIZE, shuffle=True)
        
        simmtm_loss_fn = SimMTMLoss(mask_rate=config.SIMMTM_MASK_RATE)
        optimizer = optim.AdamW(backbone.parameters(), lr=config.PRETRAIN_LR)
        
        for epoch in range(config.PRETRAIN_EPOCHS):
            epoch_loss = 0.0
            for batch_data in train_loader:
                if isinstance(batch_data, (list, tuple)):
                    X_batch = batch_data[0]
                else:
                    X_batch = batch_data
                X_batch = X_batch.to(config.DEVICE)
                
                optimizer.zero_grad()
                loss = simmtm_loss_fn(backbone, X_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch [{epoch+1}/{config.PRETRAIN_EPOCHS}] Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Save backbone
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", f"backbone_analysis_{len(X_train)}.pth")
        torch.save(backbone.state_dict(), backbone_path)
        logger.info(f"✓ 骨干网络训练完成: {backbone_path}")
        logger.info("")
    else:
        backbone_path = args.backbone_path
        logger.info(f"📥 加载骨干网络: {backbone_path}")
        backbone.load_state_dict(torch.load(backbone_path, map_location=config.DEVICE))
        logger.info("✓ 骨干网络加载完成")
        logger.info("")
    
    # Extract features
    logger.info("🔍 提取特征...")
    backbone.freeze()
    backbone.eval()
    backbone.to(config.DEVICE)
    
    features_list = []
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
        batch_size = 64
        for i in range(0, len(X_tensor), batch_size):
            X_batch = X_tensor[i:i+batch_size]
            z_batch = backbone(X_batch, return_sequence=False)
            features_list.append(z_batch.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    logger.info(f"✓ 特征提取完成: {features.shape}")
    logger.info("")
    
    # Save features
    features_path = os.path.join(analysis_dir, "extracted_features.npy")
    np.save(features_path, features)
    logger.info(f"💾 特征已保存: {features_path}")
    logger.info("")
    
    # ========================
    # 特征质量分析
    # ========================
    logger.info("="*70)
    logger.info("📊 特征质量分析")
    logger.info("="*70)
    logger.info("")
    
    # 1. 特征可分性评估
    logger.info("1️⃣  特征可分性评估 (Logistic Regression)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, y_train, test_size=0.2, stratify=y_train, random_state=config.SEED
    )
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    
    te_proba = clf.predict_proba(X_te)[:, 1]
    te_pred = (te_proba >= 0.5).astype(int)
    
    te_auc = roc_auc_score(y_te, te_proba)
    te_f1 = f1_score(y_te, te_pred, pos_label=1)
    
    logger.info(f"  ROC-AUC: {te_auc:.4f}")
    logger.info(f"  F1-Score: {te_f1:.4f}")
    logger.info("")
    
    # 2. Silhouette Score
    logger.info("2️⃣  聚类质量评估 (Silhouette Score)")
    if len(np.unique(y_train)) > 1:
        sil_score = silhouette_score(features, y_train)
        logger.info(f"  Silhouette Score: {sil_score:.4f}")
        if sil_score > 0.5:
            logger.info("  ✅ 优秀 - 特征聚类质量很好")
        elif sil_score > 0.3:
            logger.info("  ✅ 良好 - 特征聚类质量较好")
        else:
            logger.info("  ⚠️  一般 - 特征聚类质量有待提升")
    logger.info("")
    
    # 3. PCA方差解释
    logger.info("3️⃣  主成分分析 (PCA)")
    pca = PCA(n_components=min(50, features.shape[1]))
    pca.fit(features)
    
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cumsum_var >= 0.95) + 1
    n_99 = np.argmax(cumsum_var >= 0.99) + 1
    
    logger.info(f"  前10个主成分解释方差: {cumsum_var[9]:.4f}")
    logger.info(f"  达到95%方差需要: {n_95} 个主成分")
    logger.info(f"  达到99%方差需要: {n_99} 个主成分")
    logger.info("")
    
    # 4. 特征统计
    logger.info("4️⃣  特征统计信息")
    logger.info(f"  特征维度: {features.shape[1]}")
    logger.info(f"  特征均值: {features.mean():.4f}")
    logger.info(f"  特征标准差: {features.std():.4f}")
    logger.info(f"  特征最小值: {features.min():.4f}")
    logger.info(f"  特征最大值: {features.max():.4f}")
    logger.info("")
    
    # ========================
    # 生成可视化
    # ========================
    logger.info("="*70)
    logger.info("📈 生成特征分布可视化")
    logger.info("="*70)
    logger.info("")
    
    # t-SNE
    logger.info("生成 t-SNE 可视化...")
    tsne_path = os.path.join(analysis_dir, "figures", "feature_distribution_tsne.png")
    plot_feature_space(features, y_train, tsne_path, 
                      title="Feature Distribution (t-SNE)", method='tsne')
    logger.info(f"  ✓ t-SNE图: {tsne_path}")
    
    # PCA
    logger.info("生成 PCA 可视化...")
    pca_path = os.path.join(analysis_dir, "figures", "feature_distribution_pca.png")
    plot_feature_space(features, y_train, pca_path,
                      title="Feature Distribution (PCA)", method='pca')
    logger.info(f"  ✓ PCA图: {pca_path}")
    logger.info("")
    
    # ========================
    # 生成分析报告
    # ========================
    logger.info("="*70)
    logger.info("📝 生成分析报告")
    logger.info("="*70)
    logger.info("")
    
    report_path = os.path.join(analysis_dir, "reports", "feature_analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 特征提取分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**骨干网络**: {os.path.basename(backbone_path)}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. 数据集信息\n\n")
        f.write(f"- **样本数**: {len(X_train)}\n")
        f.write(f"- **正常样本**: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.2f}%)\n")
        f.write(f"- **恶意样本**: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.2f}%)\n")
        f.write(f"- **序列长度**: {X_train.shape[1]}\n")
        f.write(f"- **特征维度**: {X_train.shape[2]}\n\n")
        
        f.write("---\n\n")
        f.write("## 2. 提取特征信息\n\n")
        f.write(f"- **特征维度**: {features.shape[1]}\n")
        f.write(f"- **特征均值**: {features.mean():.4f}\n")
        f.write(f"- **特征标准差**: {features.std():.4f}\n")
        f.write(f"- **特征范围**: [{features.min():.4f}, {features.max():.4f}]\n\n")
        
        f.write("---\n\n")
        f.write("## 3. 特征可分性评估\n\n")
        f.write("### Logistic Regression 性能\n\n")
        f.write(f"- **ROC-AUC**: {te_auc:.4f}\n")
        f.write(f"- **F1-Score**: {te_f1:.4f}\n\n")
        
        if te_auc >= 0.9 and te_f1 >= 0.8:
            f.write("✅ **优秀** - 特征具有很强的判别能力\n\n")
        elif te_auc >= 0.8 and te_f1 >= 0.7:
            f.write("✅ **良好** - 特征具有较好的判别能力\n\n")
        else:
            f.write("⚠️ **一般** - 特征判别能力有待提升\n\n")
        
        f.write("### Silhouette Score\n\n")
        if len(np.unique(y_train)) > 1:
            f.write(f"- **Silhouette Score**: {sil_score:.4f}\n\n")
            if sil_score > 0.5:
                f.write("✅ **优秀** - 类内紧密，类间分离良好\n\n")
            elif sil_score > 0.3:
                f.write("✅ **良好** - 类别分离较为明显\n\n")
            else:
                f.write("⚠️ **一般** - 类别分离不够明显\n\n")
        
        f.write("---\n\n")
        f.write("## 4. 主成分分析 (PCA)\n\n")
        f.write(f"- **前10个主成分解释方差**: {cumsum_var[9]*100:.2f}%\n")
        f.write(f"- **达到95%方差需要**: {n_95} 个主成分\n")
        f.write(f"- **达到99%方差需要**: {n_99} 个主成分\n\n")
        
        if n_95 < features.shape[1] * 0.2:
            f.write("✅ **信息集中度高** - 少量主成分即可表示大部分信息\n\n")
        else:
            f.write("⚠️ **信息较分散** - 需要较多主成分才能保留足够信息\n\n")
        
        f.write("---\n\n")
        f.write("## 5. 可视化结果\n\n")
        f.write(f"- **t-SNE图**: `{tsne_path}`\n")
        f.write(f"- **PCA图**: `{pca_path}`\n\n")
        
        f.write("---\n\n")
        f.write("## 6. 建议\n\n")
        
        if te_auc >= 0.9 and sil_score > 0.5:
            f.write("### ✅ 特征质量优秀\n\n")
            f.write("- 骨干网络学到了高质量的特征表示\n")
            f.write("- 可以直接用于下游分类任务\n")
            f.write("- 建议保存此骨干网络用于后续实验\n\n")
        elif te_auc >= 0.8:
            f.write("### ✅ 特征质量良好\n\n")
            f.write("- 骨干网络学到了较好的特征表示\n")
            f.write("- 可以用于分类任务，但仍有提升空间\n")
            f.write("- 建议尝试：\n")
            f.write("  - 增加预训练轮数\n")
            f.write("  - 调整掩码率\n")
            f.write("  - 尝试实例对比学习 (InfoNCE)\n\n")
        else:
            f.write("### ⚠️ 特征质量需要改进\n\n")
            f.write("- 骨干网络学到的特征判别能力不足\n")
            f.write("- 建议：\n")
            f.write("  - 检查数据质量\n")
            f.write("  - 增加预训练轮数\n")
            f.write("  - 调整网络结构\n")
            f.write("  - 尝试不同的预训练方法\n\n")
        
        f.write("---\n\n")
        f.write("*报告由MEDAL-Lite自动生成*\n")
    
    logger.info(f"✓ 分析报告已生成: {report_path}")
    logger.info("")
    
    logger.info("="*70)
    logger.info("🎉 特征分析完成!")
    logger.info("="*70)
    logger.info("")
    logger.info("📁 输出文件:")
    logger.info(f"  - 特征文件: {features_path}")
    logger.info(f"  - t-SNE图: {tsne_path}")
    logger.info(f"  - PCA图: {pca_path}")
    logger.info(f"  - 分析报告: {report_path}")
    logger.info("")

if __name__ == "__main__":
    main()
EOF
        
        # 构建命令
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            CMD="python $FEATURE_ANALYSIS_SCRIPT --backbone_path $BACKBONE_TO_USE"
            MODE="特征提取分析 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
        else
            CMD="python $FEATURE_ANALYSIS_SCRIPT --train_backbone"
            MODE="特征提取分析 (训练新backbone)"
        fi
        LOG_PREFIX="feature_analysis"
        ;;
    6)
        # 从指定阶段开始：使用开始时选择的backbone配置
        echo ""
        echo "从指定阶段开始模式"
        echo ""
        echo "训练阶段说明:"
        echo "  Stage 1: 骨干网络预训练 (SimMTM)"
        echo "  Stage 2: 标签矫正和数据增强"
        echo "  Stage 3: 分类器微调"
        echo "  test: 模型测试"
        echo ""
        
        if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            echo "已选择使用骨干网络: $SELECTED_BACKBONE_NAME"
            echo "建议从 Stage 2 或更高阶段开始"
            echo ""
        fi
        
        echo -n "请选择起始阶段 (1/2/3/test, 默认1): "
        read -r start_stage
        start_stage=${start_stage:-1}
        
        # 如果用户选择Stage 1但已有backbone，给出警告
        if [ "$start_stage" = "1" ] && [ "$USE_EXISTING_BACKBONE" = "true" ]; then
            echo ""
            echo -e "${YELLOW}警告: 你已选择使用已有backbone，但指定从Stage 1开始${NC}"
            echo "这将重新训练backbone，已选择的backbone将被忽略"
            echo -n "是否继续? (y/n): "
            read -r confirm
            if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
                echo "已取消"
                exit 0
            fi
            # 用户确认重新训练，清除backbone配置
            USE_EXISTING_BACKBONE="false"
            BACKBONE_PATH=""
        fi
        
        # 构建命令参数
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
            
            # 根据backbone选择和起始阶段构建命令
            STAGE_ARG="--start_stage $start_stage"
            BACKBONE_ARG=""
            
            if [ "$USE_EXISTING_BACKBONE" = "true" ] && [ "$start_stage" != "1" ]; then
                BACKBONE_ARG="--backbone_path $BACKBONE_PATH"
            fi
            
            if [ "$include_test" = "y" ] || [ "$include_test" = "Y" ]; then
                CMD="python all_train_test.py $STAGE_ARG $BACKBONE_ARG"
                if [ "$USE_EXISTING_BACKBONE" = "true" ] && [ "$start_stage" != "1" ]; then
                    MODE="从Stage $start_stage开始 (含测试, 使用已有backbone: $SELECTED_BACKBONE_NAME)"
                else
                    MODE="从Stage $start_stage开始 (含测试)"
                fi
                LOG_PREFIX="all_train_test_stage${start_stage}"
            else
                CMD="python train.py $STAGE_ARG $BACKBONE_ARG"
                if [ "$USE_EXISTING_BACKBONE" = "true" ] && [ "$start_stage" != "1" ]; then
                    MODE="从Stage $start_stage开始 (仅训练, 使用已有backbone: $SELECTED_BACKBONE_NAME)"
                else
                    MODE="从Stage $start_stage开始 (仅训练)"
                fi
                LOG_PREFIX="train_stage${start_stage}"
            fi
        fi
        ;;
    7)
        # 消融实验：根据模式和开始时的backbone选择决定行为
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
                # 消融实验模式1：特征提取
                # 目的是训练backbone，所以忽略开始时的backbone选择
                echo ""
                echo "[消融-特征提取]"
                echo "说明: 将先运行 Stage 1 预训练骨干，然后用真实标签(权重=1)训练分类器并测试"
                echo "  - Stage 1: 训练新的骨干网络"
                echo "  - Stage 2-3: 跳过，直接用真实标签训练分类器"
                echo ""
                
                if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
                    echo -e "${YELLOW}注意: 你已选择使用已有backbone ($SELECTED_BACKBONE_NAME)${NC}"
                    echo "但消融实验模式1的目的是验证特征提取，需要训练新backbone"
                    echo "已有backbone将被忽略"
                    echo ""
                fi
                
                CMD="python train.py --noise_rate 0.0 --start_stage 1 --end_stage 1 && python train_clean_only_then_test.py --use_ground_truth"
                MODE="消融-特征提取 (训练新backbone)"
                LOG_PREFIX="ablation_feature_extraction"
                ;;
            2)
                # 消融实验模式2：数据增强
                # 使用开始时的backbone选择
                echo ""
                echo "[消融-数据增强]"
                echo "说明: 使用真实标签(无噪声/权重=1)，直接增强(TabDDPM)，再训练分类器并测试"
                echo ""
                
                if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
                    echo "将使用已选择的骨干网络: $SELECTED_BACKBONE_NAME"
                    echo "跳过Stage 1，从Stage 2开始"
                    BACKBONE_ARG="--backbone_path $BACKBONE_PATH --start_stage 2"
                    CMD="python train.py --noise_rate 0.0 --end_stage 3 --stage2_mode clean_augment_only $BACKBONE_ARG && python test.py"
                    MODE="消融-数据增强 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
                else
                    echo "将训练新的骨干网络"
                    CMD="python train.py --noise_rate 0.0 --start_stage 1 --end_stage 3 --stage2_mode clean_augment_only && python test.py"
                    MODE="消融-数据增强 (训练新backbone)"
                fi
                LOG_PREFIX="ablation_data_augmentation"
                ;;
            3)
                # 消融实验模式3：标签矫正
                # 使用开始时的backbone选择
                echo ""
                echo "[消融-标签矫正]"
                echo "说明: 使用30%噪声进行标签矫正分析，然后用矫正结果训练分类器并测试"
                echo ""
                
                CORR_NPZ="output/label_correction/analysis/noise_30pct/correction_results.npz"
                
                if [ "$USE_EXISTING_BACKBONE" = "true" ]; then
                    echo "将使用已选择的骨干网络: $SELECTED_BACKBONE_NAME"
                    BACKBONE_ARG="--backbone_path $BACKBONE_PATH"
                    CMD="python MoudleCode/label_correction/analysis/label_correction_analysis.py --noise_rate 0.30 $BACKBONE_ARG && python train_clean_only_then_test.py --correction_npz $CORR_NPZ $BACKBONE_ARG"
                    MODE="消融-标签矫正 (使用已有backbone: $SELECTED_BACKBONE_NAME)"
                else
                    echo "将训练新的骨干网络"
                    CMD="python MoudleCode/label_correction/analysis/label_correction_analysis.py --noise_rate 0.30 && python train_clean_only_then_test.py --correction_npz $CORR_NPZ"
                    MODE="消融-标签矫正 (训练新backbone)"
                fi
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
