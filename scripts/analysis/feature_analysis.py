"""
特征提取分析脚本
分析骨干网络提取的特征质量
"""
import sys
import os
# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

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
    from scripts.utils.preprocess import check_preprocessed_exists, load_preprocessed
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
        state = torch.load(backbone_path, map_location=config.DEVICE)
        try:
            backbone.load_state_dict(state)
        except RuntimeError as e:
            logger.warning(f"⚠ 骨干网络检查点与当前结构不完全匹配，将使用 strict=False 加载: {e}")
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"  missing_keys: {missing}")
            if unexpected:
                logger.warning(f"  unexpected_keys: {unexpected}")
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
