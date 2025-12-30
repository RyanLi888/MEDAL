"""
MEDAL-Lite Training Script
Implements the complete 3-stage training pipeline
"""
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path when running as a script
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
from datetime import datetime

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import (
    set_seed, setup_logger, inject_label_noise,
    calculate_metrics, print_metrics, save_checkpoint, find_optimal_threshold
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, silhouette_score
from MoudleCode.utils.visualization import (
    plot_feature_space, plot_noise_correction_comparison, plot_training_history
)
from MoudleCode.preprocessing.pcap_parser import load_dataset
from MoudleCode.feature_extraction.backbone import (
    MicroBiMambaBackbone, SimMTMLoss
)
from MoudleCode.feature_extraction.traffic_augmentation import DualViewAugmentation
from MoudleCode.feature_extraction.instance_contrastive import (
    InstanceContrastiveLearning, HybridPretrainingLoss
)
from MoudleCode.utils.checkpoint import load_state_dict_shape_safe

# 导入预处理模块
try:
    from scripts.utils.preprocess import check_preprocessed_exists, load_preprocessed, preprocess_train
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False
from MoudleCode.label_correction.hybrid_court import HybridCourt
from MoudleCode.data_augmentation.tabddpm import TabDDPM
from MoudleCode.classification.dual_stream import MEDAL_Classifier, DualStreamLoss

def stage1_pretrain_backbone(backbone, train_loader, config, logger):
    """
    Stage 1: Pre-train backbone with SimMTM only (unsupervised)
    
    Args:
        backbone: MicroBiMambaBackbone model
        train_loader: DataLoader with X only (no labels needed for SimMTM)
        config: configuration object
        logger: logger
        
    Returns:
        backbone: pre-trained backbone
    """
    # Clear GPU cache before starting
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✓ GPU 缓存已清理")
    
    logger.info("="*70)
    logger.info("STAGE 1: Pre-training Backbone")
    logger.info("="*70)
    logger.info(f"目标: 训练Micro-Bi-Mamba骨干网络，学习流量特征表示")
    logger.info(f"训练轮数: {config.PRETRAIN_EPOCHS} epochs")
    
    # 显示实际使用的批次大小
    actual_batch_size = train_loader.batch_size
    logger.info(f"批次大小: {actual_batch_size}")
    
    logger.info(f"学习率: {config.PRETRAIN_LR}")
    
    # 检查是否启用实例对比学习
    use_instance_contrastive = getattr(config, 'USE_INSTANCE_CONTRASTIVE', False)
    contrastive_method = getattr(config, 'CONTRASTIVE_METHOD', 'infonce')
    if use_instance_contrastive:
        method = contrastive_method
        logger.info(f"优化目标: SimMTM + {str(method).upper()} (实例对比学习)")
        if str(method).lower() in ['infonce', 'nnclr']:
            logger.info(f"  - 温度: {config.INFONCE_TEMPERATURE}")
        logger.info(f"  - 权重: {config.INFONCE_LAMBDA}")
        
        # NNCLR 默认启用梯度累积（由 PRETRAIN_GRADIENT_ACCUMULATION_STEPS 控制）
        if str(method).lower() == 'nnclr':
            gradient_accumulation_steps = int(getattr(config, 'PRETRAIN_GRADIENT_ACCUMULATION_STEPS', 1))
            if gradient_accumulation_steps > 1:
                effective_batch_size = actual_batch_size * gradient_accumulation_steps
                logger.info(f"  - 梯度累积: {gradient_accumulation_steps} 步")
                logger.info(f"  - 有效批次: {actual_batch_size} × {gradient_accumulation_steps} = {effective_batch_size}")
    else:
        logger.info(f"优化目标: SimMTM (掩码重构)")
    logger.info("")
    logger.info("📥 输入数据路径:")
    logger.info(f"  ✓ 训练数据: {config.BENIGN_TRAIN} (正常), {config.MALICIOUS_TRAIN} (恶意)")
    logger.info("")
    
    backbone.train()
    backbone.to(config.DEVICE)
    
    # 创建增强器和损失函数
    if use_instance_contrastive:
        # 实例对比学习模式
        augmentation = DualViewAugmentation(config)
        instance_contrastive = InstanceContrastiveLearning(backbone, config).to(config.DEVICE)
        simmtm_loss_fn = SimMTMLoss(mask_rate=config.SIMMTM_MASK_RATE)
        hybrid_loss_fn = HybridPretrainingLoss(
            simmtm_loss=simmtm_loss_fn,
            instance_contrastive=instance_contrastive,
            lambda_infonce=config.INFONCE_LAMBDA
        )
        logger.info(f"✓ 混合损失函数初始化完成 (SimMTM + InfoNCE)")
        
        # 优化器包括projection head
        optimizer = optim.AdamW(
            list(backbone.parameters()) + list(instance_contrastive.projection_head.parameters()),
            lr=config.PRETRAIN_LR,
            weight_decay=config.PRETRAIN_WEIGHT_DECAY
        )
    else:
        # 原始SimMTM模式
        simmtm_loss_fn = SimMTMLoss(mask_rate=config.SIMMTM_MASK_RATE)
        logger.info(f"✓ 损失函数初始化完成 (SimMTM掩码率: {config.SIMMTM_MASK_RATE})")
        
        optimizer = optim.AdamW(
            backbone.parameters(),
            lr=config.PRETRAIN_LR,
            weight_decay=config.PRETRAIN_WEIGHT_DECAY
        )
    
    pretrain_min_lr = float(getattr(config, 'PRETRAIN_MIN_LR', 1e-5))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.PRETRAIN_EPOCHS, eta_min=pretrain_min_lr
    )
    logger.info(f"✓ 优化器和学习率调度器初始化完成")
    logger.info("")
    logger.info("开始训练...")
    logger.info("-"*70)
    
    # Training loop
    if use_instance_contrastive:
        history = {'loss': [], 'simmtm': [], 'infonce': []}
    else:
        history = {'loss': [], 'simmtm': []}
    
    use_early_stopping = bool(getattr(config, 'PRETRAIN_EARLY_STOPPING', True))
    es_warmup_epochs = int(getattr(config, 'PRETRAIN_ES_WARMUP_EPOCHS', 50))
    es_patience = int(getattr(config, 'PRETRAIN_ES_PATIENCE', 20))
    es_min_delta = float(getattr(config, 'PRETRAIN_ES_MIN_DELTA', 0.01))

    best_loss = float('inf')
    best_epoch = -1
    best_state = None
    no_improve = 0
    
    # Gradient accumulation (用于 NNCLR，保持有效批次不变)
    use_gradient_accumulation = (
        use_instance_contrastive and
        str(contrastive_method).lower() == 'nnclr' and
        int(getattr(config, 'PRETRAIN_GRADIENT_ACCUMULATION_STEPS', 1)) > 1
    )

    if use_gradient_accumulation:
        gradient_accumulation_steps = int(getattr(config, 'PRETRAIN_GRADIENT_ACCUMULATION_STEPS', 2))
    else:
        gradient_accumulation_steps = 1  # 禁用梯度累积

    if use_instance_contrastive:
        method_lower = str(contrastive_method).lower()
        temperature = float(getattr(config, 'INFONCE_TEMPERATURE', getattr(config, 'SUPCON_TEMPERATURE', 0.1)))
        lambda_infonce = float(getattr(config, 'INFONCE_LAMBDA', 1.0))
        effective_batch_size = int(actual_batch_size) * int(gradient_accumulation_steps)
        logger.info("[Stage 1] Contrastive config:")
        logger.info(f"  - method: {str(contrastive_method).upper()}")
        logger.info(f"  - temperature: {temperature}")
        logger.info(f"  - lambda_infonce: {lambda_infonce}")
        logger.info(f"  - per-step batch_size: {int(actual_batch_size)}")
        logger.info(f"  - gradient_accumulation_steps: {int(gradient_accumulation_steps)}")
        logger.info(f"  - effective_batch_size: {effective_batch_size}")
        if method_lower in ['infonce', 'simclr', 'nnclr']:
            approx_negatives = max(2 * int(actual_batch_size) - 2, 0)
            logger.info(f"  - negatives per anchor (within step): ~{approx_negatives}")
        if method_lower == 'nnclr':
            nnclr_queue_size = int(getattr(config, 'NNCLR_QUEUE_SIZE', 4096))
            logger.info(f"  - nnclr queue size: {nnclr_queue_size}")

    for epoch in range(config.PRETRAIN_EPOCHS):
        epoch_loss = 0.0
        epoch_simmtm = 0.0
        epoch_infonce = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # TensorDataset with single tensor returns the tensor directly
            # TensorDataset with multiple tensors returns a tuple
            if isinstance(batch_data, (list, tuple)):
                X_batch = batch_data[0]  # Get first element (X)
            else:
                X_batch = batch_data  # Already a single tensor
            X_batch = X_batch.to(config.DEVICE)
            
            # Only zero grad at the start of accumulation
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            if use_instance_contrastive:
                # 增强视图
                x_view1, x_view2 = augmentation(X_batch)
                
                # 混合损失
                loss, loss_dict = hybrid_loss_fn(
                    backbone=backbone,
                    x_original=X_batch,
                    x_view1=x_view1,
                    x_view2=x_view2,
                    epoch=epoch
                )
                
                epoch_simmtm += loss_dict['simmtm']
                epoch_infonce += loss_dict['infonce']
            else:
                # 原始SimMTM模式
                loss = simmtm_loss_fn(backbone, X_batch)
                epoch_simmtm += loss.item()
            
            # Scale loss by accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Only step optimizer after accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
            
            epoch_loss += loss.item() * gradient_accumulation_steps  # Restore original scale for logging
        
        scheduler.step()
        
        # Average losses
        n_batches = len(train_loader)
        epoch_loss /= n_batches
        epoch_simmtm /= n_batches
        
        history['loss'].append(epoch_loss)
        history['simmtm'].append(epoch_simmtm)
        
        if use_instance_contrastive:
            epoch_infonce /= n_batches
            history['infonce'].append(epoch_infonce)
        
        # 每个epoch都输出日志（便于监控）
        progress = (epoch + 1) / config.PRETRAIN_EPOCHS * 100
        if use_instance_contrastive:
            method_name = str(contrastive_method).upper()
            logger.info(f"[Stage 1] Epoch [{epoch+1}/{config.PRETRAIN_EPOCHS}] ({progress:.1f}%) | "
                       f"Loss: {epoch_loss:.4f} | "
                       f"SimMTM: {epoch_simmtm:.4f} | "
                       f"{method_name}: {epoch_infonce:.4f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.6f}")
        else:
            logger.info(f"[Stage 1] Epoch [{epoch+1}/{config.PRETRAIN_EPOCHS}] ({progress:.1f}%) | "
                       f"Loss: {epoch_loss:.4f} | "
                       f"SimMTM: {epoch_simmtm:.4f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.6f}")

        improved = (best_loss - epoch_loss) > es_min_delta
        if improved:
            best_loss = float(epoch_loss)
            best_epoch = int(epoch + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if use_early_stopping and (epoch + 1) >= es_warmup_epochs and no_improve >= es_patience:
            logger.info(
                f"[Stage 1] EarlyStopping triggered at epoch {epoch+1}: "
                f"best_loss={best_loss:.4f} (epoch {best_epoch}), "
                f"no_improve={no_improve}, min_delta={es_min_delta}"
            )
            break

    if best_state is not None:
        load_state_dict_shape_safe(backbone, best_state, logger, prefix="backbone(best)")
        backbone.to(config.DEVICE)
    
    logger.info("-"*70)
    logger.info("✓ Stage 1 完成: 骨干网络预训练完成")
    logger.info(f"  最终损失: {history['loss'][-1]:.4f}")
    if best_epoch > 0:
        logger.info(f"  最佳损失: {best_loss:.4f} (epoch {best_epoch})")
    logger.info(f"  训练了 {len(history['loss'])} 个epoch")
    logger.info("")
    
    # 生成backbone文件名：backbone_{method}_{dim}d_{n_samples}.pth
    # method: SimMTM 或 SimCLR (如果启用实例对比学习)
    # dim: 模型维度 (MODEL_DIM)
    n_samples = len(train_loader.dataset)
    model_dim = config.MODEL_DIM
    if use_instance_contrastive:
        method = getattr(config, 'CONTRASTIVE_METHOD', 'infonce')
        method_name = f"SimMTM_{str(method).upper()}"
    else:
        method_name = "SimMTM"
    
    backbone_filename = f"backbone_{method_name}_{model_dim}d_{n_samples}.pth"
    
    logger.info("📁 输出文件路径:")
    # Save backbone to feature_extraction module directory
    backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", backbone_filename)
    torch.save(backbone.state_dict(), backbone_path)
    logger.info(f"  ✓ 骨干网络模型: {backbone_path}")
    logger.info(f"    (命名格式: backbone_{{方法}}_{{维度}}d_{{样本数}}.pth)")
    
    # 同时保存一个默认名称的副本（向后兼容）
    default_backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    torch.save(backbone.state_dict(), default_backbone_path)
    logger.info(f"  ✓ 默认副本: {default_backbone_path}")
    logger.info(f"    (用于向后兼容)")
    
    return backbone, history


def stage2_label_correction_and_augmentation(backbone, X_train, y_train_noisy, y_train_clean, config, logger, stage2_mode='standard'):
    """
    Stage 2: Label correction and data augmentation
    
    Args:
        backbone: Pre-trained frozen backbone
        X_train: (N, L, D) training sequences
        y_train_noisy: (N,) noisy labels
        y_train_clean: (N,) clean labels (for evaluation only)
        config: configuration object
        logger: logger
        
    Returns:
        X_augmented: augmented dataset
        y_augmented: augmented labels
        correction_stats: statistics about correction
    """
    logger.info("")
    logger.info("="*70)
    logger.info("STAGE 2: Label Correction & Data Augmentation")
    logger.info("="*70)
    logger.info(f"目标: 矫正标签噪声并生成增强样本")
    logger.info(f"步骤: 1) 特征提取 2) Hybrid Court标签矫正 3) TabDDPM数据增强")
    logger.info("")
    logger.info("📥 输入数据路径:")
    logger.info(f"  ✓ 训练数据: {config.BENIGN_TRAIN} (正常), {config.MALICIOUS_TRAIN} (恶意)")
    backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    logger.info(f"  ✓ 骨干网络模型: {backbone_path}")
    logger.info("")
    
    # Freeze backbone and extract features
    backbone.to(config.DEVICE)  # 确保 backbone 在正确的设备上
    backbone.freeze()
    backbone.eval()
    logger.info("✓ 骨干网络已冻结，开始特征提取...")
    
    logger.info(f"正在从 {len(X_train)} 个训练样本中提取特征...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
        
        # Extract features in batches
        features_list = []
        batch_size = 64
        total_batches = (len(X_tensor) + batch_size - 1) // batch_size
        
        for i in range(0, len(X_tensor), batch_size):
            batch_idx = i // batch_size + 1
            X_batch = X_tensor[i:i+batch_size]
            z_batch = backbone(X_batch, return_sequence=False)
            features_list.append(z_batch.cpu().numpy())
            
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                progress = batch_idx / total_batches * 100
                logger.info(f"  特征提取进度: {batch_idx}/{total_batches} batches ({progress:.1f}%)")
        
        features = np.concatenate(features_list, axis=0)
    
    logger.info(f"✓ 特征提取完成: {features.shape} (样本数×特征维度)")
    logger.info("")
    
    # 生成特征分布可视化（Stage 2特征提取后）
    logger.info("📊 生成特征分布可视化...")
    feature_dist_path = os.path.join(config.LABEL_CORRECTION_DIR, "figures", "feature_distribution_stage2.png")
    plot_feature_space(features, y_train_clean, feature_dist_path,
                      title="Stage 2: Feature Distribution (After Backbone Extraction)", method='tsne')
    logger.info(f"  ✓ 特征分布图 (t-SNE): {feature_dist_path}")
    logger.info("")

    logger.info("="*70)
    logger.info("📏 特征可分性评估 (Feature Separability)")
    logger.info("="*70)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            features, y_train_clean,
            test_size=0.2,
            stratify=y_train_clean,
            random_state=config.SEED
        )
        sep_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        sep_clf.fit(X_tr, y_tr)
        te_proba = sep_clf.predict_proba(X_te)[:, 1]
        te_auc = roc_auc_score(y_te, te_proba)
        te_pred = (te_proba >= 0.5).astype(int)
        te_f1 = f1_score(y_te, te_pred, pos_label=1, zero_division=0)
        sil = silhouette_score(features, y_train_clean) if len(np.unique(y_train_clean)) > 1 else np.nan
        logger.info(f"  LogisticRegression ROC-AUC: {te_auc:.4f}")
        logger.info(f"  LogisticRegression F1@0.5:  {te_f1:.4f}")
        logger.info(f"  Silhouette Score:          {sil:.4f}")
    except Exception as e:
        logger.warning(f"⚠ 特征可分性评估失败: {e}")
    logger.info("")
    logger.info("📁 输出文件路径:")
    # Save extracted features
    features_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "train_features.npy")
    np.save(features_path, features)
    logger.info(f"  ✓ 提取的特征: {features_path}")
    
    # Visualize feature space before correction
    save_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "figures", "feature_space_before_correction.png")
    plot_feature_space(features, y_train_clean, save_path, 
                      title="Feature Space (Ground Truth Labels)", method='tsne')
    logger.info(f"  ✓ 特征空间可视化: {save_path}")
    
    # Label correction with Hybrid Court
    logger.info("")
    logger.info("步骤 2.1: Hybrid Court 标签噪声矫正")
    logger.info(f"  输入: {len(y_train_noisy)} 个样本，噪声率: {config.LABEL_NOISE_RATE*100:.0f}%")
    logger.info("  方法: CL (置信学习) + MADE (密度估计) + KNN (语义投票)")
    logger.info("  开始矫正...")
    
    hybrid_court = HybridCourt(config)

    if stage2_mode == 'clean_augment_only':
        suspected_noise, pred_labels, pred_probs = hybrid_court.cl.fit_predict(features, y_train_clean)
        hybrid_court.made.fit(features, device=config.DEVICE)
        is_dense, density_scores = hybrid_court.made.predict_density(features, device=config.DEVICE)
        hybrid_court.knn.fit(features)
        neighbor_labels, neighbor_consistency = hybrid_court.knn.predict_semantic_label(features, y_train_clean)
        y_corrected = y_train_clean.copy()
        action_mask = np.zeros(len(y_train_clean), dtype=int)
        confidence = pred_probs.max(axis=1)
        correction_weight = np.ones(len(y_train_clean), dtype=np.float32)
        cl_confidence = pred_probs.max(axis=1)
    else:
        y_corrected, action_mask, confidence, correction_weight, density_scores, neighbor_consistency, pred_probs = hybrid_court.correct_labels(
            features, y_train_noisy, device=config.DEVICE
        )
        cl_confidence = pred_probs.max(axis=1)
    
    logger.info("✓ 标签矫正完成")
    
    # Save correction results
    correction_results_path = os.path.join(config.LABEL_CORRECTION_DIR, "models", "correction_results.npz")
    np.savez(correction_results_path,
             y_noisy=y_train_noisy if stage2_mode != 'clean_augment_only' else y_train_clean,
             y_corrected=y_corrected,
             action_mask=action_mask,
             confidence=confidence,
             correction_weight=correction_weight,
             density_scores=density_scores,
             neighbor_consistency=neighbor_consistency,
             pred_probs=pred_probs)
    logger.info("")
    logger.info("📁 输出文件路径:")
    logger.info(f"  ✓ 标签矫正结果: {correction_results_path}")
    
    # Visualize correction results
    save_path = os.path.join(config.LABEL_CORRECTION_DIR, "figures", "noise_correction_comparison.png")
    plot_noise_correction_comparison(y_train_clean, y_train_noisy, y_corrected, action_mask, save_path)
    logger.info(f"  ✓ 标签矫正对比图: {save_path}")
    
    # Calculate correction accuracy
    keep_mask = action_mask != 2  # Exclude dropped samples
    correction_accuracy = (y_corrected[keep_mask] == y_train_clean[keep_mask]).mean()
    logger.info(f"\nLabel Correction Accuracy: {correction_accuracy*100:.2f}%")
    
    correction_stats = {
        'accuracy': correction_accuracy,
        'n_keep': (action_mask == 0).sum(),
        'n_flip': (action_mask == 1).sum(),
        'n_drop': (action_mask == 2).sum(),
        'n_reweight': (action_mask == 3).sum()
    }
    
    # ========================
    # 生成详细的标签矫正分析报告
    # ========================
    logger.info("")
    logger.info("="*70)
    logger.info("📊 生成标签矫正分析报告")
    logger.info("="*70)
    
    from datetime import datetime
    
    # 准备分析数据
    n_total = len(y_train_noisy if stage2_mode != 'clean_augment_only' else y_train_clean)
    n_keep = correction_stats['n_keep']
    n_flip = correction_stats['n_flip']
    n_drop = correction_stats['n_drop']
    n_reweight = correction_stats['n_reweight']
    
    # 计算各类别的矫正情况
    y_noisy_input = y_train_noisy if stage2_mode != 'clean_augment_only' else y_train_clean
    
    # 真实噪声样本（ground truth）
    true_noise_mask = (y_noisy_input != y_train_clean)
    n_true_noise = true_noise_mask.sum()
    
    # 被识别为噪声的样本（预测）
    predicted_noise_mask = (action_mask == 1) | (action_mask == 2)
    n_predicted_noise = predicted_noise_mask.sum()
    
    # 正确识别的噪声（True Positive）
    tp_noise = (true_noise_mask & predicted_noise_mask).sum()
    # 错误识别的噪声（False Positive）
    fp_noise = (~true_noise_mask & predicted_noise_mask).sum()
    # 漏检的噪声（False Negative）
    fn_noise = (true_noise_mask & ~predicted_noise_mask).sum()
    # 正确保留的干净样本（True Negative）
    tn_noise = (~true_noise_mask & ~predicted_noise_mask).sum()
    
    # 计算噪声检测指标
    noise_precision = tp_noise / n_predicted_noise if n_predicted_noise > 0 else 0
    noise_recall = tp_noise / n_true_noise if n_true_noise > 0 else 0
    noise_f1 = 2 * noise_precision * noise_recall / (noise_precision + noise_recall) if (noise_precision + noise_recall) > 0 else 0
    
    # 矫正后的准确率
    final_accuracy = (y_corrected[keep_mask] == y_train_clean[keep_mask]).mean()
    
    # 各动作的准确率
    keep_samples = (action_mask == 0)
    flip_samples = (action_mask == 1)
    reweight_samples = (action_mask == 3)
    
    keep_accuracy = (y_corrected[keep_samples] == y_train_clean[keep_samples]).mean() if keep_samples.sum() > 0 else 0
    flip_accuracy = (y_corrected[flip_samples] == y_train_clean[flip_samples]).mean() if flip_samples.sum() > 0 else 0
    reweight_accuracy = (y_corrected[reweight_samples] == y_train_clean[reweight_samples]).mean() if reweight_samples.sum() > 0 else 0
    
    # 生成Markdown报告
    report_path = os.path.join(config.LABEL_CORRECTION_DIR, "models", "correction_analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 标签噪声矫正分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**噪声率**: {config.LABEL_NOISE_RATE*100:.1f}%\n\n")
        f.write(f"**矫正方法**: Hybrid Court (CL + MADE + KNN)\n\n")
        
        f.write("---\n\n")
        f.write("## 1. 数据集概览\n\n")
        f.write(f"- **总样本数**: {n_total}\n")
        f.write(f"- **真实噪声样本数**: {n_true_noise} ({n_true_noise/n_total*100:.2f}%)\n")
        f.write(f"- **干净样本数**: {n_total - n_true_noise} ({(n_total-n_true_noise)/n_total*100:.2f}%)\n\n")
        
        f.write("### 类别分布\n\n")
        f.write("| 标签类型 | 正常样本 | 恶意样本 | 总计 |\n")
        f.write("|---------|---------|---------|------|\n")
        f.write(f"| 真实标签 | {(y_train_clean==0).sum()} | {(y_train_clean==1).sum()} | {len(y_train_clean)} |\n")
        f.write(f"| 噪声标签 | {(y_noisy_input==0).sum()} | {(y_noisy_input==1).sum()} | {len(y_noisy_input)} |\n")
        f.write(f"| 矫正后标签 | {(y_corrected==0).sum()} | {(y_corrected==1).sum()} | {len(y_corrected)} |\n\n")
        
        f.write("---\n\n")
        f.write("## 2. 矫正动作统计\n\n")
        f.write("| 动作 | 样本数 | 占比 | 准确率 | 说明 |\n")
        f.write("|------|--------|------|--------|------|\n")
        f.write(f"| **保持 (Keep)** | {n_keep} | {n_keep/n_total*100:.2f}% | {keep_accuracy*100:.2f}% | 标签可信，保持不变 |\n")
        f.write(f"| **翻转 (Flip)** | {n_flip} | {n_flip/n_total*100:.2f}% | {flip_accuracy*100:.2f}% | 标签错误，翻转标签 |\n")
        f.write(f"| **丢弃 (Drop)** | {n_drop} | {n_drop/n_total*100:.2f}% | - | 不确定性高，丢弃样本 |\n")
        f.write(f"| **重加权 (Reweight)** | {n_reweight} | {n_reweight/n_total*100:.2f}% | {reweight_accuracy*100:.2f}% | 可疑但保留，降低权重 |\n")
        f.write(f"| **总计** | {n_total} | 100.00% | - | - |\n\n")
        
        f.write("### 动作分布可视化\n\n")
        f.write("```\n")
        f.write(f"保持:   {'█' * int(n_keep/n_total*50)} {n_keep/n_total*100:.1f}%\n")
        f.write(f"翻转:   {'█' * int(n_flip/n_total*50)} {n_flip/n_total*100:.1f}%\n")
        f.write(f"丢弃:   {'█' * int(n_drop/n_total*50)} {n_drop/n_total*100:.1f}%\n")
        f.write(f"重加权: {'█' * int(n_reweight/n_total*50)} {n_reweight/n_total*100:.1f}%\n")
        f.write("```\n\n")
        
        f.write("---\n\n")
        f.write("## 3. 噪声检测性能\n\n")
        f.write("### 混淆矩阵（噪声检测）\n\n")
        f.write("| | 预测为干净 | 预测为噪声 |\n")
        f.write("|---|-----------|----------|\n")
        f.write(f"| **真实干净** | {tn_noise} (TN) | {fp_noise} (FP) |\n")
        f.write(f"| **真实噪声** | {fn_noise} (FN) | {tp_noise} (TP) |\n\n")
        
        f.write("### 噪声检测指标\n\n")
        f.write(f"- **Precision (精确率)**: {noise_precision*100:.2f}% - 识别为噪声的样本中，真正是噪声的比例\n")
        f.write(f"- **Recall (召回率)**: {noise_recall*100:.2f}% - 所有真实噪声中，被正确识别的比例\n")
        f.write(f"- **F1-Score**: {noise_f1*100:.2f}% - 精确率和召回率的调和平均\n\n")
        
        f.write("### 性能评估\n\n")
        if noise_f1 >= 0.8:
            f.write("✅ **优秀** - 噪声检测性能很好，大部分噪声被正确识别\n\n")
        elif noise_f1 >= 0.6:
            f.write("⚠️ **良好** - 噪声检测性能尚可，但仍有改进空间\n\n")
        else:
            f.write("❌ **需改进** - 噪声检测性能较差，建议调整参数或方法\n\n")
        
        f.write("---\n\n")
        f.write("## 4. 矫正效果评估\n\n")
        f.write(f"- **矫正前准确率**: {(y_noisy_input == y_train_clean).mean()*100:.2f}%\n")
        f.write(f"- **矫正后准确率**: {final_accuracy*100:.2f}%\n")
        f.write(f"- **准确率提升**: {(final_accuracy - (y_noisy_input == y_train_clean).mean())*100:.2f}%\n\n")
        
        improvement = final_accuracy - (y_noisy_input == y_train_clean).mean()
        if improvement > 0.1:
            f.write("✅ **显著改善** - 标签矫正显著提升了数据质量\n\n")
        elif improvement > 0.05:
            f.write("✅ **有效改善** - 标签矫正有效提升了数据质量\n\n")
        elif improvement > 0:
            f.write("⚠️ **轻微改善** - 标签矫正略微提升了数据质量\n\n")
        else:
            f.write("❌ **无改善** - 标签矫正未能提升数据质量，建议检查参数\n\n")
        
        f.write("---\n\n")
        f.write("## 5. 各组件贡献分析\n\n")
        
        # CL组件分析
        cl_suspected = (pred_probs.max(axis=1) < 0.5).sum()
        f.write(f"### CL (置信学习)\n\n")
        f.write(f"- **识别可疑样本数**: {cl_suspected}\n")
        f.write(f"- **平均置信度**: {pred_probs.max(axis=1).mean():.4f}\n")
        f.write(f"- **低置信度样本 (<0.5)**: {cl_suspected} ({cl_suspected/n_total*100:.2f}%)\n\n")
        
        # MADE组件分析
        dense_samples = (density_scores > 0.5).sum()
        f.write(f"### MADE (密度估计)\n\n")
        f.write(f"- **高密度样本数**: {dense_samples} ({dense_samples/n_total*100:.2f}%)\n")
        f.write(f"- **平均密度分数**: {density_scores.mean():.4f}\n")
        f.write(f"- **低密度样本 (<0.3)**: {(density_scores < 0.3).sum()} ({(density_scores < 0.3).sum()/n_total*100:.2f}%)\n\n")
        
        # KNN组件分析
        high_consistency = (neighbor_consistency > 0.7).sum()
        f.write(f"### KNN (语义投票)\n\n")
        f.write(f"- **高一致性样本数**: {high_consistency} ({high_consistency/n_total*100:.2f}%)\n")
        f.write(f"- **平均邻居一致性**: {neighbor_consistency.mean():.4f}\n")
        f.write(f"- **低一致性样本 (<0.5)**: {(neighbor_consistency < 0.5).sum()} ({(neighbor_consistency < 0.5).sum()/n_total*100:.2f}%)\n\n")
        
        f.write("---\n\n")
        f.write("## 6. 样本权重分布\n\n")
        f.write(f"- **平均权重**: {correction_weight.mean():.4f}\n")
        f.write(f"- **权重标准差**: {correction_weight.std():.4f}\n")
        f.write(f"- **最小权重**: {correction_weight.min():.4f}\n")
        f.write(f"- **最大权重**: {correction_weight.max():.4f}\n\n")
        
        f.write("### 权重分布统计\n\n")
        f.write("| 权重范围 | 样本数 | 占比 |\n")
        f.write("|---------|--------|------|\n")
        f.write(f"| [0.0, 0.2) | {((correction_weight >= 0.0) & (correction_weight < 0.2)).sum()} | {((correction_weight >= 0.0) & (correction_weight < 0.2)).sum()/n_total*100:.2f}% |\n")
        f.write(f"| [0.2, 0.4) | {((correction_weight >= 0.2) & (correction_weight < 0.4)).sum()} | {((correction_weight >= 0.2) & (correction_weight < 0.4)).sum()/n_total*100:.2f}% |\n")
        f.write(f"| [0.4, 0.6) | {((correction_weight >= 0.4) & (correction_weight < 0.6)).sum()} | {((correction_weight >= 0.4) & (correction_weight < 0.6)).sum()/n_total*100:.2f}% |\n")
        f.write(f"| [0.6, 0.8) | {((correction_weight >= 0.6) & (correction_weight < 0.8)).sum()} | {((correction_weight >= 0.6) & (correction_weight < 0.8)).sum()/n_total*100:.2f}% |\n")
        f.write(f"| [0.8, 1.0] | {((correction_weight >= 0.8) & (correction_weight <= 1.0)).sum()} | {((correction_weight >= 0.8) & (correction_weight <= 1.0)).sum()/n_total*100:.2f}% |\n\n")
        
        f.write("---\n\n")
        f.write("## 7. 建议与总结\n\n")
        
        if final_accuracy >= 0.95:
            f.write("### ✅ 矫正效果优秀\n\n")
            f.write("- 标签矫正效果很好，数据质量高\n")
            f.write("- 可以直接用于后续训练\n\n")
        elif final_accuracy >= 0.85:
            f.write("### ✅ 矫正效果良好\n\n")
            f.write("- 标签矫正效果较好，大部分噪声被处理\n")
            f.write("- 建议关注低置信度样本\n\n")
        else:
            f.write("### ⚠️ 矫正效果一般\n\n")
            f.write("- 标签矫正效果有限，仍有较多噪声\n")
            f.write("- 建议调整以下参数：\n")
            f.write("  - 增加CL的置信度阈值\n")
            f.write("  - 调整MADE的密度阈值\n")
            f.write("  - 增加KNN的邻居数量\n\n")
        
        f.write("### 关键发现\n\n")
        if noise_precision > 0.8 and noise_recall < 0.6:
            f.write("- **高精确率，低召回率**: 矫正策略保守，漏检了部分噪声\n")
            f.write("  - 建议: 降低置信度阈值，增加噪声检测的敏感度\n\n")
        elif noise_precision < 0.6 and noise_recall > 0.8:
            f.write("- **低精确率，高召回率**: 矫正策略激进，误判了部分干净样本\n")
            f.write("  - 建议: 提高置信度阈值，减少误判\n\n")
        elif noise_precision > 0.7 and noise_recall > 0.7:
            f.write("- **平衡的检测性能**: 精确率和召回率都较高，矫正策略合理\n\n")
        
        if n_drop > n_total * 0.2:
            f.write("- **丢弃样本过多**: 超过20%的样本被丢弃\n")
            f.write("  - 建议: 检查数据质量或调整丢弃阈值\n\n")
        
        f.write("---\n\n")
        f.write("## 8. 输出文件\n\n")
        f.write(f"- **矫正结果**: `{correction_results_path}`\n")
        f.write(f"- **对比图**: `{os.path.join(config.LABEL_CORRECTION_DIR, 'figures', 'noise_correction_comparison.png')}`\n")
        f.write(f"- **特征空间图**: `{os.path.join(config.FEATURE_EXTRACTION_DIR, 'figures', 'feature_space_before_correction.png')}`\n")
        f.write(f"- **本报告**: `{report_path}`\n\n")
        
        f.write("---\n\n")
        f.write("*报告由MEDAL-Lite自动生成*\n")
    
    logger.info(f"  ✓ 标签矫正分析报告: {report_path}")
    logger.info("")
    
    # 使用全部干净样本进行训练与增强（取消验证集）
    X_clean = X_train[keep_mask]
    y_clean = y_corrected[keep_mask]
    weights_clean = correction_weight[keep_mask]
    action_clean = action_mask[keep_mask]
    density_clean = density_scores[keep_mask]
    cl_conf_clean = cl_confidence[keep_mask]
    knn_conf_clean = neighbor_consistency[keep_mask]
    
    # Data augmentation with TabDDPM
    logger.info("")
    logger.info("步骤 2.2: TabDDPM 数据增强模型训练")
    logger.info(f"  目标: 学习流量特征分布，生成符合协议逻辑的合成样本")
    ddpm_epochs = int(getattr(config, 'DDPM_EPOCHS', 100))
    logger.info(f"  训练轮数: {ddpm_epochs} epochs")
    logger.info(f"  引导策略: 恶意样本(w={config.GUIDANCE_MALICIOUS}), 良性样本(w={config.GUIDANCE_BENIGN})")
    logger.info("  开始训练...")
    
    tabddpm = TabDDPM(config).to(config.DEVICE)

    # Fit scaler on packet-level features (exclude zero-padding packets)
    X_packets_all = X_clean.reshape(-1, X_clean.shape[-1])
    vm_idx = getattr(config, 'VALID_MASK_INDEX', None)
    if vm_idx is not None and int(vm_idx) >= 0 and int(vm_idx) < X_packets_all.shape[1]:
        packet_mask = X_packets_all[:, int(vm_idx)] > 0.5
    else:
        packet_mask = np.any(X_packets_all != 0.0, axis=1)
    X_packets_valid = X_packets_all[packet_mask]
    if X_packets_valid.shape[0] == 0:
        X_packets_valid = X_packets_all
        packet_mask = None
    tabddpm.fit_scaler(X_packets_valid)
    
    # Train TabDDPM
    optimizer_ddpm = optim.AdamW(tabddpm.parameters(), lr=1e-4)
    n_epochs_ddpm = ddpm_epochs
	
    ddpm_use_early_stopping = bool(getattr(config, 'DDPM_EARLY_STOPPING', True))
    ddpm_es_warmup_epochs = int(getattr(config, 'DDPM_ES_WARMUP_EPOCHS', 20))
    ddpm_es_patience = int(getattr(config, 'DDPM_ES_PATIENCE', 30))
    ddpm_es_min_delta = float(getattr(config, 'DDPM_ES_MIN_DELTA', 0.001))
	
    ddpm_best_loss = float('inf')
    ddpm_best_epoch = -1
    ddpm_best_state = None
    ddpm_no_improve = 0
    
    # 使用 packet-level 数据训练 TabDDPM（避免只看第一个包导致分布偏移）
    # 同时过滤掉 padding=0 的无效包，避免模型学到大量“全零包”导致 Length/IAT/BurstSize 分布漂移
    X_packets = X_packets_valid  # (n_valid, D)
    y_packets_all = np.repeat(y_clean, X_clean.shape[1])    # (N*L,)
    y_packets = y_packets_all[packet_mask] if packet_mask is not None else y_packets_all
    dataset_ddpm = TensorDataset(
        torch.FloatTensor(X_packets),
        torch.LongTensor(y_packets)
    )
    loader_ddpm = DataLoader(dataset_ddpm, batch_size=2048, shuffle=True)
    
    tabddpm.train()
    for epoch in range(n_epochs_ddpm):
        epoch_loss = 0.0
        for X_batch, y_batch in loader_ddpm:
            x_0 = X_batch.to(config.DEVICE)  # (B, D)
            y_batch = y_batch.to(config.DEVICE)
            
            optimizer_ddpm.zero_grad()
            loss = tabddpm.compute_loss(
                x_0, y_batch,
                mask_prob=config.MASK_PROBABILITY,
                mask_lambda=config.MASK_LAMBDA,
                p_uncond=0.2  # Classifier-free guidance training
            )
            loss.backward()
            optimizer_ddpm.step()
            
            epoch_loss += loss.item()
        
        # 每个epoch输出
        avg_loss = epoch_loss / len(loader_ddpm)
        progress = (epoch + 1) / n_epochs_ddpm * 100
        logger.info(f"[TabDDPM] Epoch [{epoch+1}/{n_epochs_ddpm}] ({progress:.1f}%) | Loss: {avg_loss:.4f}")
	
        improved = (ddpm_best_loss - float(avg_loss)) > ddpm_es_min_delta
        if improved:
            ddpm_best_loss = float(avg_loss)
            ddpm_best_epoch = int(epoch + 1)
            ddpm_best_state = {k: v.detach().cpu().clone() for k, v in tabddpm.state_dict().items()}
            ddpm_no_improve = 0
        else:
            ddpm_no_improve += 1
	
        if ddpm_use_early_stopping and (epoch + 1) >= ddpm_es_warmup_epochs and ddpm_no_improve >= ddpm_es_patience:
            logger.info(
                f"[TabDDPM] EarlyStopping triggered at epoch {epoch+1}: "
                f"best_loss={ddpm_best_loss:.4f} (epoch {ddpm_best_epoch}), "
                f"no_improve={ddpm_no_improve}, min_delta={ddpm_es_min_delta}"
            )
            break

    if ddpm_best_state is not None:
        tabddpm.load_state_dict(ddpm_best_state)
        tabddpm.to(config.DEVICE)
    
    # Generate augmented samples (增强全部干净样本)
    logger.info("-"*70)
    logger.info("✓ TabDDPM 训练完成")
    if ddpm_best_epoch > 0:
        logger.info(f"  最佳TabDDPM损失: {ddpm_best_loss:.4f} (epoch {ddpm_best_epoch})")
    logger.info("")
    logger.info("步骤 2.3: 生成增强样本")
    logger.info(f"  训练集样本数: {len(X_clean)}")
    multipliers = np.ceil(weights_clean.astype(np.float32) * 10.0).astype(int)
    multipliers = np.maximum(multipliers, 2)
    expected_synthetic = int(multipliers.sum())
    logger.info("  增强策略: per-sample 倍数 = max(2, ceil(sample_weight*10))")
    logger.info(f"  倍数统计: min={multipliers.min()}, max={multipliers.max()}, mean={multipliers.mean():.2f}")
    logger.info(f"  预期生成: ~{expected_synthetic} 个合成样本")
    logger.info("  开始生成...")
    
    X_augmented, y_augmented, sample_weights = tabddpm.augment_dataset(
        X_clean, y_clean, action_clean, weights_clean,
        density_clean, cl_conf_clean, knn_conf_clean,
        augmentation_ratio=config.AUGMENTATION_RATIO_MIN
    )
    
    n_train_original = len(X_clean)
    n_synthetic = len(X_augmented) - n_train_original
    
    logger.info(f"✓ 数据增强完成: 从 {n_train_original} 个训练样本增强到 {len(X_augmented)} 个样本")
    logger.info(f"  原始训练数据: {n_train_original} 个样本")
    logger.info(f"  合成数据: {n_synthetic} 个样本")
    logger.info("")
    
    # ========================
    # 步骤 2.4: 生成质量诊断
    # ========================
    logger.info("步骤 2.4: 生成质量诊断 (Generation Quality Assessment)")
    logger.info("-"*70)
    logger.info("评估维度: 1) Fidelity (真实性) 2) Protocol Validity (协议有效性)")
    logger.info("")
    
    # 分离原始数据和合成数据
    X_train_original = X_augmented[:n_train_original]
    X_train_synthetic = X_augmented[n_train_original:]
    y_train_original = y_augmented[:n_train_original]
    y_train_synthetic = y_augmented[n_train_original:]
    
    # 1. Fidelity: 特征分布对比（均值/方差）
    logger.info("1️⃣  Fidelity 检查: 特征分布对比")
    feature_names = ['Length', 'Log-IAT', 'Direction', 'BurstSize', 'CumulativeLen', 'ValidMask']
    logger.info(f"{'特征':<12} | {'真实均值':<12} | {'合成均值':<12} | {'差异%':<10} | {'真实标准差':<12} | {'合成标准差':<12}")
    logger.info("-"*85)

    if vm_idx is not None and int(vm_idx) >= 0 and int(vm_idx) < X_train_original.shape[-1]:
        pad_mask_real = X_train_original[:, :, int(vm_idx)] > 0.5
        pad_mask_syn = X_train_synthetic[:, :, int(vm_idx)] > 0.5
    else:
        pad_mask_real = np.any(X_train_original != 0.0, axis=-1)
        pad_mask_syn = np.any(X_train_synthetic != 0.0, axis=-1)

    diff_pcts = []
    for i, name in enumerate(feature_names):
        real_data = X_train_original[:, :, i][pad_mask_real]
        syn_data = X_train_synthetic[:, :, i][pad_mask_syn]
        
        real_mean = real_data.mean() if real_data.size > 0 else np.nan
        syn_mean = syn_data.mean() if syn_data.size > 0 else np.nan
        real_std = real_data.std() if real_data.size > 0 else np.nan
        syn_std = syn_data.std() if syn_data.size > 0 else np.nan
        
        # 计算相对差异百分比
        if np.isfinite(real_mean) and np.isfinite(syn_mean) and np.isfinite(real_std):
            denom = max(abs(real_mean), float(real_std), 1e-8)
            diff_pct = abs(real_mean - syn_mean) / denom * 100
        else:
            diff_pct = np.nan
        diff_pcts.append(float(diff_pct))
        
        # 判断质量（差异<10%为优秀，<20%为良好，>20%为需关注）
        quality_marker = "✓" if diff_pct < 10 else ("⚠" if diff_pct < 20 else "❌")
        
        logger.info(f"{name:<12} | {real_mean:>11.4f} | {syn_mean:>11.4f} | {diff_pct:>8.2f}% {quality_marker} | "
                   f"{real_std:>11.4f} | {syn_std:>11.4f}")
    
    logger.info("")


    # 1.1 Quantiles: P50/P90/P99 (helps diagnose tail drift, esp. for skewed features like BurstSize)
    logger.info("1️⃣ 1️⃣  Quantile 检查: P50/P90/P99 (真实 vs 合成)")
    logger.info(f"{'特征':<12} | {'真实P50':<10} | {'合成P50':<10} | {'真实P90':<10} | {'合成P90':<10} | {'真实P99':<10} | {'合成P99':<10}")
    logger.info("-"*92)
    for i, name in enumerate(feature_names):
        real_data = X_train_original[:, :, i][pad_mask_real]
        syn_data = X_train_synthetic[:, :, i][pad_mask_syn]

        if real_data.size > 0:
            real_p50, real_p90, real_p99 = np.percentile(real_data, [50, 90, 99])
        else:
            real_p50, real_p90, real_p99 = np.nan, np.nan, np.nan

        if syn_data.size > 0:
            syn_p50, syn_p90, syn_p99 = np.percentile(syn_data, [50, 90, 99])
        else:
            syn_p50, syn_p90, syn_p99 = np.nan, np.nan, np.nan

        logger.info(
            f"{name:<12} | {real_p50:>9.4f} | {syn_p50:>9.4f} | {real_p90:>9.4f} | {syn_p90:>9.4f} | {real_p99:>9.4f} | {syn_p99:>9.4f}"
        )

    logger.info("")

    max_diff_pct = float(np.nanmax(diff_pcts)) if len(diff_pcts) > 0 else float('nan')
    if np.isnan(max_diff_pct):
        fidelity_level = '未知'
    elif max_diff_pct < 10:
        fidelity_level = '优秀'
    elif max_diff_pct < 20:
        fidelity_level = '良好'
    else:
        fidelity_level = '需关注'
    
    # 2. Protocol Validity: 协议约束检查
    logger.info("2️⃣  Protocol Validity 检查: 物理约束验证")
    logger.info("检查项: 负数包长、负数突发大小、异常IAT、异常方向、异常累积长度/掩码")
    logger.info("")
    
    # 检查各种违规情况
    invalid_length = (X_train_synthetic[:, :, 0] < 0).sum()
    invalid_iat = (np.isnan(X_train_synthetic[:, :, 1]) | np.isinf(X_train_synthetic[:, :, 1]) | (X_train_synthetic[:, :, 1] < 0)).sum()
    invalid_direction = ((X_train_synthetic[:, :, 2] < -1) | (X_train_synthetic[:, :, 2] > 1)).sum()
    invalid_burst = (X_train_synthetic[:, :, 3] < 0).sum()
    invalid_cum = ((X_train_synthetic[:, :, 4] < 0) | (X_train_synthetic[:, :, 4] > 1)).sum()
    invalid_mask = ((X_train_synthetic[:, :, 5] < 0) | (X_train_synthetic[:, :, 5] > 1)).sum()
    
    total_synthetic_values = X_train_synthetic.size
    invalid_total = invalid_length + invalid_burst + invalid_iat + invalid_direction + invalid_cum + invalid_mask
    validity_rate = (1 - invalid_total / total_synthetic_values) * 100
    
    logger.info(f"  ❌ 负数包长 (Length < 0):     {invalid_length:>6} 个值")
    logger.info(f"  ❌ 负数突发大小 (BurstSize < 0): {invalid_burst:>6} 个值")
    logger.info(f"  ❌ 异常IAT (NaN/Inf):         {invalid_iat:>6} 个值")
    logger.info(f"  ❌ 异常方向 (Direction ∉[-1,1]): {invalid_direction:>6} 个值")
    logger.info(f"  ❌ 异常累积长度 (CumulativeLen ∉[0,1]): {invalid_cum:>6} 个值")
    logger.info(f"  ❌ 异常掩码 (ValidMask ∉[0,1]): {invalid_mask:>6} 个值")
    logger.info(f"  ✓ 总有效率: {validity_rate:.2f}% ({total_synthetic_values - invalid_total}/{total_synthetic_values})")
    logger.info("")
    
    # 3. Class-wise 分布检查
    logger.info("3️⃣  Class-wise 分布检查: 类别平衡性")
    logger.info(f"  原始数据 - 正常: {(y_train_original==0).sum()}, 恶意: {(y_train_original==1).sum()}")
    logger.info(f"  合成数据 - 正常: {(y_train_synthetic==0).sum()}, 恶意: {(y_train_synthetic==1).sum()}")
    logger.info(f"  增强后   - 正常: {(y_augmented==0).sum()}, 恶意: {(y_augmented==1).sum()}")
    logger.info("")
    
    # 4. 结构感知检查：依赖特征的协方差
    logger.info("4️⃣  Structure-Aware 检查: 依赖特征协方差")
    logger.info("检查 Length-IAT-BurstSize 之间的相关性是否保持")
    
    # 提取依赖特征（索引 0, 1, 4）
    dep_indices = [0, 1, 3]  # Length, Log-IAT, BurstSize
    real_dep = X_train_original[:, :, dep_indices][pad_mask_real].reshape(-1, 3)
    syn_dep = X_train_synthetic[:, :, dep_indices][pad_mask_syn].reshape(-1, 3)
    
    # 计算相关系数矩阵
    real_corr = np.corrcoef(real_dep.T) if real_dep.shape[0] > 1 else np.full((3, 3), np.nan)
    syn_corr = np.corrcoef(syn_dep.T) if syn_dep.shape[0] > 1 else np.full((3, 3), np.nan)
    
    logger.info("  真实数据相关系数矩阵:")
    logger.info(f"    Length-IAT:    {real_corr[0, 1]:>7.4f}")
    logger.info(f"    Length-Burst:  {real_corr[0, 2]:>7.4f}")
    logger.info(f"    IAT-Burst:     {real_corr[1, 2]:>7.4f}")
    logger.info("  合成数据相关系数矩阵:")
    logger.info(f"    Length-IAT:    {syn_corr[0, 1]:>7.4f}")
    logger.info(f"    Length-Burst:  {syn_corr[0, 2]:>7.4f}")
    logger.info(f"    IAT-Burst:     {syn_corr[1, 2]:>7.4f}")
    
    # 计算相关性差异
    corr_diff = np.nanmean(np.abs(real_corr - syn_corr))
    logger.info(f"  平均相关性差异: {corr_diff:.4f} {'✓' if corr_diff < 0.1 else '⚠'}")
    logger.info("")
    
    logger.info("-"*70)
    logger.info("✓ 生成质量诊断完成")
    logger.info(f"  总体评估: Fidelity={fidelity_level}, "
               f"Validity={validity_rate:.1f}%, "
               f"Structure={'保持' if corr_diff < 0.1 else '部分保持'}")
    logger.info("")
    
    # 5. 可视化: Real vs Synthetic 对比图 (t-SNE)
    logger.info("5️⃣  可视化: Real vs Synthetic 特征空间对比 (t-SNE)")
    from MoudleCode.utils.visualization import plot_real_vs_synthetic_comparison
    
    comparison_save_path = os.path.join(config.DATA_AUGMENTATION_DIR, "figures", "real_vs_synthetic_tsne.png")
    plot_real_vs_synthetic_comparison(
        X_train_original, X_train_synthetic,
        y_train_original, y_train_synthetic,
        comparison_save_path,
        title='TabDDPM Generation Quality: Real vs Synthetic',
        method='tsne'
    )
    logger.info(f"  ✓ t-SNE对比图: {comparison_save_path}")
    logger.info("    (蓝色=真实良性, 红色=真实恶意, 浅色=合成样本)")
    logger.info("    理想结果: 合成样本应覆盖在真实样本之上")
    logger.info("")
    logger.info("📁 输出文件路径:")
    # Save TabDDPM model
    tabddpm_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "tabddpm.pth")
    torch.save(tabddpm.state_dict(), tabddpm_path)
    logger.info(f"  ✓ TabDDPM模型: {tabddpm_path}")
    
    # Save augmented data with metadata
    augmented_data_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "augmented_data.npz")
    # Create mask to identify original vs synthetic samples
    is_original_mask = np.zeros(len(X_augmented), dtype=bool)
    is_original_mask[:n_train_original] = True
    
    np.savez(augmented_data_path,
             X_augmented=X_augmented,
             y_augmented=y_augmented,
             is_original=is_original_mask,
             n_original=n_train_original,
             sample_weights=sample_weights)
    logger.info(f"  ✓ 增强数据: {augmented_data_path}")
    logger.info(f"    (包含原始/合成数据标记)")
    
    logger.info(f"Augmented dataset: {len(X_augmented)} samples (training)")
    logger.info("Stage 2 complete: Labels corrected and data augmented")
    
    return X_augmented, y_augmented, sample_weights, correction_stats, tabddpm, n_train_original


def stage3_finetune_classifier(backbone, X_train, y_train, sample_weights, config, logger, n_original=None, backbone_path=None):
    """
    Stage 3: Fine-tune dual-stream classifier
    
    Args:
        backbone: Frozen pre-trained backbone
        X_train: (N, L, D) training sequences (augmented, includes original + synthetic)
        y_train: (N,) training labels
        sample_weights: (N,) lifecycle weights from Stage 2 (original + synthetic)
        config: configuration object
        logger: logger
        n_original: int, number of original samples (first n_original in X_train)
                    If None, assumes all samples are original
        backbone_path: str, path to the backbone model used (for metadata recording)
        
    Returns:
        classifier: trained MEDAL classifier
        history: training history
        optimal_threshold: optimal decision threshold
    """
    logger.info("")
    logger.info("="*70)
    logger.info("STAGE 3: Fine-tuning Dual-Stream Classifier")
    logger.info("="*70)
    logger.info(f"目标: 训练双流MLP分类器进行最终威胁检测")
    logger.info(f"训练轮数: {config.FINETUNE_EPOCHS} epochs")
    logger.info(f"批次大小: {config.FINETUNE_BATCH_SIZE}")
    logger.info(f"学习率: {config.FINETUNE_LR}")
    logger.info(f"训练样本数: {len(X_train)} (包含增强样本)")
    logger.info(f"损失组件: 加权监督 + 软正交约束 + 一致性损失")
    logger.info("")
    logger.info("📥 输入数据路径:")
    augmented_data_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "augmented_data.npz")
    logger.info(f"  ✓ 增强数据: {augmented_data_path}")

    finetune_backbone = bool(getattr(config, 'FINETUNE_BACKBONE', False))
    finetune_scope = str(getattr(config, 'FINETUNE_BACKBONE_SCOPE', 'projection')).lower()
    finetune_backbone_lr = float(getattr(config, 'FINETUNE_BACKBONE_LR', 1e-5))
    
    # 使用传入的backbone_path参数，如果没有则使用默认路径
    if backbone_path is None:
        backbone_path_display = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    else:
        backbone_path_display = backbone_path
    logger.info(f"  ✓ 骨干网络模型: {backbone_path_display}")
    logger.info("")
    
    # Create classifier
    classifier = MEDAL_Classifier(backbone, config).to(config.DEVICE)
    logger.info("✓ 双流分类器创建完成 (MLP_A + MLP_B)")
    
    # Loss function
    criterion = DualStreamLoss(config)
    logger.info("✓ 损失函数初始化完成")

    logger.info(
        f"🔧 LogitMargin: enabled={bool(getattr(config, 'USE_LOGIT_MARGIN', False))} "
        f"m={float(getattr(config, 'LOGIT_MARGIN_M', 0.0))} "
        f"warmup_epochs={int(getattr(config, 'LOGIT_MARGIN_WARMUP_EPOCHS', 0))}"
    )
    logger.info(
        f"🔧 MarginLoss: enabled={bool(getattr(config, 'USE_MARGIN_LOSS', False))} "
        f"m={float(getattr(config, 'MARGIN_M', 0.0))} "
        f"s={float(getattr(config, 'MARGIN_S', 1.0))} "
        f"w={float(getattr(config, 'MARGIN_LOSS_WEIGHT', 0.0))}"
    )

    # Stage 3 online augmentation (input-level)
    use_stage3_online_aug = bool(getattr(config, 'STAGE3_ONLINE_AUGMENTATION', False))
    augmentor = None
    if use_stage3_online_aug:
        from MoudleCode.feature_extraction.traffic_augmentation import TrafficAugmentation
        augmentor = TrafficAugmentation(config)
        logger.info("✓ Stage 3 在线表征增强已启用 (TrafficAugmentation)")
    else:
        logger.info("✓ Stage 3 在线表征增强未启用")
    
    # Stage 3 ST-Mixup (Spatio-Temporal Mixup)
    use_st_mixup = bool(getattr(config, 'STAGE3_USE_ST_MIXUP', False))
    st_mixup = None
    if use_st_mixup:
        from MoudleCode.feature_extraction.st_mixup import create_st_mixup
        st_mixup_mode = str(getattr(config, 'STAGE3_ST_MIXUP_MODE', 'intra_class'))
        st_mixup = create_st_mixup(config, mode=st_mixup_mode)
        logger.info(f"✓ Stage 3 ST-Mixup已启用 (mode={st_mixup_mode})")
        logger.info(f"  - Alpha: {config.STAGE3_ST_MIXUP_ALPHA} (混合强度)")
        logger.info(f"  - Warmup: {config.STAGE3_ST_MIXUP_WARMUP_EPOCHS} epochs (渐进式启用)")
        logger.info(f"  - Max Prob: {config.STAGE3_ST_MIXUP_MAX_PROB} (最大混合概率)")
        logger.info(f"  - Time Shift: {config.STAGE3_ST_MIXUP_TIME_SHIFT_RATIO} (时间偏移比例)")
        if st_mixup_mode == 'intra_class':
            logger.info(f"  - 策略: 类内混合（只混合同类样本，避免语义冲突）")
        elif st_mixup_mode == 'selective':
            logger.info(f"  - 策略: 选择性混合（只混合困难样本）")
            logger.info(f"  - Uncertainty Threshold: {config.STAGE3_ST_MIXUP_UNCERTAINTY_THRESHOLD}")
    else:
        logger.info("✓ Stage 3 ST-Mixup未启用")
    
    # -------------------- Backbone finetuning policy --------------------
    if finetune_backbone:
        # Default: freeze everything first
        backbone.freeze()
        if finetune_scope == 'all':
            backbone.unfreeze()
            logger.info("🔥 Stage 3: Backbone finetuning enabled (scope=all)")
        elif finetune_scope == 'projection':
            # Only train the bidirectional projection head
            if hasattr(backbone, 'projection'):
                for p in backbone.projection.parameters():
                    p.requires_grad = True
                logger.info("🔥 Stage 3: Backbone finetuning enabled (scope=projection)")
            else:
                logger.warning("⚠️  FINETUNE_BACKBONE_SCOPE='projection' but backbone has no attribute 'projection'. Falling back to frozen backbone.")
                finetune_backbone = False
        else:
            logger.warning(f"⚠️  Unknown FINETUNE_BACKBONE_SCOPE='{finetune_scope}'. Falling back to frozen backbone.")
            finetune_backbone = False
    else:
        backbone.freeze()

    # Set backbone mode
    if finetune_backbone:
        backbone.train()
    else:
        backbone.eval()

    # -------------------- Optimizer (classifier + optional backbone) --------------------
    param_groups = [
        {'params': classifier.dual_mlp.parameters(), 'lr': float(config.FINETUNE_LR)}
    ]
    if finetune_backbone:
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        if len(backbone_params) == 0:
            logger.warning("⚠️  FINETUNE_BACKBONE=True but no trainable backbone params found; backbone will remain frozen.")
            finetune_backbone = False
            backbone.eval()
        else:
            param_groups.append({'params': backbone_params, 'lr': finetune_backbone_lr})
            logger.info(f"✓ 优化器包含骨干网络可训练参数: {len(backbone_params)} | backbone_lr={finetune_backbone_lr}")

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config.PRETRAIN_WEIGHT_DECAY
    )
    if finetune_backbone:
        logger.info("✓ 优化器初始化完成 (分类器 + 骨干网络微调)")
    else:
        logger.info("✓ 优化器初始化完成 (仅优化分类器参数，骨干网络已冻结)")
    
    logger.info("")
    logger.info("💡 训练流程说明:")
    if finetune_backbone:
        logger.info("  每个batch: 原始数据 → 骨干网络(可训练) → 特征 → 分类器(训练) → 损失")
        logger.info("  骨干网络参数会随分类器一起更新")
    else:
        logger.info("  每个batch: 原始数据 → 骨干网络(冻结) → 特征 → 分类器(训练) → 损失")
        logger.info("  骨干网络实时提取特征，但参数不更新")
    logger.info("")
    logger.info("="*70)
    val_split = float(getattr(config, 'FINETUNE_VAL_SPLIT', 0.2))
    val_per_class = int(getattr(config, 'FINETUNE_VAL_PER_CLASS', 0))
    if val_per_class > 0:
        logger.info(f"📊 训练/验证划分 (val_per_class={val_per_class} per class)")
    elif val_split > 0:
        logger.info(f"📊 训练/验证划分 (val_split={val_split:.2f})")
    else:
        logger.info("📊 全量训练（不划分验证集）")
    logger.info("="*70)
    logger.info(f"  样本总数: {len(X_train)} 个样本 (包含原始 + 合成)")
    logger.info("")

    X_val_split = None
    y_val_split = None
    sample_weights_val_split = None

    if val_per_class > 0:
        rng = np.random.RandomState(int(getattr(config, 'SEED', 42)))
        y_np = np.asarray(y_train).astype(int)
        all_idx = np.arange(len(y_np))

        idx0 = all_idx[y_np == 0]
        idx1 = all_idx[y_np == 1]
        n0 = int(min(val_per_class, len(idx0)))
        n1 = int(min(val_per_class, len(idx1)))
        if n0 == 0 or n1 == 0:
            logger.warning("⚠️  FINETUNE_VAL_PER_CLASS is set but one class has 0 samples; falling back to no validation split.")
            X_train_split = X_train
            y_train_split = y_train
            sample_weights_split = sample_weights
        else:
            val_idx0 = rng.choice(idx0, size=n0, replace=False)
            val_idx1 = rng.choice(idx1, size=n1, replace=False)
            val_idx = np.concatenate([val_idx0, val_idx1])
            train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)

            X_train_split = X_train[train_idx]
            y_train_split = y_train[train_idx]
            sample_weights_split = sample_weights[train_idx]
            X_val_split = X_train[val_idx]
            y_val_split = y_train[val_idx]
            sample_weights_val_split = sample_weights[val_idx]

            logger.info(f"  训练集: {len(X_train_split)}")
            logger.info(f"  验证集: {len(X_val_split)} (per_class={val_per_class})")
            logger.info("")
    elif val_split > 0:
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split, sample_weights_split, sample_weights_val_split = train_test_split(
            X_train,
            y_train,
            sample_weights,
            test_size=val_split,
            random_state=int(getattr(config, 'SEED', 42)),
            stratify=y_train
        )
        logger.info(f"  训练集: {len(X_train_split)}")
        logger.info(f"  验证集: {len(X_val_split)}")
        logger.info("")
    else:
        X_train_split = X_train
        y_train_split = y_train
        sample_weights_split = sample_weights
    
    # ==================== 温室训练策略：强制1:1平衡采样 ====================
    use_balanced_sampling = getattr(config, 'USE_BALANCED_SAMPLING', True)
    use_hard_mining = bool(getattr(config, 'STAGE3_HARD_MINING', False))
    hard_mining_warmup_epochs = int(getattr(config, 'STAGE3_HARD_MINING_WARMUP_EPOCHS', 5))
    hard_mining_freq_epochs = int(getattr(config, 'STAGE3_HARD_MINING_FREQ_EPOCHS', 3))
    hard_mining_topk_ratio = float(getattr(config, 'STAGE3_HARD_MINING_TOPK_RATIO', 0.2))
    hard_mining_multiplier = float(getattr(config, 'STAGE3_HARD_MINING_MULTIPLIER', 3.0))
    hard_mining_pos_prob_max = float(getattr(config, 'STAGE3_HARD_MINING_POS_PROB_MAX', 0.70))
    hard_mining_neg_prob_min = float(getattr(config, 'STAGE3_HARD_MINING_NEG_PROB_MIN', 0.60))
    
    # Dataset is created once; sampler may be rebuilt during training when hard mining is enabled.
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.LongTensor(y_train_split),
        torch.FloatTensor(sample_weights_split)
    )

    def _build_train_loader_with_sampler(_weights_np: np.ndarray):
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(_weights_np, dtype=torch.double),
            num_samples=len(_weights_np),
            replacement=True
        )
        return DataLoader(
            train_dataset,
            batch_size=config.FINETUNE_BATCH_SIZE,
            sampler=sampler
        )

    def _compute_sampling_weights_from_model():
        classifier.eval()
        if finetune_backbone:
            backbone.eval()
        probs_list = []
        labels_list = []
        with torch.no_grad():
            eval_loader = DataLoader(train_dataset, batch_size=getattr(config, 'TEST_BATCH_SIZE', 256), shuffle=False)
            for X_b, y_b, _w_b in eval_loader:
                X_b = X_b.to(config.DEVICE)
                z_b = backbone(X_b, return_sequence=False)
                logits_a, logits_b = classifier.dual_mlp(z_b, return_separate=True)
                logits_avg = (logits_a + logits_b) / 2.0
                p = torch.softmax(logits_avg, dim=1)[:, 1]
                probs_list.append(p.detach().cpu().numpy())
                labels_list.append(y_b.detach().cpu().numpy())

        classifier.train()
        if finetune_backbone:
            backbone.train()

        probs = np.concatenate(probs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0).astype(int)

        if use_balanced_sampling:
            class_counts = np.bincount(labels)
            class_weights = 1.0 / np.maximum(class_counts, 1)
            base_w = class_weights[labels]
        else:
            base_w = np.ones_like(probs, dtype=np.float64)

        hard_pos = (labels == 1) & (probs < hard_mining_pos_prob_max)
        hard_neg = (labels == 0) & (probs > hard_mining_neg_prob_min)
        hard_mask = hard_pos | hard_neg

        k_ratio = float(np.clip(hard_mining_topk_ratio, 0.0, 1.0))
        k = int(max(0, round(len(probs) * k_ratio)))
        if k > 0:
            difficulty = np.where(labels == 1, 1.0 - probs, probs)
            topk_idx = np.argpartition(-difficulty, k - 1)[:k]
            hard_mask[topk_idx] = True

        weights = base_w.astype(np.float64)
        weights[hard_mask] = weights[hard_mask] * hard_mining_multiplier
        return weights, {
            'n_hard_total': int(np.sum(hard_mask)),
            'n_hard_pos': int(np.sum(hard_pos)),
            'n_hard_neg': int(np.sum(hard_neg)),
            'pos_prob_max': float(hard_mining_pos_prob_max),
            'neg_prob_min': float(hard_mining_neg_prob_min),
            'topk_ratio': float(hard_mining_topk_ratio),
            'multiplier': float(hard_mining_multiplier),
        }

    if use_balanced_sampling:
        logger.info("🌡️  温室训练策略：启用加权平衡采样")
        logger.info("  目标：正常:恶意 = 1.5:1 (恶意样本权重更高)")
        logger.info("")
        
        class_counts = np.bincount(y_train_split)
        logger.info(f"  原始分布: 正常={class_counts[0]}, 恶意={class_counts[1]}")
        logger.info(f"  原始比例: {class_counts[0]}:{class_counts[1]} ≈ {class_counts[0]/class_counts[1]:.1f}:1")
        
        # 设置采样权重以达到 正常:恶意 = 1.5:1
        # 如果正常类有N个样本，恶意类有M个样本
        # 我们希望采样后的期望比例是 1.5:1
        # 设正常类权重为w0，恶意类权重为w1
        # 则 (N*w0) : (M*w1) = 1.5 : 1
        # 即：w0 = 1.5 * M / N * w1
        # 设w1 = 1，则 w0 = 1.5 * M / N
        
        n_benign = class_counts[0]
        n_malicious = class_counts[1]
        
        # 计算类别权重：正常类权重 = 1.5 * (恶意数量 / 正常数量)
        weight_benign = 1.5 * n_malicious / n_benign
        weight_malicious = 1.0
        
        class_weights = np.array([weight_benign, weight_malicious])
        sample_sampling_weights = class_weights[y_train_split]
        
        logger.info(f"  采样权重: 正常类={weight_benign:.4f}, 恶意类={weight_malicious:.4f}")
        logger.info(f"  期望采样比例: 1.5:1 (正常:恶意)")

        train_loader = _build_train_loader_with_sampler(sample_sampling_weights)
        logger.info(f"  ✓ 平衡采样器创建完成")
        logger.info(f"  ✓ 每个batch期望比例: 1.5:1 (正常:恶意)")
    else:
        logger.info("⚠️  使用原始分布训练（未启用平衡采样）")
        if use_hard_mining:
            train_loader = _build_train_loader_with_sampler(np.ones(len(y_train_split), dtype=np.float64))
        else:
            train_loader = DataLoader(train_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True)
    
    logger.info(f"✓ 数据加载器准备完成 ({len(train_loader)} 个批次)")
    logger.info("")
    logger.info("开始训练...")
    logger.info("-"*70)
    
    # Training loop
    history = {
        'train_loss': [], 'supervision': [], 'soft_f1_loss': [], 'orthogonality': [],
        'consistency': [], 'margin': [], 'lambda_orth': [],
        'lambda_con': [], 'lambda_margin': [], 'train_f1': [], 'val_f1': [], 'val_threshold': [],
        'train_f1_star': [], 'train_threshold_star': []
    }
    
    classifier.train()

    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    best_threshold = float(getattr(config, 'MALICIOUS_THRESHOLD', 0.5))
    finetuned_backbone_path = None
    
    # Early stopping variables
    use_early_stopping = bool(getattr(config, 'FINETUNE_EARLY_STOPPING', False))
    es_warmup_epochs = int(getattr(config, 'FINETUNE_ES_WARMUP_EPOCHS', 30))
    es_patience = int(getattr(config, 'FINETUNE_ES_PATIENCE', 25))
    es_min_delta = float(getattr(config, 'FINETUNE_ES_MIN_DELTA', 0.002))
    no_improve_count = 0
    best_es_metric = -1.0
    allow_train_metric_es = bool(getattr(config, 'FINETUNE_ES_ALLOW_TRAIN_METRIC', False))
    if use_early_stopping and X_val_split is None and not allow_train_metric_es:
        use_early_stopping = False
        logger.info("⚠️  早停已自动禁用：未设置验证集 (FINETUNE_VAL_SPLIT=0)，训练集指标不适合作为早停依据")
        logger.info("")
    
    if use_early_stopping:
        logger.info(f"✓ 早停机制已启用:")
        logger.info(f"  - 预热轮数: {es_warmup_epochs} (前{es_warmup_epochs}轮不触发早停)")
        logger.info(f"  - 耐心值: {es_patience} (连续{es_patience}轮不改善则停止)")
        logger.info(f"  - 改善阈值: {es_min_delta:.4f} (F1改善需超过{es_min_delta*100:.2f}%)")
        logger.info("")
    
    for epoch in range(config.FINETUNE_EPOCHS):
        if use_hard_mining and (epoch + 1) >= hard_mining_warmup_epochs:
            if hard_mining_freq_epochs <= 0:
                hard_mining_freq_epochs = 1
            if ((epoch + 1 - hard_mining_warmup_epochs) % hard_mining_freq_epochs) == 0:
                try:
                    new_weights, hm_stats = _compute_sampling_weights_from_model()
                    train_loader = _build_train_loader_with_sampler(new_weights)
                    logger.info(
                        f"🧲 Stage 3 Hard Mining: updated sampler | "
                        f"hard_total={hm_stats['n_hard_total']} "
                        f"(pos={hm_stats['n_hard_pos']}, neg={hm_stats['n_hard_neg']}) | "
                        f"topk_ratio={hm_stats['topk_ratio']:.2f} "
                        f"mult={hm_stats['multiplier']:.2f} "
                        f"pos_prob_max={hm_stats['pos_prob_max']:.2f} "
                        f"neg_prob_min={hm_stats['neg_prob_min']:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"⚠️  Stage 3 Hard Mining skipped due to error: {e}")

        epoch_loss = 0.0
        epoch_losses = {
            'supervision': 0.0, 'soft_f1': 0.0, 'orthogonality': 0.0, 
            'consistency': 0.0, 'margin': 0.0,
            'logit_margin_scale': 0.0
        }
        
        # 收集训练集预测用于计算 F1
        all_train_probs = []
        all_train_labels = []
        
        for X_batch, y_batch, w_batch in train_loader:
            X_batch = X_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)
            w_batch = w_batch.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Online augmentation (training only)
            # Step 1: TrafficAugmentation (物理感知的单样本增强)
            if augmentor is not None:
                X_batch, _ = augmentor(X_batch)
            
            # Step 2: ST-Mixup (类内时空混合，渐进式启用)
            if st_mixup is not None:
                if st_mixup_mode == 'selective':
                    # 选择性模式：需要传入模型
                    X_batch, y_batch = st_mixup(
                        X_batch, y_batch, 
                        epoch=epoch, 
                        total_epochs=config.FINETUNE_EPOCHS,
                        classifier=classifier,
                        backbone=backbone
                    )
                else:
                    # 类内模式：不需要模型
                    X_batch, y_batch = st_mixup(
                        X_batch, y_batch,
                        epoch=epoch,
                        total_epochs=config.FINETUNE_EPOCHS
                    )

            # Extract features
            if finetune_backbone:
                z = backbone(X_batch, return_sequence=False)
            else:
                with torch.no_grad():
                    z = backbone(X_batch, return_sequence=False)
            
            # Compute loss (weight-aware)
            loss, loss_dict = criterion(classifier.dual_mlp, z, y_batch, w_batch, epoch, config.FINETUNE_EPOCHS)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss_dict['total']
            epoch_losses['supervision'] += loss_dict['supervision']
            epoch_losses['soft_f1'] += loss_dict.get('soft_f1', 0.0)
            epoch_losses['orthogonality'] += loss_dict['orthogonality']
            epoch_losses['consistency'] += loss_dict['consistency']
            if 'margin' in loss_dict:
                epoch_losses['margin'] = epoch_losses.get('margin', 0.0) + loss_dict['margin']
            if 'logit_margin_scale' in loss_dict:
                epoch_losses['logit_margin_scale'] = epoch_losses.get('logit_margin_scale', 0.0) + float(loss_dict['logit_margin_scale'])
            
            # 收集预测概率用于计算 F1
            with torch.no_grad():
                logits_a, logits_b = classifier.dual_mlp(z, return_separate=True)
                logits_avg = (logits_a + logits_b) / 2.0
                probs = torch.softmax(logits_avg, dim=1)
                all_train_probs.append(probs.cpu().numpy())
                all_train_labels.append(y_batch.cpu().numpy())
        
        n_batches = len(train_loader)
        epoch_loss /= n_batches
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # 计算训练集 Binary F1-Score (固定阈值0.5，仅用于参考)
        train_probs = np.concatenate(all_train_probs)
        train_labels = np.concatenate(all_train_labels)
        train_preds = (train_probs[:, 1] >= 0.5).astype(int)
        from sklearn.metrics import f1_score
        train_f1 = f1_score(train_labels, train_preds, pos_label=1, zero_division=0)

        # 训练集单类F1*(pos=1) & 最优阈值（用于无验证集时选择 best checkpoint）
        train_f1_star = None
        train_threshold_star = None
        if X_val_split is None:
            from MoudleCode.utils.helpers import find_optimal_threshold
            train_threshold_star, _train_metric, _ = find_optimal_threshold(train_labels, train_probs, metric='f1_binary', positive_class=1)
            train_preds_star = (train_probs[:, 1] >= float(train_threshold_star)).astype(int)
            train_f1_star = f1_score(train_labels, train_preds_star, pos_label=1, zero_division=0)

        val_f1 = None
        val_threshold = None

        if X_val_split is not None:
            classifier.eval()
            all_val_probs = []
            all_val_labels = []
            with torch.no_grad():
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_split),
                    torch.LongTensor(y_val_split),
                    torch.FloatTensor(sample_weights_val_split)
                )
                val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
                for X_batch, y_batch, _w_batch in val_loader:
                    X_batch = X_batch.to(config.DEVICE)
                    y_batch = y_batch.to(config.DEVICE)
                    z = backbone(X_batch, return_sequence=False)
                    logits_a, logits_b = classifier.dual_mlp(z, return_separate=True)
                    logits_avg = (logits_a + logits_b) / 2.0
                    probs = torch.softmax(logits_avg, dim=1)
                    all_val_probs.append(probs.cpu().numpy())
                    all_val_labels.append(y_batch.cpu().numpy())

            val_probs = np.concatenate(all_val_probs)
            val_labels = np.concatenate(all_val_labels)

            from MoudleCode.utils.helpers import find_optimal_threshold
            val_threshold, val_metric, _ = find_optimal_threshold(val_labels, val_probs, metric='f1_binary', positive_class=1)
            val_preds = (val_probs[:, 1] >= float(val_threshold)).astype(int)
            val_f1 = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)

            if val_f1 > best_f1:
                best_f1 = float(val_f1)
                best_epoch = int(epoch + 1)
                best_threshold = float(val_threshold)
                best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
        else:
            if train_f1_star is None:
                raise RuntimeError("Expected train_f1_star to be computed when val_split is disabled")
            if train_f1_star > best_f1:
                best_f1 = float(train_f1_star)
                best_epoch = int(epoch + 1)
                best_threshold = float(train_threshold_star)
                best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
        
        classifier.eval()
        classifier.train()
        
        # 保存历史
        history['train_loss'].append(epoch_loss)
        history['supervision'].append(epoch_losses['supervision'])
        history['soft_f1_loss'].append(epoch_losses['soft_f1'])
        history['orthogonality'].append(epoch_losses['orthogonality'])
        history['consistency'].append(epoch_losses['consistency'])
        history['margin'].append(epoch_losses.get('margin', 0.0))
        history['lambda_orth'].append(loss_dict['lambda_orth'])
        history['lambda_con'].append(loss_dict['lambda_con'])
        history['lambda_margin'].append(loss_dict.get('lambda_margin', 0.0))
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1 if val_f1 is not None else np.nan)
        history['val_threshold'].append(val_threshold if val_threshold is not None else np.nan)
        history['train_f1_star'].append(train_f1_star if train_f1_star is not None else np.nan)
        history['train_threshold_star'].append(train_threshold_star if train_threshold_star is not None else np.nan)
        
        # Early stopping check
        if use_early_stopping and (epoch + 1) >= es_warmup_epochs:
            current_metric = val_f1 if val_f1 is not None else train_f1_star
            if current_metric is not None:
                current_metric = float(current_metric)
                if current_metric > (best_es_metric + es_min_delta):
                    best_es_metric = current_metric
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                    if no_improve_count >= es_patience:
                        logger.info("")
                        logger.info("="*70)
                        logger.info(f"🛑 早停触发 (Early Stopping Triggered)")
                        logger.info("="*70)
                        logger.info(f"  当前轮次: Epoch {epoch+1}/{config.FINETUNE_EPOCHS}")
                        logger.info(f"  最佳F1: {best_f1:.4f} (Epoch {best_epoch})")
                        logger.info(f"  当前F1: {current_metric:.4f}")
                        logger.info(f"  连续{no_improve_count}轮未改善超过{es_min_delta:.4f}")
                        logger.info(f"  提前终止训练，节省 {config.FINETUNE_EPOCHS - epoch - 1} 轮")
                        logger.info("="*70)
                        logger.info("")
                        break
        
        # 每个epoch都输出详细日志
        progress = (epoch + 1) / config.FINETUNE_EPOCHS * 100
        soft_f1_str = f" | F1Loss: {epoch_losses['soft_f1']:.4f}" if epoch_losses['soft_f1'] > 0 else ""
        logit_margin_scale = float(epoch_losses.get('logit_margin_scale', 0.0))
        logit_margin_str = f" | LogitM: {logit_margin_scale:.2f}" if logit_margin_scale > 0 else ""
        if val_f1 is not None:
            logger.info(f"[Stage 3] Epoch [{epoch+1}/{config.FINETUNE_EPOCHS}] ({progress:.1f}%) | "
                      f"TrLoss: {epoch_loss:.4f} | "
                      f"TrF1@0.5: {train_f1:.4f} | "
                      f"ValF1*: {val_f1:.4f} | "
                      f"ValTh*: {float(val_threshold):.4f} | "
                      f"Sup: {epoch_losses['supervision']:.4f}{soft_f1_str}{logit_margin_str}")
        else:
            train_f1_star_disp = float(train_f1_star) if train_f1_star is not None else float('nan')
            train_th_star_disp = float(train_threshold_star) if train_threshold_star is not None else float('nan')
            logger.info(f"[Stage 3] Epoch [{epoch+1}/{config.FINETUNE_EPOCHS}] ({progress:.1f}%) | "
                      f"TrLoss: {epoch_loss:.4f} | "
                      f"TrF1@0.5: {train_f1:.4f} | "
                      f"TrF1*: {train_f1_star_disp:.4f} | "
                      f"TrTh*: {train_th_star_disp:.4f} | "
                      f"Sup: {epoch_losses['supervision']:.4f}{soft_f1_str}{logit_margin_str}")
    
    logger.info("-"*70)
    logger.info("✓ Stage 3 完成: 分类器微调完成")
    actual_epochs = len(history['train_loss'])
    logger.info(f"  训练集 - 最终损失: {history['train_loss'][-1]:.4f}, 最终F1: {history['train_f1'][-1]:.4f}")
    logger.info(f"  实际训练轮数: {actual_epochs}/{config.FINETUNE_EPOCHS} epochs")
    if actual_epochs < config.FINETUNE_EPOCHS:
        logger.info(f"  ✓ 早停生效，节省了 {config.FINETUNE_EPOCHS - actual_epochs} 轮训练")
    if best_epoch > 0:
        if X_val_split is not None:
            logger.info(f"  最佳ValF1*(pos=1, auto-threshold): {best_f1:.4f} (epoch {best_epoch}, th={best_threshold:.4f})")
        else:
            logger.info(f"  最佳TrF1*(pos=1, auto-threshold): {best_f1:.4f} (epoch {best_epoch}, th={best_threshold:.4f})")
    logger.info("")
    
    # 生成特征分布可视化（Stage 3分类器训练后）
    logger.info("📊 生成特征分布可视化...")
    logger.info("  提取训练集特征用于可视化...")
    
    classifier.eval()
    if finetune_backbone:
        backbone.eval()
    with torch.no_grad():
        # Extract features from training set for visualization
        X_train_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
        features_list = []
        batch_size = 64
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            z_batch = backbone(X_batch, return_sequence=False)
            features_list.append(z_batch.cpu().numpy())
        train_features = np.concatenate(features_list, axis=0)
    
    feature_dist_path = os.path.join(config.CLASSIFICATION_DIR, "figures", "feature_distribution_stage3.png")
    plot_feature_space(train_features, y_train, feature_dist_path,
                      title="Stage 3: Feature Distribution (After Classifier Training)", method='tsne')
    logger.info(f"  ✓ 特征分布图 (t-SNE): {feature_dist_path}")
    logger.info("")
    
    optimal_threshold = float(best_threshold)
    
    logger.info("📁 输出文件路径:")

    # Save finetuned backbone (never overwrite original backbone checkpoint)
    if finetune_backbone:
        finetuned_backbone_path = os.path.join(config.CLASSIFICATION_DIR, "models", "backbone_finetuned.pth")
        torch.save(backbone.state_dict(), finetuned_backbone_path)
        logger.info(f"  ✓ 微调后的骨干网络: {finetuned_backbone_path}")

    # Save best model by validation F1(pos=1) if validation is enabled
    if best_state is not None:
        # Save best model
        best_path = os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_best_f1.pth")
        torch.save(best_state, best_path)
        logger.info(f"  ✓ 最佳模型: {best_path}")

    # Save final model
    final_path = os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_final.pth")
    torch.save(classifier.dual_mlp.state_dict(), final_path)
    logger.info(f"  ✓ 最终模型: {final_path}")
    
    # Save training history
    history_path = os.path.join(config.CLASSIFICATION_DIR, "models", "training_history.npz")
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})
    logger.info(f"  ✓ 训练历史: {history_path}")
    
    # Save model metadata (including backbone path used for training)
    # This helps test.py know which backbone to load
    if finetuned_backbone_path is not None:
        backbone_path = finetuned_backbone_path
    elif backbone_path is None:
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    
    metadata_path = os.path.join(config.CLASSIFICATION_DIR, "models", "model_metadata.json")
    metadata = {
        'backbone_path': backbone_path,  # The actual backbone path used in training
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X_train),
        'n_original': n_original if n_original is not None else len(X_train),
        'finetune_epochs': config.FINETUNE_EPOCHS,
    }
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ 模型元数据: {metadata_path}")
    logger.info(f"    (记录了使用的骨干网络: {os.path.basename(backbone_path)})")
    
    return classifier, history, optimal_threshold

def main(args):
    """Main training function"""
    
    # Setup
    set_seed(config.SEED)
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, "logs"), name='train')
    
    logger.info("="*70)
    logger.info("MEDAL-Lite Training Pipeline")
    logger.info("="*70)
    
    # GPU信息
    if torch.cuda.is_available():
        logger.info(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        logger.info(f"  显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"  CUDA版本: {torch.version.cuda}")
        logger.info(f"  使用设备: {config.DEVICE}")
    else:
        logger.warning("⚠ GPU不可用，使用CPU训练（速度会较慢）")
        logger.info(f"  使用设备: {config.DEVICE}")
    
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get stage range first to determine what to run
    start_stage = getattr(args, 'start_stage', 1)
    end_stage = getattr(args, 'end_stage', 3)
    
    # Convert start_stage/end_stage to int if it's a string
    if isinstance(start_stage, str):
        try:
            start_stage = int(start_stage)
        except ValueError:
            logger.error(f"❌ 无效的起始阶段: {start_stage}")
            return

    if isinstance(end_stage, str):
        try:
            end_stage = int(end_stage)
        except ValueError:
            logger.error(f"❌ 无效的结束阶段: {end_stage}")
            return

    if end_stage < start_stage:
        logger.error(f"❌ 无效阶段范围: start_stage={start_stage} > end_stage={end_stage}")
        return
    
    # ========================
    # Load Dataset (needed for Stage 1/2, and also for Stage 3-only runs)
    # ========================
    X_train = None
    y_train_clean = None
    y_train_noisy = None
    
    if start_stage <= 3:
        # Need to load raw dataset for Stage 1/2, and for Stage 3-only (skip Stage 2)
        logger.info("\n" + "="*70)
        logger.info("📦 数据集加载 Dataset Loading")
        logger.info("="*70)
        logger.info(f"训练集配置:")
        logger.info(f"  正常流量路径: {config.BENIGN_TRAIN}")
        logger.info(f"  恶意流量路径: {config.MALICIOUS_TRAIN}")
        logger.info(f"  序列长度: {config.SEQUENCE_LENGTH} 个数据包")
        logger.info(f"  说明: 将读取上述路径下所有pcap文件，流数在处理时统计")
        logger.info("")
        
        # 优先使用预处理好的数据
        if PREPROCESS_AVAILABLE and check_preprocessed_exists('train'):
            logger.info("✓ 发现预处理文件，直接加载...")
            X_train, y_train_clean, train_files = load_preprocessed('train')
            logger.info(f"  从预处理文件加载: {X_train.shape[0]} 个样本")
        else:
            # 从PCAP文件加载
            logger.info("开始加载训练数据集（从PCAP文件）...")
            logger.info("💡 提示: 运行 'python preprocess.py' 可预处理数据，加速后续训练")
            X_train, y_train_clean, train_files = load_dataset(
                benign_dir=config.BENIGN_TRAIN,
                malicious_dir=config.MALICIOUS_TRAIN,
                sequence_length=config.SEQUENCE_LENGTH
            )
        
        if X_train is None:
            logger.error("❌ 训练数据集加载失败!")
            return
        
        logger.info("")
        logger.info("✓ 训练数据集加载完成")
        logger.info(f"  数据形状: {X_train.shape} (样本数×序列长度×特征维度)")
        logger.info(f"  正常样本: {(y_train_clean==0).sum()} 个")
        logger.info(f"  恶意样本: {(y_train_clean==1).sum()} 个")
        logger.info("")
        
        # 注意：Stage 1 (特征提取) 不需要标签，是无监督学习
        # 只有 Stage 2 (标签矫正) 和 Stage 3 (分类) 才需要标签
        # 因此，标签噪声注入延迟到需要时再进行
        if start_stage >= 2:
            # Inject label noise (only needed for Stage 2+)
            logger.info(f"🔀 注入标签噪声 ({config.LABEL_NOISE_RATE*100:.0f}%)...")
            y_train_noisy, noise_mask = inject_label_noise(y_train_clean, config.LABEL_NOISE_RATE)
            logger.info(f"✓ 噪声标签创建完成: {noise_mask.sum()} 个标签被翻转")
            logger.info(f"  原始标签分布: 正常={(y_train_clean==0).sum()}, 恶意={(y_train_clean==1).sum()}")
            logger.info(f"  噪声标签分布: 正常={(y_train_noisy==0).sum()}, 恶意={(y_train_noisy==1).sum()}")
        else:
            # Stage 1 不需要标签
            logger.info("💡 Stage 1 (特征提取) 是无监督学习，不使用标签")
            y_train_noisy = None
            noise_mask = None
    else:
        # Starting from later stages, skip raw dataset loading
        logger.info("\n" + "="*70)
        logger.info("⏭️  跳过原始数据集加载")
        logger.info("="*70)
        logger.info("")
    
    # ========================
    # Stage 1: Pre-train Backbone (unsupervised, no labels needed)
    # ========================
    backbone = MicroBiMambaBackbone(config)
    
    if start_stage <= 1:
        # 根据对比学习方法选择批次大小
        use_instance_contrastive = getattr(config, 'USE_INSTANCE_CONTRASTIVE', False)
        contrastive_method = getattr(config, 'CONTRASTIVE_METHOD', 'infonce')
        
        if use_instance_contrastive and str(contrastive_method).lower() == 'nnclr':
            # NNCLR 需要更小的批次以避免显存溢出
            batch_size = getattr(config, 'PRETRAIN_BATCH_SIZE_NNCLR', 64)
            logger.info(f"✓ 检测到 NNCLR 方法，使用专用批次大小: {batch_size}")
            logger.info(f"  (NNCLR 显存占用高，需要减小批次)")
        else:
            # SimMTM 或 InfoNCE 使用默认批次大小
            batch_size = config.PRETRAIN_BATCH_SIZE
        
        # SimMTM is unsupervised, so we only need X_train, not labels
        dataset = TensorDataset(torch.FloatTensor(X_train))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        backbone, pretrain_history = stage1_pretrain_backbone(backbone, train_loader, config, logger)
        # Backbone is already saved in stage1_pretrain_backbone function
        if end_stage <= 1:
            logger.info("\n" + "="*70)
            logger.info("✅ 已完成到 Stage 1，按 end_stage 设置提前结束")
            logger.info("="*70)
            return backbone
    else:
        # Load pre-trained backbone (required for Stage 2 and 3)
        # 优先使用命令行指定的backbone_path
        if hasattr(args, 'backbone_path') and args.backbone_path:
            backbone_path = args.backbone_path
            logger.info(f"使用指定的骨干网络: {backbone_path}")
        else:
            backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
        
        retrain_backbone = bool(getattr(args, 'retrain_backbone', False))
        if retrain_backbone:
            logger.warning("⚠ --retrain_backbone 已指定：将使用随机初始化骨干网络（不加载预训练权重）")
            backbone.freeze()
        else:
            if os.path.exists(backbone_path):
                logger.info("📥 输入数据路径:")
                logger.info(f"  ✓ 骨干网络模型: {backbone_path}")
                logger.info("")
                logger.info(f"✓ 加载已有骨干网络: {backbone_path}")
                try:
                    backbone_state = torch.load(backbone_path, map_location=config.DEVICE, weights_only=True)
                except TypeError:
                    backbone_state = torch.load(backbone_path, map_location=config.DEVICE)
                load_state_dict_shape_safe(backbone, backbone_state, logger, prefix="backbone")
                logger.info(f"✓ 加载已有骨干网络: {backbone_path}")
            else:
                logger.error(f"❌ 找不到预训练骨干网络: {backbone_path}")
                logger.error("   请先运行 Stage 1 或从 Stage 1 开始训练")
                return
    
    # ========================
    # Stage 2: Label Correction & Augmentation
    # ========================
    if start_stage <= 2 and end_stage >= 2:
        # 如果从Stage 2开始，但还没有注入噪声，现在注入
        if y_train_noisy is None and y_train_clean is not None:
            logger.info("")
            logger.info(f"🔀 注入标签噪声 ({config.LABEL_NOISE_RATE*100:.0f}%)...")
            y_train_noisy, noise_mask = inject_label_noise(y_train_clean, config.LABEL_NOISE_RATE)
            logger.info(f"✓ 噪声标签创建完成: {noise_mask.sum()} 个标签被翻转")
            logger.info(f"  原始标签分布: 正常={(y_train_clean==0).sum()}, 恶意={(y_train_clean==1).sum()}")
            logger.info(f"  噪声标签分布: 正常={(y_train_noisy==0).sum()}, 恶意={(y_train_noisy==1).sum()}")
        
        stage2_mode = getattr(args, 'stage2_mode', 'standard')
        X_augmented, y_augmented, sample_weights, correction_stats, tabddpm, n_original = stage2_label_correction_and_augmentation(
            backbone, X_train, y_train_noisy, y_train_clean, config, logger, stage2_mode=stage2_mode
        )
        # TabDDPM is already saved in stage2_label_correction_and_augmentation function
        if end_stage <= 2:
            logger.info("\n" + "="*70)
            logger.info("✅ 已完成到 Stage 2，按 end_stage 设置提前结束")
            logger.info("="*70)
            return backbone
    elif end_stage >= 3:
        # 跳过Stage 2：
        # - 如果是 Stage 3-only(start_stage==3)，按决策直接使用原始干净数据（不加载任何离线增强数据）
        # - 否则：兼容旧流程（如果存在 augmented_data.npz 则加载）
        augmented_data_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "augmented_data.npz")
        if int(start_stage) == 3 and X_train is not None:
            logger.info("\n" + "="*70)
            logger.info("✅ Stage 3-only: 已跳过 Stage 2（不使用 TabDDPM 离线增强数据）")
            logger.info("="*70)
            logger.info("")
            X_augmented = X_train
            y_augmented = y_train_clean
            sample_weights = np.ones(len(X_train))
            n_original = len(X_train)
            logger.info(f"✓ 使用原始干净数据: {len(X_train)} 个样本")
            if os.path.exists(augmented_data_path):
                logger.info(f"  (已忽略离线增强文件: {augmented_data_path})")
        elif os.path.exists(augmented_data_path):
            # 加载已有的增强数据（兼容旧流程）
            logger.info("📥 输入数据路径:")
            logger.info(f"  ✓ 增强数据: {augmented_data_path}")
            backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
            logger.info(f"  ✓ 骨干网络模型: {backbone_path}")
            logger.info("")
            logger.info(f"✓ 加载已有增强数据: {augmented_data_path}")
            data = np.load(augmented_data_path)
            X_augmented = data['X_augmented']
            y_augmented = data['y_augmented']
            sample_weights = data['sample_weights'] if 'sample_weights' in data else np.ones(len(X_augmented))
            # Get n_original from saved data
            if 'n_original' in data:
                n_original = int(data['n_original'])
            else:
                # Fallback: assume all data is original if metadata not found
                n_original = len(X_augmented)
                logger.warning("⚠️  未找到原始数据数量标记，假设所有数据均为原始数据")
        elif X_train is not None:
            # 使用原始数据（不做增强和矫正）
            logger.info("\n" + "="*70)
            logger.info("⚠️  跳过 Stage 2，使用原始数据（带噪声标签）")
            logger.info("="*70)
            logger.info("")
            
            # 如果还没有注入噪声，现在注入
            if y_train_noisy is None:
                logger.info(f"🔀 注入标签噪声 ({config.LABEL_NOISE_RATE*100:.0f}%)...")
                y_train_noisy, noise_mask = inject_label_noise(y_train_clean, config.LABEL_NOISE_RATE)
                logger.info(f"✓ 噪声标签创建完成: {noise_mask.sum()} 个标签被翻转")
                logger.info(f"  原始标签分布: 正常={(y_train_clean==0).sum()}, 恶意={(y_train_clean==1).sum()}")
                logger.info(f"  噪声标签分布: 正常={(y_train_noisy==0).sum()}, 恶意={(y_train_noisy==1).sum()}")
            
            X_augmented = X_train
            y_augmented = y_train_noisy
            sample_weights = np.ones(len(X_train))
            n_original = len(X_train)
            
            logger.info(f"✓ 使用原始数据: {len(X_train)} 个样本")
            logger.info("  ⚠️  注意：未进行标签矫正和数据增强，可能影响性能")
        else:
            logger.error(f"❌ 找不到增强数据: {augmented_data_path}")
            logger.error("   且未加载原始数据")
            logger.error("   请先运行 Stage 2 或从 Stage 1 开始训练")
            return
    
    # ========================
    # Stage 3: Fine-tune Classifier
    # ========================
    if end_stage >= 3 and start_stage <= 3:
        # 确定实际使用的backbone路径（用于记录元数据）
        if hasattr(args, 'backbone_path') and args.backbone_path:
            actual_backbone_path = args.backbone_path
        else:
            actual_backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
        
        classifier, finetune_history, optimal_threshold = stage3_finetune_classifier(
            backbone, X_augmented, y_augmented, sample_weights, config, logger, 
            n_original=n_original, backbone_path=actual_backbone_path
        )
    else:
        logger.info("\n" + "="*70)
        logger.info("⏭️  跳过 Stage 3（按 end_stage 设置）")
        logger.info("="*70)
        return backbone
    
    # Plot training history
    history_fig_path = os.path.join(config.CLASSIFICATION_DIR, "figures", "training_history.png")
    plot_training_history(finetune_history, history_fig_path)
    logger.info(f"  ✓ 训练历史图表: {history_fig_path}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("🎉 训练完成! Training Complete!")
    logger.info("="*70)
    logger.info("")
    logger.info("📊 训练总结:")
    logger.info(f"  Stage 1: 骨干网络预训练 - {config.PRETRAIN_EPOCHS} epochs")
    logger.info(f"  Stage 2: 标签矫正+数据增强 - 完成")
    logger.info(f"  Stage 3: 分类器微调 - {config.FINETUNE_EPOCHS} epochs")
    logger.info("")
    logger.info("📥 输入数据路径:")
    logger.info(f"  ✓ 训练数据: {config.BENIGN_TRAIN} (正常), {config.MALICIOUS_TRAIN} (恶意)")
    logger.info("")
    logger.info("📁 输出文件路径:")
    logger.info(f"  ✓ 特征提取: {config.FEATURE_EXTRACTION_DIR}")
    logger.info(f"    - 骨干网络: {os.path.join(config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth')}")
    logger.info(f"    - 训练特征: {os.path.join(config.FEATURE_EXTRACTION_DIR, 'models', 'train_features.npy')}")
    logger.info(f"  ✓ 标签矫正: {config.LABEL_CORRECTION_DIR}")
    logger.info(f"    - 矫正结果: {os.path.join(config.LABEL_CORRECTION_DIR, 'models', 'correction_results.npz')}")
    logger.info(f"  ✓ 数据增强: {config.DATA_AUGMENTATION_DIR}")
    logger.info(f"    - TabDDPM模型: {os.path.join(config.DATA_AUGMENTATION_DIR, 'models', 'tabddpm.pth')}")
    logger.info(f"    - 增强数据: {os.path.join(config.DATA_AUGMENTATION_DIR, 'models', 'augmented_data.npz')}")
    logger.info(f"  ✓ 分类器:   {config.CLASSIFICATION_DIR}")
    logger.info(f"    - 分类器模型: {os.path.join(config.CLASSIFICATION_DIR, 'models', 'classifier_final.pth')}")
    logger.info(f"    - 训练历史: {os.path.join(config.CLASSIFICATION_DIR, 'models', 'training_history.npz')}")
    logger.info("")
    logger.info("💡 下一步: 运行 test.py 评估模型性能")
    logger.info("="*70)
    
    return classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MEDAL-Lite model")
    parser.add_argument("--noise_rate", type=float, default=0.30, help="Label noise rate")
    parser.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3], 
                       help="Start from which stage (1=backbone pretrain, 2=label correction, 3=classifier finetune)")
    parser.add_argument("--end_stage", type=int, default=3, choices=[1, 2, 3],
                       help="End at which stage (1/2/3). Use end_stage=2 for Stage2-only run")
    parser.add_argument("--stage2_mode", type=str, default="standard", choices=["standard", "clean_augment_only"])
    parser.add_argument("--retrain_backbone", action="store_true",
                       help="Use randomly initialized backbone instead of loading pretrained weights")
    parser.add_argument("--backbone_path", type=str, default=None,
                       help="Path to specific backbone model (e.g., backbone_SimCLR_500.pth)")
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.noise_rate is not None:
        config.LABEL_NOISE_RATE = args.noise_rate
    
    main(args)

