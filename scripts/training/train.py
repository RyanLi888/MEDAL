"""
MEDAL-Lite 主训练脚本 (重构版)
==============================
实现完整的3阶段训练流程，使用config.py中的最优配置
"""
import sys
import os
from pathlib import Path

# 确保项目根目录在sys.path中
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import hashlib
import argparse
from datetime import datetime

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import (
    set_seed, setup_logger, inject_label_noise,
    calculate_metrics, print_metrics, save_checkpoint, find_optimal_threshold
)
from MoudleCode.utils.logging_utils import (
    log_section_header, log_subsection_header, log_key_value,
    log_stage_start, log_stage_end, log_input_paths, log_output_paths,
    log_training_config, log_data_stats, log_model_info, log_progress,
    log_epoch_metrics, log_early_stopping, log_final_summary
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, silhouette_score
from MoudleCode.utils.visualization import (
    plot_feature_space, plot_noise_correction_comparison, plot_training_history
)
from MoudleCode.preprocessing.pcap_parser import load_dataset
from MoudleCode.feature_extraction.backbone import (
    MicroBiMambaBackbone, SimMTMLoss, build_backbone
)
from MoudleCode.feature_extraction.traffic_augmentation import DualViewAugmentation
from MoudleCode.feature_extraction.instance_contrastive import (
    InstanceContrastiveLearning, HybridPretrainingLoss
)
from MoudleCode.utils.checkpoint import load_state_dict_shape_safe

# 导入预处理模块
try:
    from scripts.utils.preprocess import check_preprocessed_exists, load_preprocessed, preprocess_train
    from scripts.utils.preprocess import normalize_burstsize_inplace
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

from MoudleCode.label_correction.hybrid_court import HybridCourt
from MoudleCode.data_augmentation.tabddpm import TabDDPM
from MoudleCode.classification.dual_stream import MEDAL_Classifier, DualStreamLoss


def _rng_fingerprint_short() -> str:
    h = hashlib.sha256()
    try:
        h.update(repr(random.getstate()).encode('utf-8'))
    except Exception:
        h.update(b'py_random_error')
    try:
        ns = np.random.get_state()
        h.update(str(ns[0]).encode('utf-8'))
        h.update(np.asarray(ns[1], dtype=np.uint32).tobytes())
        h.update(str(ns[2]).encode('utf-8'))
        h.update(str(ns[3]).encode('utf-8'))
        h.update(str(ns[4]).encode('utf-8'))
    except Exception:
        h.update(b'numpy_random_error')
    try:
        h.update(torch.get_rng_state().detach().cpu().numpy().tobytes())
    except Exception:
        h.update(b'torch_cpu_rng_error')
    try:
        if torch.cuda.is_available():
            for s in torch.cuda.get_rng_state_all():
                h.update(s.detach().cpu().numpy().tobytes())
        else:
            h.update(b'no_cuda')
    except Exception:
        h.update(b'torch_cuda_rng_error')
    return h.hexdigest()[:16]


def _seed_snapshot() -> str:
    torch_seed = None
    try:
        torch_seed = int(torch.initial_seed())
    except Exception:
        torch_seed = None
    return (
        f"config.SEED={int(getattr(config, 'SEED', -1))} | "
        f"torch.initial_seed={torch_seed}"
    )


def stage1_pretrain_backbone(backbone, train_loader, config, logger):
    """
    Stage 1: 自监督预训练骨干网络 (SimMTM + InfoNCE)
    
    Args:
        backbone: MicroBiMambaBackbone 模型
        train_loader: 数据加载器（仅需X，无需标签）
        config: 配置对象
        logger: 日志记录器
        
    Returns:
        backbone: 预训练后的骨干网络
        history: 训练历史
    """
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✓ GPU 缓存已清理")
    
    # 输出阶段配置
    log_stage_start(logger, "STAGE 1: 自监督预训练骨干网络", "训练Micro-Bi-Mamba骨干网络，学习流量特征表示")
    config.log_stage_config(logger, "Stage 1")
    
    # 输出输入路径
    log_input_paths(logger, {
        "训练数据(正常)": config.BENIGN_TRAIN,
        "训练数据(恶意)": config.MALICIOUS_TRAIN
    })
    
    backbone.train()
    backbone.to(config.DEVICE)
    
    # 检查对比学习配置
    use_instance_contrastive = getattr(config, 'USE_INSTANCE_CONTRASTIVE', False)
    contrastive_method = getattr(config, 'CONTRASTIVE_METHOD', 'infonce')
    actual_batch_size = train_loader.batch_size
    
    # 创建增强器和损失函数
    if use_instance_contrastive:
        augmentation = DualViewAugmentation(config)
        instance_contrastive = InstanceContrastiveLearning(backbone, config).to(config.DEVICE)
        simmtm_loss_fn = SimMTMLoss(
            config,
            mask_rate=config.SIMMTM_MASK_RATE,
            noise_std=getattr(config, 'PRETRAIN_NOISE_STD', 0.05)
        )
        hybrid_loss_fn = HybridPretrainingLoss(
            simmtm_loss=simmtm_loss_fn,
            instance_contrastive=instance_contrastive,
            lambda_infonce=config.INFONCE_LAMBDA
        )
        logger.info(f"✓ 混合损失函数初始化完成 (SimMTM + {contrastive_method.upper()})")
        
        optimizer = optim.AdamW(
            list(backbone.parameters()) + list(instance_contrastive.projection_head.parameters()),
            lr=config.PRETRAIN_LR,
            weight_decay=config.PRETRAIN_WEIGHT_DECAY
        )
    else:
        simmtm_loss_fn = SimMTMLoss(
            config,
            mask_rate=config.SIMMTM_MASK_RATE,
            noise_std=getattr(config, 'PRETRAIN_NOISE_STD', 0.05)
        )
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
    
    # 训练历史
    if use_instance_contrastive:
        history = {'loss': [], 'simmtm': [], 'infonce': []}
    else:
        history = {'loss': [], 'simmtm': []}
    
    # 早停配置
    use_early_stopping = bool(getattr(config, 'PRETRAIN_EARLY_STOPPING', True))
    es_warmup_epochs = int(getattr(config, 'PRETRAIN_ES_WARMUP_EPOCHS', 50))
    es_patience = int(getattr(config, 'PRETRAIN_ES_PATIENCE', 20))
    es_min_delta = float(getattr(config, 'PRETRAIN_ES_MIN_DELTA', 0.01))

    best_loss = float('inf')
    best_epoch = -1
    best_state = None
    no_improve = 0
    
    # 梯度累积配置
    use_gradient_accumulation = (
        use_instance_contrastive and
        str(contrastive_method).lower() == 'nnclr' and
        int(getattr(config, 'PRETRAIN_GRADIENT_ACCUMULATION_STEPS', 1)) > 1
    )
    gradient_accumulation_steps = int(getattr(config, 'PRETRAIN_GRADIENT_ACCUMULATION_STEPS', 2)) if use_gradient_accumulation else 1
    
    log_subsection_header(logger, "开始训练")

    # 训练循环
    for epoch in range(config.PRETRAIN_EPOCHS):
        epoch_loss = 0.0
        epoch_simmtm = 0.0
        epoch_infonce = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, (list, tuple)):
                X_batch = batch_data[0]
            else:
                X_batch = batch_data
            X_batch = X_batch.to(config.DEVICE)
            
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            if use_instance_contrastive:
                x_view1, x_view2 = augmentation(X_batch)
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
                loss = simmtm_loss_fn(backbone, X_batch)
                epoch_simmtm += loss.item()
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
        
        scheduler.step()
        
        # 计算平均损失
        n_batches = len(train_loader)
        epoch_loss /= n_batches
        epoch_simmtm /= n_batches
        
        history['loss'].append(epoch_loss)
        history['simmtm'].append(epoch_simmtm)
        
        if use_instance_contrastive:
            epoch_infonce /= n_batches
            history['infonce'].append(epoch_infonce)
        
        # 输出日志
        progress = (epoch + 1) / config.PRETRAIN_EPOCHS * 100
        if use_instance_contrastive:
            method_name = str(contrastive_method).upper()
            logger.info(f"[Stage 1] Epoch [{epoch+1}/{config.PRETRAIN_EPOCHS}] ({progress:.1f}%) | "
                       f"Loss: {epoch_loss:.4f} | SimMTM: {epoch_simmtm:.4f} | "
                       f"{method_name}: {epoch_infonce:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        else:
            logger.info(f"[Stage 1] Epoch [{epoch+1}/{config.PRETRAIN_EPOCHS}] ({progress:.1f}%) | "
                       f"Loss: {epoch_loss:.4f} | SimMTM: {epoch_simmtm:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 早停检查
        improved = (best_loss - epoch_loss) > es_min_delta
        if improved:
            best_loss = float(epoch_loss)
            best_epoch = int(epoch + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if use_early_stopping and (epoch + 1) >= es_warmup_epochs and no_improve >= es_patience:
            log_early_stopping(logger, epoch+1, best_epoch, best_loss, epoch_loss, no_improve, es_patience)
            break

    # 恢复最佳状态
    if best_state is not None:
        load_state_dict_shape_safe(backbone, best_state, logger, prefix="backbone(best)")
        backbone.to(config.DEVICE)
    
    # 输出阶段总结
    log_stage_end(logger, "Stage 1", {
        "最终损失": f"{history['loss'][-1]:.4f}",
        "最佳损失": f"{best_loss:.4f} (epoch {best_epoch})" if best_epoch > 0 else "N/A",
        "训练轮数": len(history['loss'])
    })
    
    # 保存模型
    n_samples = len(train_loader.dataset)
    model_dim = config.MODEL_DIM
    if use_instance_contrastive:
        method_name = f"SimMTM_{str(contrastive_method).upper()}"
    else:
        method_name = "SimMTM"
    
    backbone_filename = f"backbone_{method_name}_{model_dim}d_{n_samples}.pth"
    backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", backbone_filename)
    torch.save(backbone.state_dict(), backbone_path)
    
    default_backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    torch.save(backbone.state_dict(), default_backbone_path)
    
    log_output_paths(logger, {
        "骨干网络模型": backbone_path,
        "默认副本": default_backbone_path
    })
    
    return backbone, history


def stage2_label_correction_and_augmentation(backbone, X_train, y_train_noisy, y_train_clean, config, logger, stage2_mode='standard'):
    """
    Stage 2: 标签矫正 + 数据增强
    
    Args:
        backbone: 预训练的骨干网络（冻结）
        X_train: (N, L, D) 训练序列
        y_train_noisy: (N,) 噪声标签
        y_train_clean: (N,) 干净标签（仅用于评估）
        config: 配置对象
        logger: 日志记录器
        stage2_mode: 'standard' 或 'clean_augment_only'
        
    Returns:
        Z_augmented: 增强后的特征
        y_augmented: 增强后的标签
        sample_weights: 样本权重
        correction_stats: 矫正统计
        tabddpm: TabDDPM模型
        n_original: 原始样本数
    """
    log_stage_start(logger, "STAGE 2: 标签矫正 + 数据增强", "矫正标签噪声并生成增强样本")
    config.log_stage_config(logger, "Stage 2")
    
    log_input_paths(logger, {
        "训练数据(正常)": config.BENIGN_TRAIN,
        "训练数据(恶意)": config.MALICIOUS_TRAIN,
        "骨干网络模型": os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    })
    
    # 冻结骨干网络并提取特征
    backbone.to(config.DEVICE)
    backbone.freeze()
    backbone.eval()
    logger.info("✓ 骨干网络已冻结，开始特征提取...")
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
        features_list = []
        batch_size = 64
        total_batches = (len(X_tensor) + batch_size - 1) // batch_size
        
        for i in range(0, len(X_tensor), batch_size):
            batch_idx = i // batch_size + 1
            X_batch = X_tensor[i:i+batch_size]
            z_batch = backbone(X_batch, return_sequence=False)
            features_list.append(z_batch.cpu().numpy())
            
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                log_progress(logger, batch_idx, total_batches, "特征提取")
        
        features = np.concatenate(features_list, axis=0)
    
    logger.info(f"✓ 特征提取完成: {features.shape}")
    
    # 保存特征
    features_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "train_features.npy")
    np.save(features_path, features)
    
    # 特征可视化
    feature_dist_path = os.path.join(config.LABEL_CORRECTION_DIR, "figures", "feature_distribution_stage2.png")
    plot_feature_space(features, y_train_clean, feature_dist_path,
                      title="Stage 2: Feature Distribution", method='tsne')
    
    # 标签矫正
    log_subsection_header(logger, "步骤 2.1: Hybrid Court 标签矫正")
    logger.info(f"  输入: {len(y_train_noisy)} 个样本，噪声率: {config.LABEL_NOISE_RATE*100:.0f}%")
    logger.info(f"  方法: CL (置信学习) + MADE (密度估计) + KNN (语义投票)")
    
    hybrid_court = HybridCourt(config)

    if stage2_mode == 'clean_augment_only':
        # 跳过标签矫正，直接使用干净标签
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

    # drop 不删除数据：将 drop 样本归类为 reweight（低权重）
    try:
        drop_mask = (action_mask == 2)
        if hasattr(correction_weight, '__len__') and int(drop_mask.sum()) > 0:
            correction_weight = correction_weight.astype(np.float32, copy=True)
            drop_reweight = float(getattr(config, 'STAGE2_DROP_AS_REWEIGHT_WEIGHT', 0.1))
            correction_weight[drop_mask] = np.minimum(correction_weight[drop_mask], drop_reweight)
            try:
                action_mask = np.asarray(action_mask).copy()
                action_mask[drop_mask] = 3
            except Exception:
                pass
            logger.info(f"🧹 drop样本不删除：已将其归类为reweight并设为低权重 (count={int(drop_mask.sum())}, w={drop_reweight})")
    except Exception:
        pass
    
    # 保存矫正结果
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
    
    # 计算矫正统计
    keep_mask = action_mask != 2
    correction_accuracy = (y_corrected[keep_mask] == y_train_clean[keep_mask]).mean()
    
    correction_stats = {
        'accuracy': correction_accuracy,
        'n_keep': (action_mask == 0).sum(),
        'n_flip': (action_mask == 1).sum(),
        'n_drop': (action_mask == 2).sum(),
        'n_reweight': (action_mask == 3).sum()
    }
    
    log_data_stats(logger, {
        "矫正准确率": f"{correction_accuracy*100:.2f}%",
        "保持样本": correction_stats['n_keep'],
        "翻转样本": correction_stats['n_flip'],
        "丢弃样本": correction_stats['n_drop'],
        "重加权样本": correction_stats['n_reweight']
    }, "标签矫正统计")

    # Stage2 输出：保留全部原始样本（包括 drop，但其权重=0）
    X_all = X_train
    y_all = y_corrected
    weights_all = correction_weight
    Z_all = features

    # 保存原始序列用于Stage 3混合训练（保留全部原始样本，drop 的权重为0）
    try:
        real_kept_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "real_kept_data.npz")
        np.savez(real_kept_path, X_real=X_all, y_real=y_all, sample_weights_real=weights_all)
        logger.info(f"  ✓ 已保存原始序列: {real_kept_path}")
    except Exception as e:
        logger.warning(f"⚠ 无法保存原始序列: {e}")
    
    stage2_use_tabddpm = bool(getattr(config, 'STAGE2_USE_TABDDPM', True))

    if not stage2_use_tabddpm:
        logger.info("")

        logger.info("步骤 2.2: TabDDPM 数据增强（已跳过）")
        Z_augmented = Z_all
        y_augmented = y_all
        sample_weights = weights_all
        n_train_original = int(Z_all.shape[0])
        return Z_augmented, y_augmented, sample_weights, correction_stats, None, n_train_original
    
    # TabDDPM 数据增强
    log_subsection_header(logger, "步骤 2.2: TabDDPM 数据增强 (Feature Space)")
    logger.info(f"  目标: 在骨干网络特征空间中训练/生成")

    # 数据增强仅使用高权重数据（来自标签矫正权重）
    try:
        aug_min_w = float(getattr(config, 'STAGE2_AUGMENT_MIN_WEIGHT', getattr(config, 'STAGE2_FEATURE_TIER2_MIN_WEIGHT', 0.7)))
    except Exception:
        aug_min_w = 0.7
    aug_mask = (np.asarray(weights_all) >= float(aug_min_w))
    Z_clean = Z_all[aug_mask]
    y_clean = np.asarray(y_all)[aug_mask]
    weights_clean = np.asarray(weights_all)[aug_mask]
    logger.info(f"🧪 TabDDPM训练/增强仅使用高权重样本: {int(aug_mask.sum())}/{len(Z_all)} (threshold={aug_min_w})")
    try:
        if len(weights_clean) > 0:
            logger.info(
                f"🧪 TabDDPM训练集权重统计: min={float(np.min(weights_clean)):.4f}, mean={float(np.mean(weights_clean)):.4f}, max={float(np.max(weights_clean)):.4f}"
            )
    except Exception:
        pass
    
    logger.info(f"🔧 RNG指纹(Stage2-TabDDPM训练前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    
    tabddpm = TabDDPM(
        config,
        input_dim=Z_clean.shape[1],
        cond_indices=[],
        dep_indices=list(range(Z_clean.shape[1])),
        enable_protocol_constraints=False
    ).to(config.DEVICE)

    tabddpm.fit_scaler(Z_clean)

    ddpm_lr = float(getattr(config, 'DDPM_LR', 5e-4))
    optimizer_ddpm = optim.AdamW(tabddpm.parameters(), lr=ddpm_lr)
    
    # 学习率调度器（v2.3新增）
    ddpm_lr_scheduler_type = getattr(config, 'DDPM_LR_SCHEDULER', None)
    ddpm_scheduler = None
    if ddpm_lr_scheduler_type == 'cosine':
        ddpm_min_lr = float(getattr(config, 'DDPM_MIN_LR', 1e-5))
        n_epochs_ddpm = int(getattr(config, 'DDPM_EPOCHS', 100))
        ddpm_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_ddpm, T_max=n_epochs_ddpm, eta_min=ddpm_min_lr
        )
        logger.info(f"✓ TabDDPM学习率调度器: Cosine Annealing (lr={ddpm_lr} → {ddpm_min_lr})")
    else:
        n_epochs_ddpm = int(getattr(config, 'DDPM_EPOCHS', 100))
    
    ddpm_use_early_stopping = bool(getattr(config, 'DDPM_EARLY_STOPPING', True))
    ddpm_es_warmup_epochs = int(getattr(config, 'DDPM_ES_WARMUP_EPOCHS', 20))
    ddpm_es_patience = int(getattr(config, 'DDPM_ES_PATIENCE', 30))
    ddpm_es_min_delta = float(getattr(config, 'DDPM_ES_MIN_DELTA', 0.001))
    ddpm_es_smooth_window = int(getattr(config, 'DDPM_ES_SMOOTH_WINDOW', 5))

    dataset_ddpm = TensorDataset(torch.FloatTensor(Z_clean), torch.LongTensor(y_clean))
    logger.info(f"🔧 RNG指纹(Stage2-TabDDPM DataLoader创建前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    loader_ddpm = DataLoader(dataset_ddpm, batch_size=2048, shuffle=True)
    logger.info(f"🔧 RNG指纹(Stage2-TabDDPM DataLoader创建后): {_rng_fingerprint_short()} ({_seed_snapshot()})")

    ddpm_best_loss = float('inf')
    ddpm_best_epoch = -1
    ddpm_best_state = None
    ddpm_no_improve = 0
    ddpm_loss_history = []  # 用于平滑窗口的损失历史

    tabddpm.train()
    for epoch in range(n_epochs_ddpm):
        epoch_loss = 0.0
        for Z_batch, y_batch in loader_ddpm:
            z_0 = Z_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)
            optimizer_ddpm.zero_grad()
            loss = tabddpm.compute_loss(
                z_0, y_batch,
                mask_prob=float(getattr(config, 'MASK_PROBABILITY', 0.5)),
                mask_lambda=float(getattr(config, 'MASK_LAMBDA', 0.1)),
                p_uncond=0.2
            )
            loss.backward()
            optimizer_ddpm.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(len(loader_ddpm), 1)
        ddpm_loss_history.append(avg_loss)
        
        # 计算平滑损失（移动平均）
        if len(ddpm_loss_history) >= ddpm_es_smooth_window:
            smoothed_loss = float(np.mean(ddpm_loss_history[-ddpm_es_smooth_window:]))
        else:
            smoothed_loss = avg_loss
        
        # 学习率调度（v2.3新增）
        if ddpm_scheduler is not None:
            ddpm_scheduler.step()
            current_lr = optimizer_ddpm.param_groups[0]['lr']
            if (epoch + 1) % 100 == 0:  # 每100轮记录一次学习率
                logger.info(f"[TabDDPM] Epoch [{epoch+1}/{n_epochs_ddpm}] | Loss: {avg_loss:.4f} | Smoothed: {smoothed_loss:.4f} | LR: {current_lr:.6f}")
            else:
                logger.info(f"[TabDDPM] Epoch [{epoch+1}/{n_epochs_ddpm}] | Loss: {avg_loss:.4f} | Smoothed: {smoothed_loss:.4f}")
        else:
            logger.info(f"[TabDDPM] Epoch [{epoch+1}/{n_epochs_ddpm}] | Loss: {avg_loss:.4f} | Smoothed: {smoothed_loss:.4f}")

        # 使用平滑损失进行早停判断
        improved = (ddpm_best_loss - smoothed_loss) > ddpm_es_min_delta
        if improved:
            ddpm_best_loss = smoothed_loss
            ddpm_best_epoch = int(epoch + 1)
            ddpm_best_state = {k: v.detach().cpu().clone() for k, v in tabddpm.state_dict().items()}
            ddpm_no_improve = 0
        else:
            ddpm_no_improve += 1

        if ddpm_use_early_stopping and (epoch + 1) >= ddpm_es_warmup_epochs and ddpm_no_improve >= ddpm_es_patience:
            log_early_stopping(logger, epoch+1, ddpm_best_epoch, ddpm_best_loss, smoothed_loss, ddpm_no_improve, ddpm_es_patience)
            break

    if ddpm_best_state is not None:
        tabddpm.load_state_dict(ddpm_best_state)
        tabddpm.to(config.DEVICE)

    # 生成增强特征
    logger.info("步骤 2.3: 生成增强特征")
    Z_syn, y_syn, w_syn = tabddpm.augment_feature_dataset(Z_clean, y_clean, weights_clean)

    n_train_original = int(Z_all.shape[0])
    n_syn = int(Z_syn.shape[0] - int(Z_clean.shape[0]))
    if n_syn > 0:
        Z_syn_only = Z_syn[-n_syn:]
        y_syn_only = y_syn[-n_syn:]
        w_syn_only = np.ones((len(y_syn_only),), dtype=np.float32)
    else:
        Z_syn_only = np.zeros((0, Z_all.shape[1]), dtype=Z_all.dtype)
        y_syn_only = np.zeros((0,), dtype=np.asarray(y_all).dtype)
        w_syn_only = np.zeros((0,), dtype=np.asarray(weights_all).dtype)

    Z_augmented = np.concatenate([Z_all, Z_syn_only], axis=0)
    y_augmented = np.concatenate([np.asarray(y_all), np.asarray(y_syn_only)], axis=0)
    sample_weights = np.concatenate([np.asarray(weights_all, dtype=np.float32), np.asarray(w_syn_only, dtype=np.float32)], axis=0)
    logger.info(f"✓ 特征增强完成: 原始={n_train_original} + 合成={len(Z_syn_only)} → 总计={len(Z_augmented)}")

    # 保存模型和数据
    tabddpm_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "tabddpm_feature.pth")
    torch.save(tabddpm.state_dict(), tabddpm_path)
    
    augmented_data_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "augmented_features.npz")
    is_original_mask = np.zeros(len(Z_augmented), dtype=bool)
    is_original_mask[:n_train_original] = True
    np.savez(augmented_data_path, Z_augmented=Z_augmented, y_augmented=y_augmented,
             is_original=is_original_mask, n_original=n_train_original, sample_weights=sample_weights)
    
    log_output_paths(logger, {
        "矫正结果": correction_results_path,
        "TabDDPM模型": tabddpm_path,
        "增强特征": augmented_data_path
    })
    
    log_stage_end(logger, "Stage 2", {
        "原始样本": n_train_original,
        "增强后样本": len(Z_augmented),
        "合成样本": n_syn
    })

    return Z_augmented, y_augmented, sample_weights, correction_stats, tabddpm, n_train_original


def stage3_finetune_classifier(backbone, X_train, y_train, sample_weights, config, logger, 
                               n_original=None, backbone_path=None, X_train_real=None, use_mixed_stream=None):
    """
    Stage 3: 分类器微调
    
    Args:
        backbone: 预训练的骨干网络
        X_train: (N, L, D) 训练序列 或 (N, D) 增强特征
        y_train: (N,) 训练标签
        sample_weights: (N,) 样本权重
        config: 配置对象
        logger: 日志记录器
        n_original: 原始样本数
        backbone_path: 骨干网络路径
        X_train_real: 原始序列（用于混合训练）
        use_mixed_stream: 是否使用混合训练
        
    Returns:
        classifier: 训练好的分类器
        history: 训练历史
        optimal_threshold: 最优阈值
    """
    log_stage_start(logger, "STAGE 3: 分类器微调", "训练双流MLP分类器进行最终威胁检测")
    config.log_stage_config(logger, "Stage 3")
    
    # 确定输入类型
    if use_mixed_stream is None:
        use_mixed_stream = bool(getattr(config, 'STAGE3_MIXED_STREAM', False))
    
    has_real_sequences = X_train_real is not None and len(X_train_real) > 0
    
    # 自动检测输入是特征还是序列
    input_is_features = bool(getattr(config, 'CLASSIFIER_INPUT_IS_FEATURES', False))
    try:
        if hasattr(X_train, 'ndim'):
            if int(X_train.ndim) == 2:
                input_is_features = True
            elif int(X_train.ndim) == 3:
                input_is_features = False
    except Exception:
        pass

    # 显示数据信息
    logger.info("")
    logger.info("📊 训练数据信息:")
    logger.info(f"  - 数据形状: {X_train.shape}")
    logger.info(f"  - 样本总数: {len(X_train)}")
    if n_original is not None and n_original != len(X_train):
        logger.info(f"  - 原始样本: {n_original}")
        logger.info(f"  - 增强样本: {len(X_train) - n_original}")
    logger.info(f"  - 正常样本: {(y_train == 0).sum()}")
    logger.info(f"  - 恶意样本: {(y_train == 1).sum()}")
    if input_is_features:
        logger.info(f"  - 输入类型: 特征向量 (2D)")
    else:
        logger.info(f"  - 输入类型: 原始序列 (3D)")
    logger.info("")

    # 原始样本：权重>0.8 的做有标签监督训练（样本权重=2.0）；<=0.8 的进入无标签半监督训练
    sw = np.asarray(sample_weights, dtype=np.float32) if hasattr(sample_weights, '__len__') else None
    sup_thr = float(getattr(config, 'STAGE3_SUP_WEIGHT_THRESHOLD', 0.8))
    real_sup_weight = float(getattr(config, 'STAGE3_REAL_SUP_WEIGHT', 2.0))
    syn_weight = float(getattr(config, 'STAGE3_SYN_WEIGHT', 1.0))
    unlabeled_scale = float(getattr(config, 'STAGE3_UNLABELED_LOSS_SCALE', 1.0))

    orig_n = 0
    if n_original is not None:
        try:
            orig_n = min(int(n_original), int(len(y_train)))
        except Exception:
            orig_n = 0

    orig_sup_mask = None
    orig_unlab_mask = None
    if sw is not None and orig_n > 0:
        orig_sup_mask = (sw[:orig_n] > sup_thr)
        orig_unlab_mask = ~orig_sup_mask
        logger.info(f"� Stage3 原始样本监督/无标签划分: supervised={int(orig_sup_mask.sum())}, unlabeled={int(orig_unlab_mask.sum())}, threshold={sup_thr}")

    # 混合训练模式检查
    if use_mixed_stream and not has_real_sequences:
        logger.info("⚠️ 混合训练需要额外的原始序列(X_train_real)，当前未提供")
        use_mixed_stream = False
    
    if use_mixed_stream and not input_is_features:
        logger.info("⚠️ 混合训练需要增强特征作为主输入，当前输入是原始序列")
        use_mixed_stream = False

    finetune_backbone_requested = bool(getattr(config, 'FINETUNE_BACKBONE', False))
    
    # 特征输入时不能微调骨干网络
    if input_is_features and not use_mixed_stream:
        if finetune_backbone_requested:
            logger.info("⚠️ 输入为特征向量，骨干微调将被禁用（无法反向传播到骨干网络）")
        finetune_backbone_requested = False
    
    if use_mixed_stream:
        finetune_backbone_requested = True
        logger.info("🔀 混合训练模式已启用")
        log_data_stats(logger, {
            "原始序列": f"{len(X_train_real)} 个样本",
            "增强特征": f"{len(X_train)} 个样本",
            "原始序列批次": config.STAGE3_MIXED_REAL_BATCH_SIZE,
            "增强特征批次": config.STAGE3_MIXED_SYN_BATCH_SIZE
        }, "混合训练配置")
    
    finetune_scope = str(getattr(config, 'FINETUNE_BACKBONE_SCOPE', 'projection')).lower()
    finetune_backbone_lr = float(getattr(config, 'FINETUNE_BACKBONE_LR', 1e-5))
    finetune_backbone_warmup_epochs = int(getattr(config, 'FINETUNE_BACKBONE_WARMUP_EPOCHS', 0))
    
    if backbone_path is None:
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    
    log_input_paths(logger, {
        "增强特征": os.path.join(config.DATA_AUGMENTATION_DIR, "models", "augmented_features.npz"),
        "骨干网络模型": backbone_path
    })

    if input_is_features:
        config.CLASSIFIER_INPUT_IS_FEATURES = True

    # 创建分类器
    classifier = MEDAL_Classifier(backbone, config).to(config.DEVICE)
    criterion = DualStreamLoss(config)
    logger.info("✓ 双流分类器创建完成")
    logger.info(f"🔧 FocalLoss: alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA}")

    # 骨干网络微调策略
    backbone_param_candidates = []
    if finetune_backbone_requested:
        if finetune_scope == 'all':
            backbone_param_candidates = list(backbone.parameters())
        elif finetune_scope == 'projection':
            if hasattr(backbone, 'projection'):
                backbone_param_candidates = list(backbone.projection.parameters())
            else:
                logger.warning("⚠️ backbone没有projection层，回退到冻结模式")
                finetune_backbone_requested = False

    backbone_finetune_active = False
    backbone_finetune_started = False

    def _set_backbone_trainable(enable: bool) -> None:
        nonlocal backbone_finetune_active, backbone_finetune_started, finetune_backbone_requested
        backbone.freeze()
        if not enable or not finetune_backbone_requested:
            backbone.eval()
            backbone_finetune_active = False
            return

        if finetune_scope == 'all':
            backbone.unfreeze()
        elif finetune_scope == 'projection':
            if hasattr(backbone, 'projection'):
                for p in backbone.projection.parameters():
                    p.requires_grad = True

        backbone.train()
        backbone_finetune_active = True
        backbone_finetune_started = True

    if finetune_backbone_requested and finetune_backbone_warmup_epochs > 0:
        _set_backbone_trainable(False)
        logger.info(f"🧊 骨干网络前{finetune_backbone_warmup_epochs}轮冻结，之后微调(scope={finetune_scope})")
    elif finetune_backbone_requested:
        _set_backbone_trainable(True)
        logger.info(f"🔥 骨干网络微调已启用 (scope={finetune_scope})")
    else:
        _set_backbone_trainable(False)

    # 优化器
    param_groups = [{'params': classifier.dual_mlp.parameters(), 'lr': float(config.FINETUNE_LR)}]
    if finetune_backbone_requested and len(backbone_param_candidates) > 0:
        param_groups.append({'params': backbone_param_candidates, 'lr': finetune_backbone_lr})
        logger.info(f"✓ 优化器包含骨干网络参数: {len(backbone_param_candidates)} | lr={finetune_backbone_lr}")

    optimizer = optim.AdamW(param_groups, weight_decay=config.PRETRAIN_WEIGHT_DECAY)
    logger.info("✓ 优化器初始化完成")
    
    # 验证集划分
    val_split = float(getattr(config, 'FINETUNE_VAL_SPLIT', 0.0))
    X_val_split = None
    y_val_split = None
    sample_weights_val_split = None

    if val_split > 0:
        X_train_split, X_val_split, y_train_split, y_val_split, sample_weights_split, sample_weights_val_split = train_test_split(
            X_train, y_train, sample_weights, test_size=val_split,
            random_state=int(getattr(config, 'SEED', 42)), stratify=y_train
        )
        logger.info(f"📊 训练集: {len(X_train_split)}, 验证集: {len(X_val_split)}")
    else:
        X_train_split = X_train
        y_train_split = y_train
        sample_weights_split = sample_weights
        logger.info("📊 全量训练（不划分验证集）")
    
    # 数据加载器
    # 构建监督/无标签数据（支持 2D 特征 或 mixed-stream 的 3D 原始序列）
    def _sym_kl(p, q):
        p = p.clamp_min(1e-8)
        q = q.clamp_min(1e-8)
        return (p * (p.log() - q.log())).sum(dim=1) + (q * (q.log() - p.log())).sum(dim=1)

    X_np = X_train_split
    y_np = np.asarray(y_train_split)
    sw_np = np.asarray(sample_weights_split, dtype=np.float32)

    # 合成样本（默认位于尾部：原始n_original + syn_only）
    if orig_n > 0:
        syn_idx_start = orig_n
    else:
        syn_idx_start = 0

    syn_mask = np.zeros(len(sw_np), dtype=bool)
    if syn_idx_start < len(sw_np):
        syn_mask[syn_idx_start:] = True

    sup_mask = syn_mask.copy()
    unlab_mask = np.zeros(len(sw_np), dtype=bool)
    if orig_n > 0 and orig_sup_mask is not None:
        sup_mask[:orig_n] = orig_sup_mask
        unlab_mask[:orig_n] = orig_unlab_mask
    else:
        # 无 n_original 信息时：按权重阈值划分（保守）
        if sw is not None:
            sup_mask = (sw_np > sup_thr)
            unlab_mask = ~sup_mask

    # 监督样本权重：原始高权重=2.0，合成=1.0
    sw_sup = np.ones(int(sup_mask.sum()), dtype=np.float32) * syn_weight
    try:
        if orig_n > 0 and orig_sup_mask is not None:
            n_orig_sup = int(orig_sup_mask.sum())
            if n_orig_sup > 0:
                sw_sup[:n_orig_sup] = real_sup_weight
    except Exception:
        pass

    X_sup = X_np[sup_mask]
    y_sup = y_np[sup_mask]

    train_dataset = TensorDataset(
        torch.FloatTensor(X_sup),
        torch.LongTensor(y_sup),
        torch.FloatTensor(sw_sup)
    )

    unlab_dataset = None
    if int(unlab_mask.sum()) > 0:
        X_unlab = X_np[unlab_mask]
        unlab_dataset = TensorDataset(torch.FloatTensor(X_unlab))
    
    # 平衡采样（基于 supervised 数据集）
    use_balanced_sampling = getattr(config, 'USE_BALANCED_SAMPLING', True)
    if use_balanced_sampling:
        from torch.utils.data import WeightedRandomSampler
        target_ratio = float(getattr(config, 'BALANCED_SAMPLING_RATIO', 1.0))
        try:
            class_counts = np.bincount(np.asarray(y_sup, dtype=int))
            if len(class_counts) < 2:
                class_counts = np.pad(class_counts, (0, 2 - len(class_counts)), constant_values=0)
            weight_benign = target_ratio * class_counts[1] / max(class_counts[0], 1)
            weight_malicious = 1.0
            class_weights = np.array([weight_benign, weight_malicious])
            sample_sampling_weights = class_weights[np.asarray(y_sup, dtype=int)]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_sampling_weights, dtype=torch.double),
                num_samples=len(sample_sampling_weights), replacement=True
            )
            train_loader = DataLoader(train_dataset, batch_size=config.FINETUNE_BATCH_SIZE, sampler=sampler)
            logger.info(f"🌡️ 温室训练: 平衡采样 (目标比例 {target_ratio}:1)")
        except Exception:
            train_loader = DataLoader(train_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True)
    
    # 混合训练数据加载器（原始序列分为 supervised / unlabeled）
    real_loader = None
    unlab_real_loader = None
    if use_mixed_stream and has_real_sequences:
        n_real = len(X_train_real)
        real_n = min(int(orig_n), int(n_real)) if orig_n > 0 else min(len(y_train_split), int(n_real))

        y_real_all = y_np[:real_n]
        if orig_sup_mask is None or orig_unlab_mask is None or real_n == 0:
            real_sup_sel = np.ones(real_n, dtype=bool)
            real_unlab_sel = np.zeros(real_n, dtype=bool)
        else:
            real_sup_sel = orig_sup_mask[:real_n]
            real_unlab_sel = orig_unlab_mask[:real_n]

        X_real_sup = X_train_real[:real_n][real_sup_sel]
        y_real_sup = y_real_all[real_sup_sel]
        w_real_sup = np.ones(len(y_real_sup), dtype=np.float32) * real_sup_weight

        real_dataset = TensorDataset(
            torch.FloatTensor(X_real_sup),
            torch.LongTensor(y_real_sup),
            torch.FloatTensor(w_real_sup)
        )
        real_batch_size = int(getattr(config, 'STAGE3_MIXED_REAL_BATCH_SIZE', 32))
        real_loader = DataLoader(real_dataset, batch_size=real_batch_size, shuffle=True)

        if int(real_unlab_sel.sum()) > 0:
            X_real_unlab = X_train_real[:real_n][real_unlab_sel]
            unlab_real_dataset = TensorDataset(torch.FloatTensor(X_real_unlab))
            unlab_real_loader = DataLoader(unlab_real_dataset, batch_size=real_batch_size, shuffle=True)

        syn_batch_size = int(getattr(config, 'STAGE3_MIXED_SYN_BATCH_SIZE', 96))
        train_loader = DataLoader(train_dataset, batch_size=syn_batch_size, shuffle=True)
        
        logger.info(f"✓ 混合训练加载器: 原始={len(real_loader)}批, 增强={len(train_loader)}批")
        if unlab_real_loader is not None:
            logger.info(f"✓ 无标签原始加载器: {len(unlab_real_loader)}批")
    
    logger.info(f"✓ 数据加载器准备完成 ({len(train_loader)} 个批次)")

    unlab_iter = None

    # 训练历史
    history = {
        'train_loss': [], 'supervision': [], 'train_f1': [],
        'val_f1': [], 'val_threshold': [],
        'train_f1_star': [], 'train_threshold_star': []
    }
    
    classifier.train()
    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    best_threshold = float(getattr(config, 'MALICIOUS_THRESHOLD', 0.5))
    finetuned_backbone_path = None

    keep_last_k_minloss = int(getattr(config, 'KEEP_LAST_K_MINLOSS_EPOCHS', 10))
    last_k_loss_states = []
    
    # 早停配置
    use_early_stopping = bool(getattr(config, 'FINETUNE_EARLY_STOPPING', False))
    es_warmup_epochs = int(getattr(config, 'FINETUNE_ES_WARMUP_EPOCHS', 30))
    es_patience = int(getattr(config, 'FINETUNE_ES_PATIENCE', 25))
    es_min_delta = float(getattr(config, 'FINETUNE_ES_MIN_DELTA', 0.0))  # 改为0.0，使用纯耐心值策略
    no_improve_count = 0
    best_es_metric = -1.0
    allow_train_metric_es = bool(getattr(config, 'FINETUNE_ES_ALLOW_TRAIN_METRIC', False))
    
    if use_early_stopping and X_val_split is None and not allow_train_metric_es:
        use_early_stopping = False
        logger.info("⚠️ 早停已禁用：无验证集且不允许使用训练集指标")
    
    real_loss_scale = float(getattr(config, 'STAGE3_MIXED_REAL_LOSS_SCALE', 2.0))
    syn_loss_scale = float(getattr(config, 'STAGE3_MIXED_SYN_LOSS_SCALE', 1.0))
    
    log_subsection_header(logger, "开始训练")
    
    for epoch in range(config.FINETUNE_EPOCHS):
        # 骨干网络微调激活
        if finetune_backbone_requested and (not backbone_finetune_active) and (epoch >= finetune_backbone_warmup_epochs):
            _set_backbone_trainable(True)
            logger.info(f"🔥 骨干网络微调在epoch {epoch+1}激活 (scope={finetune_scope})")
        
        epoch_loss = 0.0
        epoch_losses = {'total': 0.0, 'supervision': 0.0, 'stream_a': 0.0, 'stream_b': 0.0}
        all_train_probs = []
        all_train_labels = []
        
        if use_mixed_stream and real_loader is not None:
            import itertools
            max_batches = max(len(real_loader), len(train_loader))
            real_iter = itertools.cycle(real_loader)
            syn_iter = itertools.cycle(train_loader)
            unlab_iter = itertools.cycle(unlab_real_loader) if unlab_real_loader is not None else None
            
            for batch_idx in range(max_batches):
                X_real, y_real, w_real = next(real_iter)
                X_real = X_real.to(config.DEVICE)
                y_real = y_real.to(config.DEVICE)
                w_real = w_real.to(config.DEVICE)
                
                Z_syn, y_syn, w_syn = next(syn_iter)
                Z_syn = Z_syn.to(config.DEVICE)
                y_syn = y_syn.to(config.DEVICE)
                w_syn = w_syn.to(config.DEVICE)
                
                optimizer.zero_grad()
                
                if backbone_finetune_active:
                    z_real = backbone(X_real, return_sequence=False)
                else:
                    with torch.no_grad():
                        z_real = backbone(X_real, return_sequence=False)
                
                loss_real, loss_dict_real = criterion(classifier.dual_mlp, z_real, y_real, w_real, epoch, config.FINETUNE_EPOCHS)
                loss_syn, loss_dict_syn = criterion(classifier.dual_mlp, Z_syn, y_syn, w_syn, epoch, config.FINETUNE_EPOCHS)

                loss_unlab = 0.0
                if unlab_iter is not None and unlabeled_scale > 0:
                    (X_unlab_real,) = next(unlab_iter)
                    X_unlab_real = X_unlab_real.to(config.DEVICE)
                    if backbone_finetune_active:
                        z_unlab = backbone(X_unlab_real, return_sequence=False)
                    else:
                        with torch.no_grad():
                            z_unlab = backbone(X_unlab_real, return_sequence=False)
                    logits_a, logits_b = classifier.dual_mlp(z_unlab, return_separate=True)
                    p_a = torch.softmax(logits_a, dim=1)
                    p_b = torch.softmax(logits_b, dim=1)
                    loss_unlab = _sym_kl(p_a, p_b).mean()

                loss = real_loss_scale * loss_real + syn_loss_scale * loss_syn + unlabeled_scale * loss_unlab
                loss.backward()
                optimizer.step()
                
                epoch_loss += float(loss.item())
                epoch_losses['total'] += float(real_loss_scale * loss_dict_real['total'] + syn_loss_scale * loss_dict_syn['total'])
                epoch_losses['supervision'] += float(real_loss_scale * loss_dict_real['supervision'] + syn_loss_scale * loss_dict_syn['supervision'])
                epoch_losses['stream_a'] += float(real_loss_scale * loss_dict_real['stream_a'] + syn_loss_scale * loss_dict_syn['stream_a'])
                epoch_losses['stream_b'] += float(real_loss_scale * loss_dict_real['stream_b'] + syn_loss_scale * loss_dict_syn['stream_b'])
                
                with torch.no_grad():
                    logits_a_syn, logits_b_syn = classifier.dual_mlp(Z_syn, return_separate=True)
                    logits_avg_syn = (logits_a_syn + logits_b_syn) / 2.0
                    probs_syn = torch.softmax(logits_avg_syn, dim=1)
                    all_train_probs.append(probs_syn.cpu().numpy())
                    all_train_labels.append(y_syn.cpu().numpy())
            
            n_batches = max_batches
        else:
            for X_batch, y_batch, w_batch in train_loader:
                X_batch = X_batch.to(config.DEVICE)
                y_batch = y_batch.to(config.DEVICE)
                w_batch = w_batch.to(config.DEVICE)

                optimizer.zero_grad()

                if input_is_features:
                    z = X_batch
                else:
                    if backbone_finetune_active:
                        z = backbone(X_batch, return_sequence=False)
                    else:
                        with torch.no_grad():
                            z = backbone(X_batch, return_sequence=False)

                loss, loss_dict = criterion(classifier.dual_mlp, z, y_batch, w_batch, epoch, config.FINETUNE_EPOCHS)

                # 无标签半监督：对低权重原始样本做双头一致性（仅特征模式有效）
                loss_unlab = 0.0
                if unlab_dataset is not None and unlabeled_scale > 0:
                    try:
                        (X_unlab_feat,) = next(unlab_iter)
                    except Exception:
                        unlab_iter = iter(DataLoader(unlab_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True))
                        (X_unlab_feat,) = next(unlab_iter)
                    X_unlab_feat = X_unlab_feat.to(config.DEVICE)
                    logits_a, logits_b = classifier.dual_mlp(X_unlab_feat, return_separate=True)
                    p_a = torch.softmax(logits_a, dim=1)
                    p_b = torch.softmax(logits_b, dim=1)
                    loss_unlab = _sym_kl(p_a, p_b).mean()

                (loss + unlabeled_scale * loss_unlab).backward()
                optimizer.step()

                epoch_loss += float(loss_dict['total'])
                epoch_losses['total'] += float(loss_dict['total'])
                epoch_losses['supervision'] += float(loss_dict['supervision'])
                epoch_losses['stream_a'] += float(loss_dict.get('stream_a', 0.0))
                epoch_losses['stream_b'] += float(loss_dict.get('stream_b', 0.0))

                with torch.no_grad():
                    logits_a, logits_b = classifier.dual_mlp(z, return_separate=True)
                    logits_avg = (logits_a + logits_b) / 2.0
                    probs = torch.softmax(logits_avg, dim=1)
                    all_train_probs.append(probs.cpu().numpy())
                    all_train_labels.append(y_batch.cpu().numpy())
            
            n_batches = int(max(len(train_loader), 1))

        epoch_loss /= n_batches
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        # 记录最后K轮的loss最小模型（仅用于训练结束后选取一个保存）
        if keep_last_k_minloss > 0:
            current_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
            last_k_loss_states.append((int(epoch + 1), float(epoch_loss), current_state))
            if len(last_k_loss_states) > keep_last_k_minloss:
                last_k_loss_states = last_k_loss_states[-keep_last_k_minloss:]
        
        # 计算训练集F1
        train_probs = np.concatenate(all_train_probs)
        train_labels = np.concatenate(all_train_labels)
        train_preds = (train_probs[:, 1] >= 0.5).astype(int)
        train_f1 = f1_score(train_labels, train_preds, pos_label=1, zero_division=0)

        monitor_pr_auc = bool(getattr(config, 'MONITOR_PR_AUC', False))
        train_ap = None
        if monitor_pr_auc:
            try:
                from sklearn.metrics import average_precision_score
                train_ap = float(average_precision_score(train_labels, train_probs[:, 1]))
            except Exception as e:
                logger.warning(f"⚠ 计算训练集PR-AUC失败: {e}")

        train_f1_star = None
        train_threshold_star = None
        if X_val_split is None:
            train_threshold_star, _train_metric, _ = find_optimal_threshold(train_labels, train_probs, metric='f1_binary', positive_class=1)
            train_preds_star = (train_probs[:, 1] >= float(train_threshold_star)).astype(int)
            train_f1_star = f1_score(train_labels, train_preds_star, pos_label=1, zero_division=0)

        val_f1 = None
        val_threshold = None
        val_ap = None

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
                    if input_is_features:
                        z = X_batch
                    else:
                        z = backbone(X_batch, return_sequence=False)
                    logits_a, logits_b = classifier.dual_mlp(z, return_separate=True)
                    logits_avg = (logits_a + logits_b) / 2.0
                    probs = torch.softmax(logits_avg, dim=1)
                    all_val_probs.append(probs.cpu().numpy())
                    all_val_labels.append(y_batch.cpu().numpy())

            val_probs = np.concatenate(all_val_probs)
            val_labels = np.concatenate(all_val_labels)
            if monitor_pr_auc:
                try:
                    from sklearn.metrics import average_precision_score
                    val_ap = float(average_precision_score(val_labels, val_probs[:, 1]))
                except Exception as e:
                    logger.warning(f"⚠ 计算验证集PR-AUC失败: {e}")
            val_threshold, val_metric, _ = find_optimal_threshold(val_labels, val_probs, metric='f1_binary', positive_class=1)
            val_preds = (val_probs[:, 1] >= float(val_threshold)).astype(int)
            val_f1 = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)

            if val_f1 > best_f1:
                best_f1 = float(val_f1)
                best_epoch = int(epoch + 1)
                best_threshold = float(val_threshold)
                best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
        else:
            if train_f1_star is not None and train_f1_star > best_f1:
                best_f1 = float(train_f1_star)
                best_epoch = int(epoch + 1)
                best_threshold = float(train_threshold_star)
                best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
        
        classifier.train()
        
        # 保存历史
        history['train_loss'].append(epoch_loss)
        history['supervision'].append(epoch_losses['supervision'])
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1 if val_f1 is not None else np.nan)
        history['val_threshold'].append(val_threshold if val_threshold is not None else np.nan)
        history['train_f1_star'].append(train_f1_star if train_f1_star is not None else np.nan)
        history['train_threshold_star'].append(train_threshold_star if train_threshold_star is not None else np.nan)
        if monitor_pr_auc:
            if 'train_ap' not in history:
                history['train_ap'] = []
            if 'val_ap' not in history:
                history['val_ap'] = []
            history['train_ap'].append(train_ap if train_ap is not None else np.nan)
            history['val_ap'].append(val_ap if val_ap is not None else np.nan)
        
        # 输出日志（先输出再早停，保证触发早停的最后一轮也有epoch日志）
        progress = (epoch + 1) / config.FINETUNE_EPOCHS * 100
        if val_f1 is not None:
            msg = (
                f"[Stage 3] Epoch [{epoch+1}/{config.FINETUNE_EPOCHS}] ({progress:.1f}%) | "
                f"Loss: {epoch_loss:.4f} | "
                f"L(total={epoch_losses['total']:.4f}, sup={epoch_losses['supervision']:.4f}, a={epoch_losses['stream_a']:.4f}, b={epoch_losses['stream_b']:.4f}) | "
                f"TrF1: {train_f1:.4f} | ValF1*: {val_f1:.4f} | Th: {val_threshold:.4f}"
            )
            if monitor_pr_auc:
                msg += f" | TrAP: {float(train_ap) if train_ap is not None else float('nan'):.4f} | ValAP: {float(val_ap) if val_ap is not None else float('nan'):.4f}"
            logger.info(msg)
        else:
            train_f1_star_disp = float(train_f1_star) if train_f1_star is not None else float('nan')
            train_th_star_disp = float(train_threshold_star) if train_threshold_star is not None else float('nan')
            msg = (
                f"[Stage 3] Epoch [{epoch+1}/{config.FINETUNE_EPOCHS}] ({progress:.1f}%) | "
                f"Loss: {epoch_loss:.4f} | "
                f"L(total={epoch_losses['total']:.4f}, sup={epoch_losses['supervision']:.4f}, a={epoch_losses['stream_a']:.4f}, b={epoch_losses['stream_b']:.4f}) | "
                f"TrF1: {train_f1:.4f} | TrF1*: {train_f1_star_disp:.4f} | Th: {train_th_star_disp:.4f}"
            )
            if monitor_pr_auc:
                msg += f" | TrAP: {float(train_ap) if train_ap is not None else float('nan'):.4f}"
            logger.info(msg)

        # 早停检查
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
                        log_early_stopping(logger, epoch+1, best_epoch, best_f1, current_metric, no_improve_count, es_patience)
                        break

    # 输出阶段总结
    actual_epochs = len(history['train_loss'])
    log_stage_end(logger, "Stage 3", {
        "最终损失": f"{history['train_loss'][-1]:.4f}",
        "最终F1": f"{history['train_f1'][-1]:.4f}",
        "最佳F1": f"{best_f1:.4f} (epoch {best_epoch})" if best_epoch > 0 else "N/A",
        "实际训练轮数": f"{actual_epochs}/{config.FINETUNE_EPOCHS}"
    })
    
    # 生成特征可视化
    classifier.eval()
    if backbone_finetune_active:
        backbone.eval()
    with torch.no_grad():
        if input_is_features:
            train_features = np.asarray(X_train, dtype=np.float32)
        else:
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
                      title="Stage 3: Feature Distribution", method='tsne')
    
    optimal_threshold = float(best_threshold)
    
    # 保存模型
    if backbone_finetune_started:
        finetuned_backbone_path = os.path.join(config.CLASSIFICATION_DIR, "models", "backbone_finetuned.pth")
        torch.save(backbone.state_dict(), finetuned_backbone_path)

    if best_state is not None:
        best_path = os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_best_f1.pth")
        torch.save(best_state, best_path)

    # 保存最后10轮(默认)中loss最小的模型
    minloss_path = None
    if keep_last_k_minloss > 0 and len(last_k_loss_states) > 0:
        try:
            min_epoch, min_loss, min_state = min(last_k_loss_states, key=lambda t: float(t[1]))
            minloss_path = os.path.join(
                config.CLASSIFICATION_DIR,
                "models",
                f"classifier_last{int(keep_last_k_minloss)}_minloss_epoch{int(min_epoch)}.pth"
            )
            torch.save(min_state, minloss_path)
            logger.info(f"✓ 已保存最后{int(keep_last_k_minloss)}轮中loss最小模型: epoch={int(min_epoch)} loss={float(min_loss):.6f}")
        except Exception as e:
            logger.warning(f"⚠ 保存最后{int(keep_last_k_minloss)}轮loss最小模型失败: {e}")

    final_path = os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_final.pth")
    final_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
    torch.save(final_state, final_path)
    
    history_path = os.path.join(config.CLASSIFICATION_DIR, "models", "training_history.npz")
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})
    
    # 保存元数据
    if finetuned_backbone_path is not None:
        backbone_path = finetuned_backbone_path
    elif backbone_path is None:
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    
    metadata_path = os.path.join(config.CLASSIFICATION_DIR, "models", "model_metadata.json")
    metadata = {
        'backbone_path': backbone_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X_train),
        'n_original': n_original if n_original is not None else len(X_train),
        'finetune_epochs': config.FINETUNE_EPOCHS,
        'input_is_features': input_is_features,
        'feature_dim': int(getattr(config, 'OUTPUT_DIM', config.MODEL_DIM)),
    }
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    output_files = {
        "最佳模型": os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_best_f1.pth"),
        "最终模型": final_path,
        "训练历史": history_path,
        "模型元数据": metadata_path
    }
    if minloss_path is not None:
        output_files[f"最后{int(keep_last_k_minloss)}轮loss最小模型"] = minloss_path
    if finetuned_backbone_path:
        output_files["微调骨干网络"] = finetuned_backbone_path
    
    log_output_paths(logger, output_files)
    
    return classifier, history, optimal_threshold


def main(args):
    """主训练函数"""
    
    # 初始化
    rng_fp_before_seed = _rng_fingerprint_short()
    set_seed(config.SEED)
    rng_fp_after_seed = _rng_fingerprint_short()
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, "logs"), name='train')

    logger.info(f"🔧 RNG指纹(seed前): {rng_fp_before_seed}")
    logger.info(f"🔧 RNG指纹(seed后): {rng_fp_after_seed} ({_seed_snapshot()})")
    
    log_section_header(logger, "🚀 MEDAL-Lite 训练流程")
    logger.info(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU信息
    if torch.cuda.is_available():
        logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"  设备: {config.DEVICE}")
    else:
        logger.warning("⚠ 使用CPU训练")
    
    # 获取阶段范围
    start_stage = getattr(args, 'start_stage', 1)
    end_stage = getattr(args, 'end_stage', 3)
    
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
    
    # 加载数据集
    X_train = None
    y_train_clean = None
    y_train_noisy = None
    
    if start_stage <= 3:
        log_section_header(logger, "📦 数据集加载")
        logger.info(f"🔧 RNG指纹(加载训练数据前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        log_data_stats(logger, {
            "正常流量路径": config.BENIGN_TRAIN,
            "恶意流量路径": config.MALICIOUS_TRAIN,
            "序列长度": config.SEQUENCE_LENGTH
        }, "训练集配置")
        
        if PREPROCESS_AVAILABLE and check_preprocessed_exists('train'):
            logger.info("✓ 发现预处理文件，直接加载...")
            X_train, y_train_clean, train_files = load_preprocessed('train')
            X_train = normalize_burstsize_inplace(X_train)
        else:
            logger.info("开始加载训练数据集（从PCAP文件）...")
            X_train, y_train_clean, train_files = load_dataset(
                benign_dir=config.BENIGN_TRAIN,
                malicious_dir=config.MALICIOUS_TRAIN,
                sequence_length=config.SEQUENCE_LENGTH
            )
            X_train = normalize_burstsize_inplace(X_train)

        logger.info(f"🔧 RNG指纹(加载训练数据后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        
        if X_train is None:
            logger.error("❌ 训练数据集加载失败!")
            return
        
        log_data_stats(logger, {
            "数据形状": X_train.shape,
            "正常样本": (y_train_clean==0).sum(),
            "恶意样本": (y_train_clean==1).sum()
        }, "训练数据集")
        
        if start_stage >= 2 and start_stage != 3:
            logger.info(f"🔀 注入标签噪声 ({config.LABEL_NOISE_RATE*100:.0f}%)...")
            y_train_noisy, noise_mask = inject_label_noise(y_train_clean, config.LABEL_NOISE_RATE)
            logger.info(f"✓ 噪声标签创建完成: {noise_mask.sum()} 个标签被翻转")
        else:
            y_train_noisy = None
    
    # 构建骨干网络
    logger.info(f"🔧 RNG指纹(构建backbone前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    backbone = build_backbone(config, logger=logger)
    backbone = backbone.to(config.DEVICE)
    logger.info(f"🔧 RNG指纹(构建backbone后): {_rng_fingerprint_short()} ({_seed_snapshot()})")

    # Stage 1: 预训练骨干网络
    if start_stage <= 1:
        logger.info(f"🔧 RNG指纹(Stage1调用前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        use_instance_contrastive = getattr(config, 'USE_INSTANCE_CONTRASTIVE', False)
        contrastive_method = getattr(config, 'CONTRASTIVE_METHOD', 'infonce')
        
        method_lower = str(contrastive_method).lower()
        if use_instance_contrastive and method_lower == 'nnclr':
            batch_size = getattr(config, 'PRETRAIN_BATCH_SIZE_NNCLR', 64)
        elif use_instance_contrastive and method_lower == 'simsiam':
            batch_size = getattr(config, 'PRETRAIN_BATCH_SIZE_SIMSIAM', config.PRETRAIN_BATCH_SIZE)
        else:
            batch_size = config.PRETRAIN_BATCH_SIZE
        
        dataset = TensorDataset(torch.FloatTensor(X_train))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        backbone, pretrain_history = stage1_pretrain_backbone(backbone, train_loader, config, logger)
        logger.info(f"🔧 RNG指纹(Stage1返回后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        
        if end_stage <= 1:
            logger.info("✅ 已完成到 Stage 1")
            return backbone
    else:
        # 加载预训练骨干网络
        if hasattr(args, 'backbone_path') and args.backbone_path:
            backbone_path = args.backbone_path
        else:
            backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
        
        retrain_backbone = bool(getattr(args, 'retrain_backbone', False))
        if retrain_backbone:
            logger.warning("⚠ 使用随机初始化骨干网络")
            backbone.freeze()
        elif os.path.exists(backbone_path):
            logger.info(f"✓ 加载骨干网络: {backbone_path}")
            logger.info(f"🔧 RNG指纹(加载backbone权重前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
            try:
                backbone_state = torch.load(backbone_path, map_location=config.DEVICE, weights_only=True)
            except TypeError:
                backbone_state = torch.load(backbone_path, map_location=config.DEVICE)
            load_state_dict_shape_safe(backbone, backbone_state, logger, prefix="backbone")
            logger.info(f"🔧 RNG指纹(加载backbone权重后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        else:
            logger.error(f"❌ 找不到骨干网络: {backbone_path}")
            return
    
    # Stage 2: 标签矫正 + 数据增强
    if start_stage <= 2 and end_stage >= 2:
        logger.info(f"🔧 RNG指纹(Stage2调用前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        if y_train_noisy is None and y_train_clean is not None:
            logger.info(f"🔀 注入标签噪声 ({config.LABEL_NOISE_RATE*100:.0f}%)...")
            y_train_noisy, noise_mask = inject_label_noise(y_train_clean, config.LABEL_NOISE_RATE)
            logger.info(f"✓ 噪声标签创建完成: {noise_mask.sum()} 个标签被翻转")
        
        stage2_mode = getattr(args, 'stage2_mode', 'standard')
        X_augmented, y_augmented, sample_weights, correction_stats, tabddpm, n_original = stage2_label_correction_and_augmentation(
            backbone, X_train, y_train_noisy, y_train_clean, config, logger, stage2_mode=stage2_mode
        )
        logger.info(f"🔧 RNG指纹(Stage2返回后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        
        if end_stage <= 2:
            logger.info("✅ 已完成到 Stage 2")
            return backbone
    elif end_stage >= 3:
        # 跳过Stage 2
        augmented_data_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "augmented_features.npz")
        finetune_backbone_enabled = bool(getattr(config, 'FINETUNE_BACKBONE', False))
        
        if int(start_stage) == 3 and X_train is not None:
            logger.info("✅ Stage 3-only: 跳过 Stage 2")
            
            if finetune_backbone_enabled:
                X_augmented = X_train
                y_augmented = y_train_clean
                sample_weights = np.ones(len(X_train), dtype=np.float32)
                n_original = len(X_train)
                logger.info(f"✓ 使用原始序列: {len(X_train)} 个样本")
            else:
                X_train_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
                with torch.no_grad():
                    Z_clean = backbone(X_train_tensor, return_sequence=False).detach().cpu().numpy().astype(np.float32)
                X_augmented = Z_clean
                y_augmented = y_train_clean
                sample_weights = np.ones(len(Z_clean), dtype=np.float32)
                n_original = len(Z_clean)
                logger.info(f"✓ 使用特征向量: {len(Z_clean)} 个样本")
        elif os.path.exists(augmented_data_path):
            logger.info(f"✓ 加载增强特征: {augmented_data_path}")
            data = np.load(augmented_data_path)
            X_augmented = data['Z_augmented']
            y_augmented = data['y_augmented']
            sample_weights = data['sample_weights'] if 'sample_weights' in data else np.ones(len(X_augmented))
            n_original = int(data['n_original']) if 'n_original' in data else len(X_augmented)
        else:
            logger.error(f"❌ 找不到增强数据: {augmented_data_path}")
            return

    # Stage 3: 分类器微调
    if end_stage >= 3 and start_stage <= 3:
        logger.info(f"🔧 RNG指纹(Stage3调用前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        if hasattr(args, 'backbone_path') and args.backbone_path:
            actual_backbone_path = args.backbone_path
        else:
            actual_backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
        
        X_train_real = None
        try:
            real_kept_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "real_kept_data.npz")
            if os.path.exists(real_kept_path):
                real_pack = np.load(real_kept_path)
                X_train_real = real_pack.get('X_real', None)
        except Exception:
            X_train_real = None

        classifier, finetune_history, optimal_threshold = stage3_finetune_classifier(
            backbone, X_augmented, y_augmented, sample_weights, config, logger,
            n_original=n_original, backbone_path=actual_backbone_path, X_train_real=X_train_real,
            use_mixed_stream=bool(getattr(config, 'STAGE3_MIXED_STREAM', False))
        )
        logger.info(f"🔧 RNG指纹(Stage3返回后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    else:
        logger.info("⏭️ 跳过 Stage 3")
        return backbone
    
    # 绘制训练历史
    history_fig_path = os.path.join(config.CLASSIFICATION_DIR, "figures", "training_history.png")
    plot_training_history(finetune_history, history_fig_path)
    
    # 最终总结
    log_final_summary(logger, "训练完成", {
        "Stage 1": f"骨干网络预训练 - {config.PRETRAIN_EPOCHS} epochs",
        "Stage 2": "标签矫正+数据增强 - 完成",
        "Stage 3": f"分类器微调 - {config.FINETUNE_EPOCHS} epochs"
    }, {
        "特征提取": config.FEATURE_EXTRACTION_DIR,
        "标签矫正": config.LABEL_CORRECTION_DIR,
        "数据增强": config.DATA_AUGMENTATION_DIR,
        "分类器": config.CLASSIFICATION_DIR
    })
    
    return classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEDAL-Lite 训练脚本")
    
    parser.add_argument("--noise_rate", type=float, default=None, help="标签噪声率（默认使用config.LABEL_NOISE_RATE）")
    parser.add_argument("--start_stage", type=str, default="1", choices=["1", "2", "3"], help="起始阶段")
    parser.add_argument("--end_stage", type=str, default="3", choices=["1", "2", "3"], help="结束阶段")
    parser.add_argument("--backbone_path", type=str, default=None, help="骨干网络路径")
    parser.add_argument("--retrain_backbone", action="store_true", help="重新训练骨干网络")
    parser.add_argument("--stage2_mode", type=str, default="standard", choices=["standard", "clean_augment_only"], help="Stage 2模式")
    
    args = parser.parse_args()
    if args.noise_rate is not None:
        config.LABEL_NOISE_RATE = args.noise_rate
    
    main(args)
