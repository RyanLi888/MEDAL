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
from scripts.training.common import (
    add_finetune_backbone_cli_args,
    apply_finetune_backbone_override,
    load_stage4_real_sequences,
)


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
    
    # 梯度累积配置（InfoNCE不需要特殊处理）
    gradient_accumulation_steps = 1
    
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


def stage2_label_correction(backbone, X_train, y_train_noisy, y_train_clean, config, logger, stage2_mode='standard', backbone_path=None):
    """
    Stage 2: 标签矫正（可独立重跑，也可复用主流程状态）
    
    Args:
        backbone: 预训练的骨干网络（可选，如果为None则重新加载）
        X_train: (N, L, D) 训练序列（可选，如果为None则重新加载）
        y_train_noisy: (N,) 噪声标签（可选，如果为None则重新生成）
        y_train_clean: (N,) 干净标签（可选，如果为None则重新加载）
        config: 配置对象
        logger: 日志记录器
        stage2_mode: 'standard' 或 'clean_augment_only'
        backbone_path: backbone路径（可选，用于重新加载）
        
    Returns:
        features: 提取的特征
        y_corrected: 矫正后的标签
        correction_weight: 样本权重
        correction_stats: 矫正统计
        n_original: 原始样本数
    """
    log_stage_start(logger, "STAGE 2: 标签矫正", "矫正标签噪声")
    config.log_stage_config(logger, "Stage 2")
    
    # ========================
    # Step 1: 重置随机种子（复现标签矫正分析流程）
    # ========================
    logger.info("🔧 重置随机种子以确保可复现性...")
    set_seed(config.SEED)
    logger.info(f"🔧 RNG指纹(Stage2重置种子后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    
    # ========================
    # Step 2: 加载数据集（复现标签矫正分析流程）
    # ========================
    logger.info("┌─ Step 1: 加载数据集")
    logger.info(f"│  数据路径: {config.BENIGN_TRAIN} | {config.MALICIOUS_TRAIN}")
    logger.info(f"│  序列长度: {config.SEQUENCE_LENGTH}")
    
    if X_train is None or y_train_clean is None:
        # 优先使用预处理好的数据
        if PREPROCESS_AVAILABLE and check_preprocessed_exists('train'):
            logger.info("│  ✓ 使用预处理文件")
            X_train, y_train_clean, train_files = load_preprocessed('train')
        else:
            # 从PCAP文件加载
            logger.info("│  ⚠ 从PCAP文件加载（建议先预处理以加速）")
            X_train, y_train_clean, train_files = load_dataset(
                benign_dir=config.BENIGN_TRAIN,
                malicious_dir=config.MALICIOUS_TRAIN,
                sequence_length=config.SEQUENCE_LENGTH
            )
        
        if X_train is None:
            logger.error("│  ❌ 数据集加载失败!")
            raise RuntimeError("数据集加载失败")
    
    logger.info(f"└─ ✓ 完成: {X_train.shape[0]} 个样本 | 正常={(y_train_clean==0).sum()} | 恶意={(y_train_clean==1).sum()}")
    logger.info("")
    
    # ========================
    # Step 3: 注入标签噪声（复现标签矫正分析流程）
    # ========================
    logger.info("┌─ Step 2: 注入标签噪声")
    logger.info(f"│  噪声率: {config.LABEL_NOISE_RATE*100:.0f}%")
    
    if y_train_noisy is None:
        # 固定随机种子，确保相同噪声率的结果可复现
        set_seed(config.SEED)
        y_train_noisy, noise_mask = inject_label_noise(y_train_clean, config.LABEL_NOISE_RATE)
        logger.info(f"└─ ✓ 完成: {noise_mask.sum()} 个标签被翻转 | 原始纯度: {100*(y_train_clean==y_train_noisy).mean():.1f}%")
    else:
        noise_mask = (y_train_clean != y_train_noisy)
        logger.info(f"└─ ✓ 使用已有噪声标签: {noise_mask.sum()} 个标签被翻转 | 原始纯度: {100*(y_train_clean==y_train_noisy).mean():.1f}%")
    logger.info("")
    
    # ========================
    # Step 4: 提取特征（复现标签矫正分析流程）
    # ========================
    logger.info("┌─ Step 3: 提取特征")
    
    # 确定backbone路径（当未传入backbone时用于加载）
    if backbone_path is None:
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    
    if backbone is None:
        logger.info(f"│  加载预训练backbone: {os.path.basename(backbone_path)}")
        from MoudleCode.utils.model_loader import load_backbone_safely
        backbone = load_backbone_safely(
            backbone_path=backbone_path,
            config=config,
            device=config.DEVICE,
            logger=logger
        )
        logger.info("│  ✓ Backbone加载完成")
    else:
        logger.info("│  复用主流程中的backbone实例")
        backbone = backbone.to(config.DEVICE)
        logger.info("│  ✓ Backbone复用完成")
    
    # 提取特征（使用与标签矫正分析相同的函数逻辑）
    logger.info("│  Extracting features using backbone...")
    
    backbone.freeze()
    backbone.eval()
    backbone.to(config.DEVICE)
    
    features_list = []
    batch_size = 64
    X_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
    
    total_batches = (len(X_tensor) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_idx = i // batch_size + 1
            X_batch = X_tensor[i:i+batch_size]
            z_batch = backbone(X_batch, return_sequence=False)
            features_list.append(z_batch.cpu().numpy())
            
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                progress = batch_idx / total_batches * 100
                logger.info(f"│    Feature extraction progress: {batch_idx}/{total_batches} batches ({progress:.1f}%)")
        
        features = np.concatenate(features_list, axis=0)
    logger.info(f"│  ✓ Feature extraction complete: {features.shape}")
    
    # 保存特征
    features_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "train_features.npy")
    np.save(features_path, features)
    logger.info(f"└─ ✓ 完成: 特征维度 {features.shape} | 已保存")
    logger.info("")
    
    # 特征可视化
    feature_dist_path = os.path.join(config.LABEL_CORRECTION_DIR, "figures", "feature_distribution_stage2.png")
    plot_feature_space(features, y_train_clean, feature_dist_path,
                      title="Stage 2: Feature Distribution", method='tsne')
    
    # ========================
    # Step 5: 运行 Hybrid Court 标签矫正（复现标签矫正分析流程）
    # ========================
    logger.info("┌─ Step 4: 运行 Hybrid Court 标签矫正")
    logger.info("")
    
    log_subsection_header(logger, "步骤 2.1: Hybrid Court 标签矫正")
    logger.info(f"  输入: {len(y_train_noisy)} 个样本，噪声率: {config.LABEL_NOISE_RATE*100:.0f}%")
    logger.info(f"  方法: CL (置信学习) + AUM (训练动态) + KNN (语义投票)")
    
    hybrid_court = HybridCourt(config)

    if stage2_mode == 'clean_augment_only':
        # 跳过标签矫正，直接使用干净标签，所有样本权重为1
        logger.info("  ⚠️  模式7: 跳过标签矫正，直接使用真实标签，所有样本权重=1.0")
        y_corrected = y_train_clean.copy()
        action_mask = np.zeros(len(y_train_clean), dtype=int)  # 全部为Keep
        confidence = np.ones(len(y_train_clean), dtype=np.float32)  # 置信度设为1.0
        correction_weight = np.ones(len(y_train_clean), dtype=np.float32)  # 所有样本权重=1.0
        # 为了兼容后续代码，设置一些占位值
        aum_scores = np.zeros((len(y_train_clean),), dtype=np.float32)
        neighbor_consistency = np.ones(len(y_train_clean), dtype=np.float32)
        # 创建简单的pred_probs（用于兼容性）
        n_classes = len(np.unique(y_train_clean))
        pred_probs = np.zeros((len(y_train_clean), n_classes), dtype=np.float32)
        for i in range(len(y_train_clean)):
            pred_probs[i, y_train_clean[i]] = 1.0
        cl_confidence = pred_probs.max(axis=1)
    else:
        cl_threshold = float(getattr(config, 'STAGE2_CL_THRESHOLD', 0.7))
        aum_threshold = float(getattr(config, 'STAGE2_AUM_THRESHOLD', 0.0))
        aum_epochs = int(getattr(config, 'AUM_EPOCHS', 30))
        aum_batch_size = int(getattr(config, 'AUM_BATCH_SIZE', 128))
        aum_lr = float(getattr(config, 'AUM_LR', 0.01))
        knn_purity_threshold = float(getattr(config, 'STAGE2_KNN_PURITY_THRESHOLD', 0.8))
        use_drop = bool(getattr(config, 'STAGE2_USE_DROP', False))

        y_corrected, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs = hybrid_court.correct_labels(
            features=features,
            noisy_labels=y_train_noisy,
            device=str(config.DEVICE),
            y_true=y_train_clean,
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
             aum_scores=aum_scores,
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
    
    log_output_paths(logger, {
        "矫正结果": correction_results_path
    })
    
    log_stage_end(logger, "Stage 2", {
        "原始样本": len(X_train),
        "矫正准确率": f"{correction_accuracy*100:.2f}%"
    })

    n_original = len(X_train)
    return features, y_corrected, correction_weight, correction_stats, n_original


def stage3_data_augmentation(backbone, features, y_corrected, correction_weight, config, logger):
    """
    Stage 3: 数据增强 (TabDDPM)
    
    Args:
        backbone: 预训练的骨干网络（冻结）
        features: (N, D) 特征向量（来自Stage 2）
        y_corrected: (N,) 矫正后的标签
        correction_weight: (N,) 样本权重
        config: 配置对象
        logger: 日志记录器
        
    Returns:
        Z_augmented: 增强后的特征
        y_augmented: 增强后的标签
        sample_weights: 样本权重
        tabddpm: 训练好的TabDDPM模型
        n_original: 原始样本数
    """
    log_stage_start(logger, "STAGE 3: 数据增强", "使用TabDDPM在特征空间进行数据增强")
    config.log_stage_config(logger, "Stage 3")
    
    log_input_paths(logger, {
        "矫正结果": os.path.join(config.LABEL_CORRECTION_DIR, "models", "correction_results.npz"),
        "骨干网络模型": os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    })
    
    # 数据增强仅使用高权重数据（来自标签矫正权重）
    try:
        aug_min_w = float(getattr(config, 'STAGE3_AUGMENT_MIN_WEIGHT', getattr(config, 'STAGE2_AUGMENT_MIN_WEIGHT', 0.7)))
    except Exception:
        aug_min_w = 0.7
    aug_mask = (np.asarray(correction_weight) >= float(aug_min_w))
    Z_clean = features[aug_mask]
    y_clean = np.asarray(y_corrected)[aug_mask]
    weights_clean = np.asarray(correction_weight)[aug_mask]
    logger.info(f"🧪 TabDDPM训练/增强仅使用高权重样本: {int(aug_mask.sum())}/{len(features)} (threshold={aug_min_w})")
    try:
        if len(weights_clean) > 0:
            logger.info(
                f"🧪 TabDDPM训练集权重统计: min={float(np.min(weights_clean)):.4f}, mean={float(np.mean(weights_clean)):.4f}, max={float(np.max(weights_clean)):.4f}"
            )
    except Exception:
        pass
    
    logger.info(f"🔧 RNG指纹(Stage3-TabDDPM训练前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    
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
    
    # 学习率调度器
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
    logger.info(f"🔧 RNG指纹(Stage3-TabDDPM DataLoader创建前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    loader_ddpm = DataLoader(dataset_ddpm, batch_size=2048, shuffle=True)
    logger.info(f"🔧 RNG指纹(Stage3-TabDDPM DataLoader创建后): {_rng_fingerprint_short()} ({_seed_snapshot()})")

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
        
        # 学习率调度
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
    logger.info("生成增强特征")
    Z_syn, y_syn, w_syn = tabddpm.augment_feature_dataset(Z_clean, y_clean, weights_clean)

    n_train_original = int(features.shape[0])
    n_syn = int(Z_syn.shape[0] - int(Z_clean.shape[0]))
    if n_syn > 0:
        Z_syn_only = Z_syn[-n_syn:]
        y_syn_only = y_syn[-n_syn:]
        w_syn_only = np.ones((len(y_syn_only),), dtype=np.float32)
    else:
        Z_syn_only = np.zeros((0, features.shape[1]), dtype=features.dtype)
        y_syn_only = np.zeros((0,), dtype=np.asarray(y_corrected).dtype)
        w_syn_only = np.zeros((0,), dtype=np.asarray(correction_weight).dtype)

    Z_augmented = np.concatenate([features, Z_syn_only], axis=0)
    y_augmented = np.concatenate([np.asarray(y_corrected), np.asarray(y_syn_only)], axis=0)
    sample_weights = np.concatenate([np.asarray(correction_weight, dtype=np.float32), np.asarray(w_syn_only, dtype=np.float32)], axis=0)
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
        "TabDDPM模型": tabddpm_path,
        "增强特征": augmented_data_path
    })
    
    log_stage_end(logger, "Stage 3", {
        "原始样本": n_train_original,
        "增强后样本": len(Z_augmented),
        "合成样本": n_syn
    })

    return Z_augmented, y_augmented, sample_weights, tabddpm, n_train_original


def stage4_finetune_classifier(backbone, X_train, y_train, sample_weights, config, logger, 
                               n_original=None, backbone_path=None, X_train_real=None, use_mixed_stream=None):
    """
    Stage 4: 分类器微调
    
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
    log_stage_start(logger, "STAGE 4: 分类器微调", "训练双流MLP分类器进行最终威胁检测")
    config.log_stage_config(logger, "Stage 4")
    
    # 确定输入类型
    if use_mixed_stream is None:
        use_mixed_stream = bool(getattr(config, 'STAGE3_MIXED_STREAM', True))  # 默认启用混合训练
    
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

    # 原始样本权重划分策略（全部有监督，无无监督分支）：
    # - 权重 <= 0.3: 低置信度有标签训练（原drop类），权重=0.3
    # - 权重 > 0.3 且 <= 0.9: 中等置信度有标签训练，权重=1.0
    # - 权重 > 0.9: 高置信度有标签训练，权重=STAGE4_REAL_HIGH_WEIGHT
    # - 合成数据: 权重=0.8
    sw = np.asarray(sample_weights, dtype=np.float32) if hasattr(sample_weights, '__len__') else None
    unlabeled_thr = float(getattr(config, 'STAGE4_UNLABELED_WEIGHT_THRESHOLD', 0.3))
    high_weight_thr = float(getattr(config, 'STAGE4_HIGH_WEIGHT_THRESHOLD', 0.9))
    real_high_weight = float(getattr(config, 'STAGE4_REAL_HIGH_WEIGHT', 2.0))   # 原w>0.9 → STAGE4_REAL_HIGH_WEIGHT
    real_low_weight = float(getattr(config, 'STAGE4_REAL_LOW_WEIGHT', 1.0))     # 原0.3<w≤0.9 → 1.0
    real_drop_weight = float(getattr(config, 'STAGE4_REAL_DROP_WEIGHT', 0.3))   # 原w≤0.3 (drop类) → 0.3
    # 兼容旧配置
    sup_thr = float(getattr(config, 'STAGE3_SUP_WEIGHT_THRESHOLD', 0.8))
    real_sup_weight = float(getattr(config, 'STAGE3_REAL_SUP_WEIGHT', 3.0))
    syn_weight = float(getattr(config, 'STAGE3_SYN_WEIGHT', 0.8))               # 合成数据 → 0.8
    unlabeled_scale = 0.0  # 强制禁用无监督 KL 损失

    orig_n = 0
    if n_original is not None:
        try:
            orig_n = min(int(n_original), int(len(y_train)))
        except Exception:
            orig_n = 0

    orig_sup_mask = None
    orig_unlab_mask = None
    orig_high_weight_mask = None
    orig_low_weight_mask = None
    if sw is not None and orig_n > 0:
        orig_drop_mask = (sw[:orig_n] <= unlabeled_thr)          # 原低置信度（≤0.3）
        orig_high_weight_mask = (sw[:orig_n] > high_weight_thr)  # 高置信度（>0.9）
        orig_low_weight_mask = (~orig_drop_mask) & (~orig_high_weight_mask)  # 中等置信度
        orig_unlab_mask = np.zeros(orig_n, dtype=bool)  # 全部改为有监督，无无监督样本
        orig_sup_mask = np.ones(orig_n, dtype=bool)     # 所有原始样本均参与有监督训练
        logger.info(f"📊 Stage4 原始样本权重划分(全监督): drop类(≤{unlabeled_thr},w={real_drop_weight})={int(orig_drop_mask.sum())}, 中权重({unlabeled_thr}<w≤{high_weight_thr},w={real_low_weight})={int(orig_low_weight_mask.sum())}, 高权重(>{high_weight_thr},w={real_high_weight})={int(orig_high_weight_mask.sum())}")
        logger.info(f"🎯 骨干梯度约束: 仅高权重原始样本(>{high_weight_thr})参与骨干微调，其余样本仅训练分类器")

    finetune_backbone_requested = bool(getattr(config, 'FINETUNE_BACKBONE', False))

    # FINETUNE_BACKBONE 与分类器训练自动适配策略：
    # - 输入为特征且有原始序列: 自动启用 mixed-stream 以支持骨干微调
    # - 输入为特征但无原始序列: 自动降级为仅训练分类器（骨干冻结）
    # - 输入为原始序列: 直接按序列模式进行骨干微调（无需 mixed-stream）
    if finetune_backbone_requested:
        if input_is_features:
            if not has_real_sequences:
                logger.warning("⚠️ FINETUNE_BACKBONE=True 但未提供原始序列 X_train_real，自动关闭骨干微调，仅训练分类器")
                finetune_backbone_requested = False
                use_mixed_stream = False
            elif not use_mixed_stream:
                logger.info("🔀 FINETUNE_BACKBONE=True 且输入为特征，自动启用混合训练以接入原始序列")
                use_mixed_stream = True
        else:
            # 主输入本身就是原始序列，已经满足骨干微调要求；无需 mixed-stream
            if use_mixed_stream:
                logger.info("ℹ️ 输入为原始序列，禁用混合训练并按序列模式进行骨干微调")
            use_mixed_stream = False

    # 混合训练模式检查（仅在仍请求混合训练时执行）
    if use_mixed_stream and not has_real_sequences:
        logger.info("⚠️ 混合训练需要额外的原始序列(X_train_real)，当前未提供，已自动禁用混合训练")
        use_mixed_stream = False

    if use_mixed_stream and not input_is_features:
        logger.info("⚠️ 混合训练需要增强特征作为主输入，当前输入是原始序列，已自动禁用混合训练")
        use_mixed_stream = False

    if use_mixed_stream:
        logger.info("🔀 混合训练模式已启用")
        log_data_stats(logger, {
            "原始序列": f"{len(X_train_real)} 个样本",
            "增强特征": f"{len(X_train)} 个样本",
            "原始序列批次": config.STAGE3_MIXED_REAL_BATCH_SIZE,
            "增强特征批次": config.STAGE3_MIXED_SYN_BATCH_SIZE
        }, "混合训练配置")
    
    classifier_epochs = int(getattr(config, 'FINETUNE_CLASSIFIER_EPOCHS', getattr(config, 'FINETUNE_EPOCHS', 500)))
    if classifier_epochs <= 0:
        logger.warning(
            f"⚠️ FINETUNE_CLASSIFIER_EPOCHS={classifier_epochs} 非法，回退到 FINETUNE_EPOCHS={getattr(config, 'FINETUNE_EPOCHS', 500)}"
        )
        classifier_epochs = int(getattr(config, 'FINETUNE_EPOCHS', 500))
    if classifier_epochs <= 0:
        logger.warning("⚠️ 分类器训练轮数仍非法，强制设为1轮")
        classifier_epochs = 1

    finetune_scope = str(getattr(config, 'FINETUNE_BACKBONE_SCOPE', 'projection')).lower()
    finetune_backbone_lr = float(getattr(config, 'FINETUNE_BACKBONE_LR', 1e-5))
    finetune_backbone_warmup_epochs = int(getattr(config, 'FINETUNE_BACKBONE_WARMUP_EPOCHS', 0))
    finetune_three_stage = bool(getattr(config, 'FINETUNE_BACKBONE_THREE_STAGE', True))
    finetune_stage1_epochs = int(getattr(config, 'FINETUNE_BACKBONE_STAGE1_EPOCHS', finetune_backbone_warmup_epochs))
    finetune_stage2_epochs = int(getattr(config, 'FINETUNE_BACKBONE_STAGE2_EPOCHS', 0))
    finetune_stage3_epochs_cfg = int(getattr(config, 'FINETUNE_BACKBONE_STAGE3_EPOCHS', -1))
    backbone_max_train_epochs_cfg = int(getattr(config, 'FINETUNE_BACKBONE_MAX_EPOCHS', -1))
    finetune_stage2_lr = float(getattr(config, 'FINETUNE_BACKBONE_STAGE2_LR', finetune_backbone_lr))
    finetune_stage3_lr = float(getattr(config, 'FINETUNE_BACKBONE_STAGE3_LR', max(finetune_backbone_lr * 0.5, 1e-6)))
    
    if backbone_path is None:
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    initial_backbone_path = backbone_path
    
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
    if getattr(config, 'USE_BCE_LOSS', False):
        logger.info(f"🔧 BCE Loss: pos_weight={config.BCE_POS_WEIGHT}, label_smoothing={config.BCE_LABEL_SMOOTHING}")
    else:
        logger.info(f"🔧 FocalLoss: alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA}")

    # 骨干网络微调策略（三阶段：head-only -> projection -> all）
    projection_params = []
    backbone_non_projection_params = []
    has_projection = hasattr(backbone, 'projection')
    if has_projection:
        projection_params = list(backbone.projection.parameters())

    if finetune_backbone_requested:
        all_backbone_params = list(backbone.parameters())
        if finetune_scope == 'projection':
            if not has_projection or len(projection_params) == 0:
                logger.warning("⚠️ backbone没有projection层，回退到冻结模式")
                finetune_backbone_requested = False
            else:
                finetune_three_stage = False
        elif finetune_scope == 'all':
            if has_projection and len(projection_params) > 0:
                proj_ids = {id(p) for p in projection_params}
                backbone_non_projection_params = [p for p in all_backbone_params if id(p) not in proj_ids]
            else:
                backbone_non_projection_params = all_backbone_params
        else:
            logger.warning(f"⚠️ 未知FINETUNE_BACKBONE_SCOPE={finetune_scope}，回退到all")
            finetune_scope = 'all'
            if has_projection and len(projection_params) > 0:
                proj_ids = {id(p) for p in projection_params}
                backbone_non_projection_params = [p for p in all_backbone_params if id(p) not in proj_ids]
            else:
                backbone_non_projection_params = all_backbone_params

    backbone_finetune_active = False
    backbone_finetune_started = False
    current_backbone_phase = 'head'
    projection_group_idx = None
    non_projection_group_idx = None

    stage1_epochs = max(0, min(int(finetune_stage1_epochs), int(classifier_epochs)))
    stage2_epochs = max(0, min(int(finetune_stage2_epochs), max(int(classifier_epochs - stage1_epochs), 0)))
    stage3_epochs = None
    if finetune_three_stage and finetune_scope == 'all' and finetune_stage3_epochs_cfg >= 0:
        stage3_epochs = max(0, int(finetune_stage3_epochs_cfg))
        stage3_epochs = min(stage3_epochs, max(int(classifier_epochs - stage1_epochs - stage2_epochs), 0))

    backbone_train_end_epoch = None
    if finetune_backbone_requested:
        trainable_end_candidates = []
        if finetune_three_stage and finetune_scope == 'all' and stage3_epochs is not None:
            trainable_end_candidates.append(stage1_epochs + stage2_epochs + stage3_epochs)
        if backbone_max_train_epochs_cfg >= 0:
            trainable_end_candidates.append(backbone_max_train_epochs_cfg)
        if len(trainable_end_candidates) > 0:
            backbone_train_end_epoch = int(min(trainable_end_candidates))
            backbone_train_end_epoch = max(0, min(backbone_train_end_epoch, int(classifier_epochs)))

    if finetune_backbone_requested and finetune_three_stage and finetune_scope == 'all':
        phase3_desc = "其余轮" if stage3_epochs is None else f"{stage3_epochs}轮"
        logger.info(
            f"🧭 三阶段微调启用: "
            f"Phase1(head)={stage1_epochs}轮, "
            f"Phase2(projection)={stage2_epochs}轮(lr={finetune_stage2_lr}), "
            f"Phase3(all)={phase3_desc}(lr={finetune_stage3_lr})"
        )
    elif finetune_backbone_requested:
        logger.info(f"🧭 骨干微调启用: scope={finetune_scope}, lr={finetune_backbone_lr}")
    if finetune_backbone_requested and backbone_train_end_epoch is not None:
        logger.info(
            f"🧩 骨干可训练轮数上限: 前{backbone_train_end_epoch}轮；达到后自动冻结骨干，仅训练分类器"
        )

    def _set_backbone_trainable(mode: str) -> None:
        nonlocal backbone_finetune_active, backbone_finetune_started
        backbone.freeze()
        if (not finetune_backbone_requested) or mode == 'head':
            backbone.eval()
            backbone_finetune_active = False
            return

        if mode == 'projection':
            if hasattr(backbone, 'projection'):
                for p in backbone.projection.parameters():
                    p.requires_grad = True
        elif mode == 'all':
            backbone.unfreeze()

        backbone.train()
        backbone_finetune_active = True
        backbone_finetune_started = True

    def _resolve_backbone_phase(epoch_idx: int) -> str:
        if not finetune_backbone_requested:
            return 'head'
        if backbone_train_end_epoch is not None and epoch_idx >= backbone_train_end_epoch:
            return 'head'
        if finetune_scope == 'projection':
            if epoch_idx < stage1_epochs:
                return 'head'
            return 'projection'
        if not finetune_three_stage:
            if epoch_idx < finetune_backbone_warmup_epochs:
                return 'head'
            return 'all'
        if epoch_idx < stage1_epochs:
            return 'head'
        if epoch_idx < (stage1_epochs + stage2_epochs) and has_projection and len(projection_params) > 0:
            return 'projection'
        return 'all'

    # 优化器（参数组按阶段动态开关）
    param_groups = [{'params': classifier.dual_mlp.parameters(), 'lr': float(config.FINETUNE_LR)}]
    if finetune_backbone_requested and has_projection and len(projection_params) > 0:
        projection_group_idx = len(param_groups)
        param_groups.append({'params': projection_params, 'lr': 0.0})
    if finetune_backbone_requested and finetune_scope == 'all' and len(backbone_non_projection_params) > 0:
        non_projection_group_idx = len(param_groups)
        param_groups.append({'params': backbone_non_projection_params, 'lr': 0.0})

    optimizer = optim.AdamW(param_groups, weight_decay=config.PRETRAIN_WEIGHT_DECAY)
    logger.info("✓ 优化器初始化完成")

    def _apply_backbone_phase(phase: str, epoch_idx: int) -> None:
        _set_backbone_trainable(phase)
        if projection_group_idx is not None:
            optimizer.param_groups[projection_group_idx]['lr'] = finetune_stage2_lr if phase in ('projection', 'all') else 0.0
        if non_projection_group_idx is not None:
            optimizer.param_groups[non_projection_group_idx]['lr'] = finetune_stage3_lr if phase == 'all' else 0.0

        phase_name = {
            'head': 'Phase1-HeadOnly',
            'projection': 'Phase2-Projection',
            'all': 'Phase3-AllBackbone',
        }.get(phase, phase)
        if (
            phase == 'head'
            and finetune_backbone_requested
            and backbone_train_end_epoch is not None
            and epoch_idx >= backbone_train_end_epoch
        ):
            phase_name = 'BackboneFrozen-ClassifierOnly'
        logger.info(
            f"🔁 切换微调阶段: {phase_name} (epoch={epoch_idx+1}) | "
            f"proj_lr={optimizer.param_groups[projection_group_idx]['lr'] if projection_group_idx is not None else 0.0} | "
            f"backbone_lr={optimizer.param_groups[non_projection_group_idx]['lr'] if non_projection_group_idx is not None else 0.0}"
        )

    current_backbone_phase = _resolve_backbone_phase(0)
    _apply_backbone_phase(current_backbone_phase, 0)
    
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

    # 全监督模式：所有样本（含原来的drop类）统一纳入有监督训练
    sup_mask = syn_mask.copy()
    unlab_mask = np.zeros(len(sw_np), dtype=bool)  # 无无监督样本
    if orig_n > 0 and orig_sup_mask is not None:
        sup_mask[:orig_n] = orig_sup_mask  # orig_sup_mask 已全为 True
    else:
        # 无 n_original 信息时：全部作为有监督
        sup_mask = np.ones(len(sw_np), dtype=bool)

    # 监督样本权重：高权重=STAGE4_REAL_HIGH_WEIGHT，中权重=1.0，drop类=0.3，合成=0.8
    sw_sup = np.ones(int(sup_mask.sum()), dtype=np.float32) * syn_weight
    try:
        if orig_n > 0 and orig_sup_mask is not None and orig_high_weight_mask is not None and orig_low_weight_mask is not None:
            orig_drop_mask_full = (sw[:orig_n] <= unlabeled_thr) if sw is not None else np.zeros(orig_n, dtype=bool)
            # 在sup_mask中找到原始样本的位置
            sup_indices = np.where(sup_mask)[0]
            orig_sup_positions = [i for i, idx in enumerate(sup_indices) if idx < orig_n]

            for pos in orig_sup_positions:
                orig_idx = sup_indices[pos]
                if orig_idx < orig_n:
                    if orig_high_weight_mask[orig_idx]:
                        sw_sup[pos] = real_high_weight   # 高置信度 → STAGE4_REAL_HIGH_WEIGHT
                    elif orig_low_weight_mask[orig_idx]:
                        sw_sup[pos] = real_low_weight    # 中置信度 → 1.0
                    else:
                        sw_sup[pos] = real_drop_weight   # drop类（原≤0.3）→ 0.3
    except Exception as e:
        logger.warning(f"⚠️ 权重分配失败，使用默认权重: {e}")
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
    
    # 混合训练数据加载器（原始序列全部为有监督，无无标签分支）
    real_loader = None
    unlab_real_loader = None  # 无监督分支已禁用
    if use_mixed_stream and has_real_sequences:
        n_real = len(X_train_real)
        real_n = min(int(orig_n), int(n_real)) if orig_n > 0 else min(len(y_train_split), int(n_real))

        y_real_all = y_np[:real_n]
        # 全监督：所有原始序列均参与有监督训练
        real_sup_sel = np.ones(real_n, dtype=bool)

        X_real_sup = X_train_real[:real_n][real_sup_sel]
        y_real_sup = y_real_all[real_sup_sel]
        # 权重分配：高置信度=STAGE4_REAL_HIGH_WEIGHT，中置信度=1.0，drop类=0.3
        w_real_sup = np.ones(len(y_real_sup), dtype=np.float32) * real_drop_weight  # 默认drop类权重
        high_real_sup = np.zeros(len(y_real_sup), dtype=np.float32)
        if orig_high_weight_mask is not None and orig_low_weight_mask is not None:
            orig_drop_mask_real = (sw[:real_n] <= unlabeled_thr) if sw is not None else np.zeros(real_n, dtype=bool)
            real_high_in_sup = orig_high_weight_mask[:real_n][real_sup_sel]
            real_low_in_sup = orig_low_weight_mask[:real_n][real_sup_sel]
            real_drop_in_sup = orig_drop_mask_real[real_sup_sel]
            w_real_sup[real_high_in_sup] = real_high_weight   # 高置信度 → STAGE4_REAL_HIGH_WEIGHT
            w_real_sup[real_low_in_sup] = real_low_weight     # 中置信度 → 1.0
            w_real_sup[real_drop_in_sup] = real_drop_weight   # drop类 → 0.3
            high_real_sup[real_high_in_sup] = 1.0

        real_dataset = TensorDataset(
            torch.FloatTensor(X_real_sup),
            torch.LongTensor(y_real_sup),
            torch.FloatTensor(w_real_sup),
            torch.FloatTensor(high_real_sup)
        )
        real_batch_size = int(getattr(config, 'STAGE3_MIXED_REAL_BATCH_SIZE', 32))
        real_loader = DataLoader(real_dataset, batch_size=real_batch_size, shuffle=True)

        # 无标签分支已禁用，unlab_real_loader 保持 None

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
    best_backbone_state = None
    best_threshold = float(getattr(config, 'MALICIOUS_THRESHOLD', 0.5))
    finetuned_backbone_path = None

    keep_last_k_minloss = int(getattr(config, 'KEEP_LAST_K_MINLOSS_EPOCHS', 10))
    last_k_loss_states = []
    
    # 早停配置
    use_early_stopping = bool(getattr(config, 'FINETUNE_EARLY_STOPPING', False))
    es_warmup_epochs = int(getattr(config, 'FINETUNE_ES_WARMUP_EPOCHS', 30))
    es_patience = int(getattr(config, 'FINETUNE_ES_PATIENCE', 25))
    # 统一“最佳模型更新”和“早停计数重置”的改善阈值，避免两套标准导致日志/行为不一致
    es_min_delta = float(getattr(config, 'FINETUNE_ES_MIN_DELTA', 0.0))
    no_improve_count = 0
    best_es_metric = -1.0
    allow_train_metric_es = bool(getattr(config, 'FINETUNE_ES_ALLOW_TRAIN_METRIC', False))
    
    if use_early_stopping and X_val_split is None and not allow_train_metric_es:
        use_early_stopping = False
        logger.info("⚠️ 早停已禁用：无验证集且不允许使用训练集指标")
    
    real_loss_scale = float(getattr(config, 'STAGE3_MIXED_REAL_LOSS_SCALE', 2.0))
    syn_loss_scale = float(getattr(config, 'STAGE3_MIXED_SYN_LOSS_SCALE', 1.0))
    
    log_subsection_header(logger, "开始训练")
    
    for epoch in range(classifier_epochs):
        # 三阶段微调状态机
        next_backbone_phase = _resolve_backbone_phase(epoch)
        if next_backbone_phase != current_backbone_phase:
            current_backbone_phase = next_backbone_phase
            _apply_backbone_phase(current_backbone_phase, epoch)
        
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
                X_real, y_real, w_real, high_real = next(real_iter)
                X_real = X_real.to(config.DEVICE)
                y_real = y_real.to(config.DEVICE)
                w_real = w_real.to(config.DEVICE)
                high_real = (high_real.to(config.DEVICE) > 0.5)
                
                Z_syn, y_syn, w_syn = next(syn_iter)
                Z_syn = Z_syn.to(config.DEVICE)
                y_syn = y_syn.to(config.DEVICE)
                w_syn = w_syn.to(config.DEVICE)
                
                optimizer.zero_grad()
                
                if backbone_finetune_active:
                    z_real = backbone(X_real, return_sequence=False)
                    # 骨干仅由高权重样本更新；其余样本仍用于分类器监督
                    if torch.any(high_real):
                        z_real = torch.where(high_real.unsqueeze(1), z_real, z_real.detach())
                    else:
                        z_real = z_real.detach()
                else:
                    with torch.no_grad():
                        z_real = backbone(X_real, return_sequence=False)
                
                loss_real, loss_dict_real = criterion(classifier.dual_mlp, z_real, y_real, w_real, epoch, classifier_epochs)
                loss_syn, loss_dict_syn = criterion(classifier.dual_mlp, Z_syn, y_syn, w_syn, epoch, classifier_epochs)

                # 无监督 KL 损失已禁用（unlabeled_scale=0.0），直接跳过
                loss = real_loss_scale * loss_real + syn_loss_scale * loss_syn
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
                        # 序列直训时同样仅高权重样本更新骨干，其余样本仅训练分类器头
                        high_batch = (w_batch >= (real_high_weight - 1e-6))
                        if torch.any(high_batch):
                            z = torch.where(high_batch.unsqueeze(1), z, z.detach())
                        else:
                            z = z.detach()
                    else:
                        with torch.no_grad():
                            z = backbone(X_batch, return_sequence=False)

                loss, loss_dict = criterion(classifier.dual_mlp, z, y_batch, w_batch, epoch, classifier_epochs)

                # 无监督 KL 损失已禁用（unlabeled_scale=0.0），直接进行有监督反向传播
                loss.backward()
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
            current_backbone_state = None
            if backbone_finetune_started:
                current_backbone_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
            last_k_loss_states.append((int(epoch + 1), float(epoch_loss), current_state, current_backbone_state))
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

            if (float(val_f1) - float(best_f1)) > es_min_delta:
                best_f1 = float(val_f1)
                best_epoch = int(epoch + 1)
                best_threshold = float(val_threshold)
                best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
                if backbone_finetune_started:
                    best_backbone_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
                else:
                    best_backbone_state = None
        else:
            if train_f1_star is not None and (float(train_f1_star) - float(best_f1)) > es_min_delta:
                best_f1 = float(train_f1_star)
                best_epoch = int(epoch + 1)
                best_threshold = float(train_threshold_star)
                best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
                if backbone_finetune_started:
                    best_backbone_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
                else:
                    best_backbone_state = None
        
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
        progress = (epoch + 1) / classifier_epochs * 100
        if val_f1 is not None:
            msg = (
                f"[Stage 4] Epoch [{epoch+1}/{classifier_epochs}] ({progress:.1f}%) | "
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
                f"[Stage 4] Epoch [{epoch+1}/{classifier_epochs}] ({progress:.1f}%) | "
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
    log_stage_end(logger, "Stage 4", {
        "最终损失": f"{history['train_loss'][-1]:.4f}",
        "最终F1": f"{history['train_f1'][-1]:.4f}",
        "最佳F1": f"{best_f1:.4f} (epoch {best_epoch})" if best_epoch > 0 else "N/A",
        "实际训练轮数": f"{actual_epochs}/{classifier_epochs}"
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
    
    feature_dist_path = os.path.join(config.CLASSIFICATION_DIR, "figures", "feature_distribution_stage4.png")
    plot_feature_space(train_features, y_train, feature_dist_path,
                      title="Stage 4: Feature Distribution", method='tsne')
    
    optimal_threshold = float(best_threshold)
    
    # 保存模型
    backbone_best_f1_path = initial_backbone_path
    backbone_final_path = initial_backbone_path

    if backbone_finetune_started:
        finetuned_backbone_path = os.path.join(config.CLASSIFICATION_DIR, "models", "backbone_finetuned.pth")
        torch.save(backbone.state_dict(), finetuned_backbone_path)
        backbone_final_path = finetuned_backbone_path

    if best_backbone_state is not None:
        backbone_best_f1_path = os.path.join(config.CLASSIFICATION_DIR, "models", "backbone_best_f1.pth")
        torch.save(best_backbone_state, backbone_best_f1_path)

    if best_state is not None:
        best_path = os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_best_f1.pth")
        torch.save(best_state, best_path)

    # 保存最后10轮(默认)中loss最小的模型
    minloss_path = None
    minloss_backbone_path = None
    if keep_last_k_minloss > 0 and len(last_k_loss_states) > 0:
        try:
            min_epoch, min_loss, min_state, min_backbone_state = min(last_k_loss_states, key=lambda t: float(t[1]))
            minloss_path = os.path.join(
                config.CLASSIFICATION_DIR,
                "models",
                f"classifier_last{int(keep_last_k_minloss)}_minloss_epoch{int(min_epoch)}.pth"
            )
            torch.save(min_state, minloss_path)
            if min_backbone_state is not None:
                minloss_backbone_path = os.path.join(
                    config.CLASSIFICATION_DIR,
                    "models",
                    f"backbone_last{int(keep_last_k_minloss)}_minloss_epoch{int(min_epoch)}.pth"
                )
                torch.save(min_backbone_state, minloss_backbone_path)
            logger.info(f"✓ 已保存最后{int(keep_last_k_minloss)}轮中loss最小模型: epoch={int(min_epoch)} loss={float(min_loss):.6f}")
        except Exception as e:
            logger.warning(f"⚠ 保存最后{int(keep_last_k_minloss)}轮loss最小模型失败: {e}")

    final_path = os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_final.pth")
    final_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
    torch.save(final_state, final_path)
    
    history_path = os.path.join(config.CLASSIFICATION_DIR, "models", "training_history.npz")
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})
    
    # 保存元数据（默认 backbone_path 指向 best_f1 配对骨干，兼容旧测试逻辑）
    if backbone_best_f1_path is None:
        backbone_best_f1_path = initial_backbone_path
    if backbone_final_path is None:
        backbone_final_path = initial_backbone_path

    model_backbone_pairs = {
        "classifier_best_f1.pth": backbone_best_f1_path,
        "classifier_final.pth": backbone_final_path,
    }
    if minloss_path is not None:
        model_backbone_pairs[os.path.basename(minloss_path)] = minloss_backbone_path if minloss_backbone_path is not None else backbone_final_path
    
    metadata_path = os.path.join(config.CLASSIFICATION_DIR, "models", "model_metadata.json")
    metadata = {
        'backbone_path': backbone_best_f1_path,
        'backbone_best_f1_path': backbone_best_f1_path,
        'backbone_final_path': backbone_final_path,
        'model_backbone_pairs': model_backbone_pairs,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X_train),
        'n_original': n_original if n_original is not None else len(X_train),
        'finetune_epochs': classifier_epochs,
        'backbone_trainable_epochs': backbone_train_end_epoch if backbone_train_end_epoch is not None else (classifier_epochs if finetune_backbone_requested else 0),
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
    if minloss_backbone_path is not None:
        output_files[f"最后{int(keep_last_k_minloss)}轮loss最小配对骨干"] = minloss_backbone_path
    if best_backbone_state is not None:
        output_files["最佳配对骨干网络"] = backbone_best_f1_path
    if finetuned_backbone_path:
        output_files["微调骨干网络(最终)"] = finetuned_backbone_path
    
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
    end_stage = getattr(args, 'end_stage', 4)
    
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
    
    if start_stage <= 4:
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
        
        if start_stage >= 2 and start_stage != 3 and start_stage != 4:
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
        # 只支持InfoNCE，使用标准批次大小
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
            # 使用安全的模型加载函数（自动处理兼容性）
            from MoudleCode.utils.model_loader import load_backbone_safely
            backbone = load_backbone_safely(
                backbone_path=backbone_path,
                config=config,
                device=config.DEVICE,
                logger=logger
            )
            backbone.train()  # 设置为训练模式
            logger.info(f"🔧 RNG指纹(加载backbone权重后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        else:
            logger.error(f"❌ 找不到骨干网络: {backbone_path}")
            return
    
    # Stage 2: 标签矫正
    features = None
    y_corrected = None
    correction_weight = None
    correction_stats = None
    n_original = None
    
    if start_stage <= 2 and end_stage >= 2:
        logger.info(f"🔧 RNG指纹(Stage2调用前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        
        # Stage 2 默认独立重跑（可通过 config.STAGE2_FORCE_INDEPENDENT 控制）
        stage2_mode = getattr(args, 'stage2_mode', 'standard')
        stage2_force_independent = bool(getattr(config, 'STAGE2_FORCE_INDEPENDENT', True))
        
        # 确定backbone路径
        backbone_path = None
        if hasattr(args, 'backbone_path') and args.backbone_path:
            backbone_path = args.backbone_path
        else:
            backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
        
        if stage2_force_independent:
            logger.info("🔁 Stage2执行策略: 独立重跑（重载数据/噪声/backbone）")
            stage2_backbone = None
            stage2_x_train = None
            stage2_y_train_noisy = None
            stage2_y_train_clean = None
        else:
            logger.info("⚡ Stage2执行策略: 复用主流程已加载状态")
            stage2_backbone = backbone
            stage2_x_train = X_train
            stage2_y_train_noisy = y_train_noisy
            stage2_y_train_clean = y_train_clean

        features, y_corrected, correction_weight, correction_stats, n_original = stage2_label_correction(
            backbone=stage2_backbone,
            X_train=stage2_x_train,
            y_train_noisy=stage2_y_train_noisy,
            y_train_clean=stage2_y_train_clean,
            config=config,
            logger=logger,
            stage2_mode=stage2_mode,
            backbone_path=backbone_path
        )
        logger.info(f"🔧 RNG指纹(Stage2返回后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        
        if end_stage <= 2:
            logger.info("✅ 已完成到 Stage 2")
            return backbone
    elif end_stage >= 3:
        # 加载Stage 2的结果
        correction_results_path = os.path.join(config.LABEL_CORRECTION_DIR, "models", "correction_results.npz")
        if os.path.exists(correction_results_path):
            logger.info(f"✓ 加载标签矫正结果: {correction_results_path}")
            data = np.load(correction_results_path)
            y_corrected = data['y_corrected']
            correction_weight = data['correction_weight']
            # 需要重新提取特征
            if X_train is not None:
                backbone.to(config.DEVICE)
                backbone.freeze()
                backbone.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
                    features_list = []
                    batch_size = 64
                    for i in range(0, len(X_tensor), batch_size):
                        X_batch = X_tensor[i:i+batch_size]
                        z_batch = backbone(X_batch, return_sequence=False)
                        features_list.append(z_batch.cpu().numpy())
                    features = np.concatenate(features_list, axis=0)
                n_original = len(X_train)
        else:
            logger.warning("⚠️ 找不到标签矫正结果，将使用原始标签")
            if X_train is not None:
                backbone.to(config.DEVICE)
                backbone.freeze()
                backbone.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
                    features_list = []
                    batch_size = 64
                    for i in range(0, len(X_tensor), batch_size):
                        X_batch = X_tensor[i:i+batch_size]
                        z_batch = backbone(X_batch, return_sequence=False)
                        features_list.append(z_batch.cpu().numpy())
                    features = np.concatenate(features_list, axis=0)
                y_corrected = y_train_clean if y_train_clean is not None else np.zeros(len(X_train))
                correction_weight = np.ones(len(X_train), dtype=np.float32)
                n_original = len(X_train)
    
    # Stage 3: 数据增强
    Z_augmented = None
    y_augmented = None
    sample_weights = None
    tabddpm = None
    
    if start_stage <= 3 and end_stage >= 3:
        if features is None or y_corrected is None or correction_weight is None:
            logger.error("❌ Stage 3需要Stage 2的输出，请先运行Stage 2")
            return
        
        logger.info(f"🔧 RNG指纹(Stage3调用前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        Z_augmented, y_augmented, sample_weights, tabddpm, n_original = stage3_data_augmentation(
            backbone, features, y_corrected, correction_weight, config, logger
        )
        logger.info(f"🔧 RNG指纹(Stage3返回后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        
        if end_stage <= 3:
            logger.info("✅ 已完成到 Stage 3")
            return backbone
    elif end_stage >= 4:
        # 加载Stage 3的结果
        augmented_data_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "augmented_features.npz")
        if os.path.exists(augmented_data_path):
            logger.info(f"✓ 加载增强特征: {augmented_data_path}")
            data = np.load(augmented_data_path)
            Z_augmented = data['Z_augmented']
            y_augmented = data['y_augmented']
            sample_weights = data['sample_weights'] if 'sample_weights' in data else np.ones(len(Z_augmented))
            n_original = int(data['n_original']) if 'n_original' in data else len(Z_augmented)
        else:
            logger.error(f"❌ 找不到增强数据: {augmented_data_path}")
            return

    # Stage 4: 分类器微调
    if end_stage >= 4 and start_stage <= 4:
        if Z_augmented is None or y_augmented is None or sample_weights is None:
            logger.error("❌ Stage 4需要Stage 3的输出，请先运行Stage 3")
            return
        
        logger.info(f"🔧 RNG指纹(Stage4调用前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        if hasattr(args, 'backbone_path') and args.backbone_path:
            actual_backbone_path = args.backbone_path
        else:
            actual_backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
        
        X_train_real, _ = load_stage4_real_sequences(config.DATA_AUGMENTATION_DIR, logger=logger)

        classifier, finetune_history, optimal_threshold = stage4_finetune_classifier(
            backbone, Z_augmented, y_augmented, sample_weights, config, logger,
            n_original=n_original, backbone_path=actual_backbone_path, X_train_real=X_train_real
        )
        logger.info(f"🔧 RNG指纹(Stage4返回后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    else:
        logger.info("⏭️ 跳过 Stage 4")
        return backbone
    
    # 绘制训练历史
    history_fig_path = os.path.join(config.CLASSIFICATION_DIR, "figures", "training_history.png")
    plot_training_history(finetune_history, history_fig_path)
    
    # 最终总结
    classifier_epochs_for_log = int(getattr(config, 'FINETUNE_CLASSIFIER_EPOCHS', getattr(config, 'FINETUNE_EPOCHS', 0)))
    log_final_summary(logger, "训练完成", {
        "Stage 1": f"骨干网络预训练 - {config.PRETRAIN_EPOCHS} epochs",
        "Stage 2": "标签矫正 - 完成",
        "Stage 3": "数据增强 - 完成",
        "Stage 4": f"分类器微调 - {classifier_epochs_for_log} epochs"
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
    parser.add_argument("--start_stage", type=str, default="1", choices=["1", "2", "3", "4"], help="起始阶段")
    parser.add_argument("--end_stage", type=str, default="4", choices=["1", "2", "3", "4"], help="结束阶段")
    parser.add_argument("--backbone_path", type=str, default=None, help="骨干网络路径")
    parser.add_argument("--retrain_backbone", action="store_true", help="重新训练骨干网络")
    parser.add_argument("--stage2_mode", type=str, default="standard", choices=["standard", "clean_augment_only"], help="Stage 2模式")
    add_finetune_backbone_cli_args(
        parser,
        enable_help="启用骨干微调（训练过程自动适配）",
        disable_help="禁用骨干微调",
    )
    
    args = parser.parse_args()
    if args.noise_rate is not None:
        config.LABEL_NOISE_RATE = args.noise_rate
    apply_finetune_backbone_override(args, config)
    
    main(args)
