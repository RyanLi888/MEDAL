"""
MEDAL-Lite 完整流程去除数据增强训练+测试脚本
================================================
完整流程：特征提取 -> 标签矫正 -> 分类训练（使用矫正后的数据，不进行数据增强）
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import json
import argparse
from datetime import datetime

import numpy as np
import torch
import random
import hashlib

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger, inject_label_noise
from MoudleCode.utils.logging_utils import (
    log_section_header, log_data_stats, log_input_paths, log_output_paths, log_final_summary
)
from MoudleCode.preprocessing.pcap_parser import load_dataset
from MoudleCode.feature_extraction.backbone import build_backbone

try:
    from scripts.utils.preprocess import check_preprocessed_exists, load_preprocessed, normalize_burstsize_inplace
    PREPROCESS_AVAILABLE = True
except Exception:
    PREPROCESS_AVAILABLE = False

from scripts.training.train import stage1_pretrain_backbone, stage2_label_correction, stage3_data_augmentation, stage4_finetune_classifier
from scripts.testing.test import main as test_main

# 导入标签矫正模块
from MoudleCode.label_correction.hybrid_court import HybridCourt
from MoudleCode.utils.logging_utils import log_stage_start, log_stage_end


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _seed_snapshot(args_seed: int) -> str:
    torch_seed = None
    try:
        torch_seed = int(torch.initial_seed())
    except Exception:
        torch_seed = None
    return (
        f"args.seed={int(args_seed)} | "
        f"config.SEED={int(getattr(config, 'SEED', -1))} | "
        f"torch.initial_seed={torch_seed}"
    )


def _load_train_dataset():
    """加载训练数据集"""
    if PREPROCESS_AVAILABLE and check_preprocessed_exists('train'):
        X_train, y_train, _ = load_preprocessed('train')
        X_train = normalize_burstsize_inplace(X_train)
        return X_train, y_train

    X_train, y_train, _ = load_dataset(
        benign_dir=config.BENIGN_TRAIN,
        malicious_dir=config.MALICIOUS_TRAIN,
        sequence_length=config.SEQUENCE_LENGTH,
    )
    X_train = normalize_burstsize_inplace(X_train)
    return X_train, y_train


# 注意：不再需要自定义的stage2函数，直接复用train.py中的stage2_label_correction_and_augmentation


def main():
    parser = argparse.ArgumentParser(description="完整流程去除数据增强训练+测试")
    parser.add_argument('--retrain_backbone', action='store_true', help='重新训练骨干网络')
    parser.add_argument('--backbone_path', type=str, default='', help='骨干网络路径')
    finetune_group = parser.add_mutually_exclusive_group()
    finetune_group.add_argument('--finetune_backbone', dest='finetune_backbone', action='store_true', help='启用骨干微调')
    finetune_group.add_argument('--no_finetune_backbone', dest='finetune_backbone', action='store_false', help='禁用骨干微调')
    parser.set_defaults(finetune_backbone=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_tag', type=str, default='')
    args = parser.parse_args()

    rng_fp_before_seed = _rng_fingerprint_short()
    set_seed(args.seed)
    rng_fp_after_seed = _rng_fingerprint_short()
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, 'logs'), name='no_aug_train_test')

    logger.info(f"🔧 RNG指纹(seed前): {rng_fp_before_seed}")
    logger.info(f"🔧 RNG指纹(seed后): {rng_fp_after_seed} ({_seed_snapshot(args.seed)})")

    # 配置：去除数据增强
    config.USE_FOCAL_LOSS = True
    config.USE_BCE_LOSS = False
    config.USE_SOFT_F1_LOSS = False
    config.STAGE3_ONLINE_AUGMENTATION = False
    config.STAGE3_USE_ST_MIXUP = False
    if args.finetune_backbone is not None:
        config.FINETUNE_BACKBONE = bool(args.finetune_backbone)
    config.FINETUNE_VAL_SPLIT = 0.0
    config.FINETUNE_ES_ALLOW_TRAIN_METRIC = True
    config.STAGE3_MIXED_STREAM = False  # 默认不启用混合训练（若骨干微调+特征输入，Stage4会自动切换）
    config.CLASSIFIER_INPUT_IS_FEATURES = False  # 输入是原始序列
    config.STAGE3_UNLABELED_LOSS_SCALE = 0.0  # 禁用无标签数据的半监督学习
    config.STAGE2_USE_TABDDPM = False  # 关键：禁用TabDDPM数据增强

    run_tag = args.run_tag.strip() or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config.LABEL_CORRECTION_DIR, 'analysis', 'no_augmentation_runs', run_tag)
    _safe_makedirs(run_dir)
    
    # 隔离输出目录
    original_classification_dir = config.CLASSIFICATION_DIR
    original_result_dir = config.RESULT_DIR
    config.CLASSIFICATION_DIR = os.path.join(run_dir, 'classification')
    config.RESULT_DIR = os.path.join(run_dir, 'result')
    
    for d in [config.CLASSIFICATION_DIR, config.RESULT_DIR]:
        _safe_makedirs(d)
        _safe_makedirs(os.path.join(d, 'models'))
        _safe_makedirs(os.path.join(d, 'figures'))
        _safe_makedirs(os.path.join(d, 'logs'))

    try:
        log_section_header(logger, "🚀 完整流程去除数据增强模式")
        logger.info(f'运行目录: {run_dir}')
        logger.info(f'时间戳: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info('')
        logger.info('📋 训练模式说明:')
        logger.info('  - Stage 1: 骨干网络预训练（特征提取）')
        logger.info('  - Stage 2: 标签矫正（无数据增强）')
        logger.info('  - Stage 3: 分类器训练（使用矫正后的原始数据）')
        logger.info('  - 数据增强: 完全跳过')
        logger.info(f"  - 骨干微调: {'启用' if config.FINETUNE_BACKBONE else '关闭'}")
        logger.info('')
        
        # 加载数据
        logger.info(f"🔧 RNG指纹(加载训练数据前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        X_train, y_train_clean = _load_train_dataset()
        logger.info(f"🔧 RNG指纹(加载训练数据后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        if X_train is None:
            raise RuntimeError('训练数据集加载失败')

        # 注入标签噪声
        logger.info(f"🔀 注入标签噪声 ({config.LABEL_NOISE_RATE*100:.0f}%)...")
        y_train_noisy, noise_mask = inject_label_noise(y_train_clean, config.LABEL_NOISE_RATE)
        logger.info(f"✓ 噪声标签创建完成: {noise_mask.sum()} 个标签被翻转")

        log_data_stats(logger, {
            "数据形状": f"{X_train.shape}",
            "训练样本总数": len(X_train),
            "正常样本（真实）": int((y_train_clean == 0).sum()),
            "恶意样本（真实）": int((y_train_clean == 1).sum()),
            "正常样本（噪声）": int((y_train_noisy == 0).sum()),
            "恶意样本（噪声）": int((y_train_noisy == 1).sum()),
            "噪声标签数": int(noise_mask.sum()),
            "噪声率": f"{config.LABEL_NOISE_RATE*100:.1f}%"
        }, "训练数据统计")
        
        # Stage 1: 骨干网络训练或加载
        backbone = None
        backbone_path = args.backbone_path if args.backbone_path else os.path.join(
            config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth'
        )
        
        if args.retrain_backbone or not os.path.exists(backbone_path):
            logger.info(f"🔧 RNG指纹(Stage1调用前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            backbone = stage1_pretrain_backbone(X_train, config, logger)
            logger.info(f"🔧 RNG指纹(Stage1返回后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            
            if backbone is not None:
                torch.save(backbone.state_dict(), backbone_path)
                logger.info(f'✓ 已保存新骨干网络: {backbone_path}')
        else:
            logger.info(f'✓ 使用已有骨干网络: {backbone_path}')
        
        # 加载骨干网络
        if backbone is None:
            logger.info(f"🔧 RNG指纹(构建backbone前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            backbone = build_backbone(config, logger=logger)
            logger.info(f"🔧 RNG指纹(构建backbone后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            
            if os.path.exists(backbone_path):
                logger.info(f"🔧 RNG指纹(加载backbone权重前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
                # 使用安全的模型加载函数（自动处理兼容性）
                from MoudleCode.utils.model_loader import load_backbone_safely
                backbone = load_backbone_safely(
                    backbone_path=backbone_path,
                    config=config,
                    device=config.DEVICE,
                    logger=logger
                )
                logger.info(f"🔧 RNG指纹(加载backbone权重后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            else:
                logger.warning('⚠ 骨干网络文件不存在，使用随机初始化')
        
        backbone.to(config.DEVICE)
        backbone.freeze()
        
        # Stage 2: 标签矫正
        # Stage 2会完全独立运行，复现标签矫正分析的流程，包括重新加载数据和注入噪声
        # 因此传递None，让Stage 2自己重新加载以确保流程一致
        logger.info(f"🔧 RNG指纹(Stage2调用前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        
        # 确定backbone路径
        backbone_path_for_stage2 = args.backbone_path if args.backbone_path else os.path.join(
            config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth'
        )
        
        features, y_corrected, correction_weight, correction_stats, n_original = stage2_label_correction(
            backbone=None,  # 传递None，让Stage 2重新加载以确保状态一致
            X_train=None,  # 传递None，让Stage 2重新加载以确保流程一致
            y_train_noisy=None,  # 传递None，让Stage 2重新注入噪声以确保流程一致
            y_train_clean=None,  # 传递None，让Stage 2重新加载以确保流程一致
            config=config,
            logger=logger,
            stage2_mode='standard',
            backbone_path=backbone_path_for_stage2
        )
        logger.info(f"🔧 RNG指纹(Stage2返回后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        logger.info(f"✓ Stage2完成: 特征形状={features.shape}, 标签形状={y_corrected.shape}")
        logger.info(f"✓ 矫正准确率: {correction_stats['accuracy']*100:.2f}%")
        
        # Stage 3: 数据增强（跳过，直接使用特征）
        logger.info("⏭️ 跳过Stage 3数据增强，直接使用Stage 2的特征")
        Z_augmented = features
        y_augmented = y_corrected
        sample_weights = correction_weight
        
        # Stage 4: 分类器训练（输入为特征；若启用骨干微调需同时提供原始序列）
        logger.info(f"🔧 RNG指纹(Stage4调用前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        real_kept_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "real_kept_data.npz")
        X_real = None
        if os.path.exists(real_kept_path):
            logger.info(f'✓ 加载原始序列: {real_kept_path}')
            real_data = np.load(real_kept_path)
            X_real = real_data['X_real']
            logger.info(f'  - 原始序列形状: {X_real.shape}')
        elif bool(getattr(config, 'FINETUNE_BACKBONE', False)):
            logger.warning('⚠️ 未找到real_kept_data，回退使用当前训练序列作为原始序列输入')
            X_real = X_train

        stage4_finetune_classifier(
            backbone, Z_augmented, y_augmented, sample_weights,
            config, logger, n_original=n_original, backbone_path=backbone_path,
            X_train_real=X_real
        )
        logger.info(f"🔧 RNG指纹(Stage4返回后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")

        # 测试
        log_section_header(logger, "🧪 测试评估")
        logger.info(f"🔧 RNG指纹(测试前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        classifier_best = os.path.join(config.CLASSIFICATION_DIR, 'models', 'classifier_best_f1.pth')
        
        meta_path = os.path.join(config.CLASSIFICATION_DIR, 'models', 'model_metadata.json')
        backbone_path_for_test = backbone_path
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            bp = meta.get('backbone_path', '')
            if bp:
                if os.path.exists(bp):
                    backbone_path_for_test = bp
                    logger.info(f"✓ 使用元数据中记录的骨干网络: {bp}")
                else:
                    logger.warning(f"⚠ 元数据中记录的骨干网络不存在: {bp}")
                    logger.warning(f"  回退到训练时使用的骨干网络: {backbone_path}")

        test_args = argparse.Namespace(backbone_path=backbone_path_for_test)
        test_main(test_args)
        logger.info(f"🔧 RNG指纹(测试后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")

        log_final_summary(logger, "完成", {
            "矫正准确率": f"{correction_stats['accuracy']*100:.2f}%",
            "保持样本": correction_stats['n_keep'],
            "翻转样本": correction_stats['n_flip'],
            "丢弃样本": correction_stats['n_drop'],
            "重加权样本": correction_stats['n_reweight']
        }, {
            "运行目录": run_dir,
            "分类器": classifier_best,
            "测试结果": config.RESULT_DIR
        })
        
    finally:
        config.CLASSIFICATION_DIR = original_classification_dir
        config.RESULT_DIR = original_result_dir


if __name__ == '__main__':
    main()
