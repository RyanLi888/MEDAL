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
from torch.utils.data import DataLoader, TensorDataset

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger, inject_label_noise
from MoudleCode.utils.logging_utils import (
    log_section_header, log_data_stats, log_final_summary
)
from MoudleCode.feature_extraction.backbone import build_backbone

from scripts.training.train import stage1_pretrain_backbone, stage2_label_correction, stage4_finetune_classifier
from scripts.training.common import (
    safe_makedirs,
    rng_fingerprint_short,
    seed_snapshot,
    add_finetune_backbone_cli_args,
    apply_finetune_backbone_override,
    load_stage4_real_sequences,
)
from scripts.training.data_loading import load_train_dataset
from scripts.testing.test import main as test_main

# 注意：不再需要自定义的stage2函数，直接复用train.py中的stage2_label_correction_and_augmentation


def main():
    parser = argparse.ArgumentParser(description="完整流程去除数据增强训练+测试")
    parser.add_argument('--retrain_backbone', action='store_true', help='重新训练骨干网络')
    parser.add_argument('--backbone_path', type=str, default='', help='骨干网络路径')
    add_finetune_backbone_cli_args(
        parser,
        enable_help='启用骨干微调（训练过程自动适配）',
        disable_help='禁用骨干微调'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_tag', type=str, default='')
    args = parser.parse_args()

    rng_fp_before_seed = rng_fingerprint_short()
    set_seed(args.seed)
    rng_fp_after_seed = rng_fingerprint_short()
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, 'logs'), name='no_aug_train_test')

    logger.info(f"🔧 RNG指纹(seed前): {rng_fp_before_seed}")
    logger.info(f"🔧 RNG指纹(seed后): {rng_fp_after_seed} ({seed_snapshot(config, args.seed)})")

    # 配置：仅保留与消融相关的差异，其余沿用主配置
    apply_finetune_backbone_override(args, config)
    config.STAGE2_USE_TABDDPM = False  # 关键：禁用TabDDPM数据增强

    run_tag = args.run_tag.strip() or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config.LABEL_CORRECTION_DIR, 'analysis', 'no_augmentation_runs', run_tag)
    safe_makedirs(run_dir)
    
    # 隔离输出目录
    original_classification_dir = config.CLASSIFICATION_DIR
    original_result_dir = config.RESULT_DIR
    config.CLASSIFICATION_DIR = os.path.join(run_dir, 'classification')
    config.RESULT_DIR = os.path.join(run_dir, 'result')
    
    for d in [config.CLASSIFICATION_DIR, config.RESULT_DIR]:
        safe_makedirs(d)
        safe_makedirs(os.path.join(d, 'models'))
        safe_makedirs(os.path.join(d, 'figures'))
        safe_makedirs(os.path.join(d, 'logs'))

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
        logger.info(f"🔧 RNG指纹(加载训练数据前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        X_train, y_train_clean, _ = load_train_dataset(config, prefer_preprocessed=True, normalize=True)
        logger.info(f"🔧 RNG指纹(加载训练数据后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
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
            logger.info(f"🔧 RNG指纹(Stage1调用前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            backbone = build_backbone(config, logger=logger).to(config.DEVICE)
            pretrain_dataset = TensorDataset(torch.FloatTensor(X_train))
            pretrain_loader = DataLoader(pretrain_dataset, batch_size=config.PRETRAIN_BATCH_SIZE, shuffle=True)
            backbone, _ = stage1_pretrain_backbone(backbone, pretrain_loader, config, logger)
            logger.info(f"🔧 RNG指纹(Stage1返回后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            
            if backbone is not None:
                torch.save(backbone.state_dict(), backbone_path)
                logger.info(f'✓ 已保存新骨干网络: {backbone_path}')
        else:
            logger.info(f'✓ 使用已有骨干网络: {backbone_path}')
        
        # 加载骨干网络
        if backbone is None:
            logger.info(f"🔧 RNG指纹(构建backbone前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            backbone = build_backbone(config, logger=logger)
            logger.info(f"🔧 RNG指纹(构建backbone后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            
            if os.path.exists(backbone_path):
                logger.info(f"🔧 RNG指纹(加载backbone权重前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
                # 使用安全的模型加载函数（自动处理兼容性）
                from MoudleCode.utils.model_loader import load_backbone_safely
                backbone = load_backbone_safely(
                    backbone_path=backbone_path,
                    config=config,
                    device=config.DEVICE,
                    logger=logger
                )
                logger.info(f"🔧 RNG指纹(加载backbone权重后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            else:
                logger.warning('⚠ 骨干网络文件不存在，使用随机初始化')
        
        backbone.to(config.DEVICE)
        backbone.freeze()
        
        # Stage 2: 标签矫正
        # Stage 2会完全独立运行，复现标签矫正分析的流程，包括重新加载数据和注入噪声
        # 因此传递None，让Stage 2自己重新加载以确保流程一致
        logger.info(f"🔧 RNG指纹(Stage2调用前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        
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
        logger.info(f"🔧 RNG指纹(Stage2返回后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        logger.info(f"✓ Stage2完成: 特征形状={features.shape}, 标签形状={y_corrected.shape}")
        logger.info(f"✓ 矫正准确率: {correction_stats['accuracy']*100:.2f}%")
        
        # Stage 3: 数据增强（跳过，直接使用特征）
        logger.info("⏭️ 跳过Stage 3数据增强，直接使用Stage 2的特征")
        Z_augmented = features
        y_augmented = y_corrected
        sample_weights = correction_weight
        
        # Stage 4: 分类器训练（输入为特征；若启用骨干微调需同时提供原始序列）
        logger.info(f"🔧 RNG指纹(Stage4调用前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        X_real, real_kept_path = load_stage4_real_sequences(config.DATA_AUGMENTATION_DIR, logger=logger)
        if X_real is not None:
            logger.info(f'✓ 加载原始序列: {real_kept_path}')
            logger.info(f'  - 原始序列形状: {X_real.shape}')
        elif bool(getattr(config, 'FINETUNE_BACKBONE', False)):
            logger.warning('⚠️ 未找到real_kept_data，回退使用当前训练序列作为原始序列输入')
            X_real = X_train

        stage4_finetune_classifier(
            backbone, Z_augmented, y_augmented, sample_weights,
            config, logger, n_original=n_original, backbone_path=backbone_path,
            X_train_real=X_real
        )
        logger.info(f"🔧 RNG指纹(Stage4返回后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")

        # 测试
        log_section_header(logger, "🧪 测试评估")
        logger.info(f"🔧 RNG指纹(测试前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
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
        logger.info(f"🔧 RNG指纹(测试后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")

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
