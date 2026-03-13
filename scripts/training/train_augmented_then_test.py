"""
MEDAL-Lite 数据增强训练+测试脚本 (重构版)
=========================================
使用真实标签 + TabDDPM数据增强训练分类器，复用主流程代码
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
from torch.utils.data import TensorDataset, DataLoader

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger
from MoudleCode.utils.logging_utils import (
    log_section_header, log_data_stats, log_input_paths, log_final_summary
)
from MoudleCode.feature_extraction.backbone import build_backbone

from scripts.training.train import stage1_pretrain_backbone, stage2_label_correction, stage3_data_augmentation, stage4_finetune_classifier
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

def _advance_rng_to_stage2_baseline(logger, args_seed: int) -> None:
    """Align RNG state to match the non-retrain pipeline right before Stage2.

    In the non-retrain run, RNG typically advances due to backbone initialization.
    We reproduce that consumption by building a throwaway backbone once.
    """
    logger.info(f"🔧 RNG对齐(Stage2基线) - 起点: {rng_fingerprint_short()} ({seed_snapshot(config, args_seed)})")
    _tmp = build_backbone(config, logger=logger)
    del _tmp
    logger.info(f"🔧 RNG对齐(Stage2基线) - 重放后: {rng_fingerprint_short()} ({seed_snapshot(config, args_seed)})")


def main():
    parser = argparse.ArgumentParser(description="数据增强训练+测试")
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
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, 'logs'), name='augmented_train_test')

    logger.info(f"🔧 RNG指纹(seed前): {rng_fp_before_seed}")
    logger.info(f"🔧 RNG指纹(seed后): {rng_fp_after_seed} ({seed_snapshot(config, args.seed)})")

    # 配置：仅保留与消融相关的差异，其余沿用主配置
    apply_finetune_backbone_override(args, config)

    run_tag = args.run_tag.strip() or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config.LABEL_CORRECTION_DIR, 'analysis', 'augmented_runs', run_tag)
    safe_makedirs(run_dir)
    
    # 隔离输出目录
    original_classification_dir = config.CLASSIFICATION_DIR
    original_result_dir = config.RESULT_DIR
    original_data_aug_dir = config.DATA_AUGMENTATION_DIR
    
    config.CLASSIFICATION_DIR = os.path.join(run_dir, 'classification')
    config.RESULT_DIR = os.path.join(run_dir, 'result')
    config.DATA_AUGMENTATION_DIR = os.path.join(run_dir, 'data_augmentation')
    
    for d in [config.CLASSIFICATION_DIR, config.RESULT_DIR, config.DATA_AUGMENTATION_DIR]:
        safe_makedirs(d)
        safe_makedirs(os.path.join(d, 'models'))
        safe_makedirs(os.path.join(d, 'figures'))
        safe_makedirs(os.path.join(d, 'logs'))

    try:
        log_section_header(logger, "🚀 数据增强训练+测试模式（消融实验）")
        logger.info(f'运行目录: {run_dir}')
        logger.info(f'时间戳: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info('')
        logger.info('📋 训练模式说明:')
        logger.info('  - 数据来源: 原始数据 + TabDDPM增强数据')
        logger.info('  - 标签: 真实标签（无噪声）')
        logger.info('  - 标签矫正: 跳过（使用真实标签）')
        logger.info('  - 数据增强: TabDDPM（特征空间）')
        logger.info(f"  - 骨干微调: {'启用' if config.FINETUNE_BACKBONE else '关闭'}")
        logger.info('')
        
        # 输出配置
        config.log_stage_config(logger, "Stage 2")
        config.log_stage_config(logger, "Stage 3")
        
        # 加载数据
        logger.info(f"🔧 RNG指纹(加载数据前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        X_train, y_train_true, _ = load_train_dataset(config, prefer_preprocessed=True, normalize=True)
        logger.info(f"🔧 RNG指纹(加载数据后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        if X_train is None:
            raise RuntimeError('训练数据集加载失败')

        y_corrected = y_train_true.astype(int)
        correction_weight = np.ones(len(y_corrected), dtype=np.float32)

        log_data_stats(logger, {
            "数据形状": f"{X_train.shape} (样本数×序列长度×特征维度)",
            "训练样本总数": len(X_train),
            "正常样本": int((y_corrected == 0).sum()),
            "恶意样本": int((y_corrected == 1).sum()),
            "数据类型": "原始序列（将进行特征空间增强）"
        }, "原始训练数据统计")
        
        # 构建/加载骨干网络（根据是否重训决定）
        backbone = None
        backbone_path = args.backbone_path if args.backbone_path else os.path.join(config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth')

        if args.retrain_backbone:
            logger.info('🔁 重新训练骨干网络（Stage 1 自监督预训练）...')
            log_input_paths(logger, {
                "训练数据(正常)": config.BENIGN_TRAIN,
                "训练数据(恶意)": config.MALICIOUS_TRAIN,
            })
            logger.info(f"🔧 RNG指纹(Stage1分支进入): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")

            logger.info(f"🔧 RNG指纹(Stage1-构建backbone前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            backbone = build_backbone(config, logger=logger)
            logger.info(f"🔧 RNG指纹(Stage1-构建backbone后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")

            # 只支持InfoNCE，使用标准批次大小
            batch_size = config.PRETRAIN_BATCH_SIZE

            dataset = TensorDataset(torch.FloatTensor(X_train))
            logger.info(f"🔧 RNG指纹(Stage1-DataLoader创建前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            logger.info(f"🔧 RNG指纹(Stage1-DataLoader创建后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")

            backbone = backbone.to(config.DEVICE)
            logger.info(f"🔧 RNG指纹(Stage1训练前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            backbone, _pretrain_history = stage1_pretrain_backbone(backbone, train_loader, config, logger)
            logger.info(f"🔧 RNG指纹(Stage1训练后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")

            reseed_fp_before = rng_fingerprint_short()
            set_seed(args.seed)
            reseed_fp_after = rng_fingerprint_short()
            logger.info(
                f"🔧 RNG指纹(Stage1后重置seed前/后): {reseed_fp_before} -> {reseed_fp_after} ({seed_snapshot(config, args.seed)})"
            )
            _advance_rng_to_stage2_baseline(logger, args.seed)

            default_backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth')
            if args.backbone_path:
                logger.info(f"🔧 RNG指纹(Stage1保存前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
                torch.save(backbone.state_dict(), args.backbone_path)
                backbone_path = args.backbone_path
                logger.info(f'✓ 已保存新骨干网络: {backbone_path}')
                logger.info(f"🔧 RNG指纹(Stage1保存后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            else:
                backbone_path = default_backbone_path
                logger.info(f'✓ 已保存新骨干网络: {backbone_path}')
            logger.info(f"🔧 RNG指纹(Stage1分支结束/进入Stage2前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        else:
            logger.info(f"🔧 RNG指纹(构建backbone前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            backbone = build_backbone(config, logger=logger)
            logger.info(f"🔧 RNG指纹(构建backbone后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")

            if os.path.exists(backbone_path):
                logger.info(f'✓ 加载骨干网络: {backbone_path}')
                # 使用安全的模型加载函数（自动处理兼容性）
                from MoudleCode.utils.model_loader import load_backbone_safely
                backbone = load_backbone_safely(
                    backbone_path=backbone_path,
                    config=config,
                    device=config.DEVICE,
                    logger=logger
                )
                backbone.freeze()
            else:
                logger.warning('⚠ 使用随机初始化骨干网络')
                backbone.freeze()
        
        # Stage 2: 标签矫正（跳过，使用干净标签）
        # Stage 2会完全独立运行，复现标签矫正分析的流程
        # clean_augment_only模式会跳过标签矫正，直接使用真实标签
        logger.info(f"🔧 RNG指纹(Stage2调用前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        
        # 确定backbone路径
        backbone_path_for_stage2 = args.backbone_path if args.backbone_path else os.path.join(
            config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth'
        )
        
        features, y_corrected_stage2, correction_weight, correction_stats, n_original_stage2 = stage2_label_correction(
            backbone=None,  # 传递None，让Stage 2重新加载以确保状态一致
            X_train=None,  # 传递None，让Stage 2重新加载以确保流程一致
            y_train_noisy=None,  # 传递None，clean_augment_only模式会使用真实标签
            y_train_clean=None,  # 传递None，让Stage 2重新加载以确保流程一致
            config=config,
            logger=logger,
            stage2_mode='clean_augment_only',
            backbone_path=backbone_path_for_stage2
        )
        logger.info(f"🔧 RNG指纹(Stage2返回后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")

        # Stage 3: 数据增强
        logger.info(f"🔧 RNG指纹(Stage3调用前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        Z_aug, y_aug, w_aug, tabddpm, n_original = stage3_data_augmentation(
            backbone, features, y_corrected_stage2, correction_weight, config, logger
        )
        logger.info(f"🔧 RNG指纹(Stage3返回后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")

        # Stage 4: 分类器训练
        logger.info(f"🔧 RNG指纹(Stage4调用前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        # 加载原始序列：FINETUNE_BACKBONE=True 且输入特征时，Stage4会自动启用混合训练
        X_real, real_kept_path = load_stage4_real_sequences(config.DATA_AUGMENTATION_DIR, logger=logger)
        if X_real is not None:
            logger.info(f'✓ 加载原始序列: {real_kept_path}')
            logger.info(f'  - 原始序列形状: {X_real.shape}')
        else:
            if bool(getattr(config, 'FINETUNE_BACKBONE', False)):
                logger.warning(f"⚠️ FINETUNE_BACKBONE=True 但缺少原始序列文件: {real_kept_path}，Stage4将自动降级为仅训练分类器")
            logger.info('ℹ️ 未找到原始序列文件，将按特征模式训练（骨干冻结）')

        stage4_finetune_classifier(
            backbone, Z_aug, y_aug, w_aug, config, logger,
            n_original=n_original, backbone_path=backbone_path,
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

        log_final_summary(logger, "完成", {}, {
            "运行目录": run_dir,
            "分类器": classifier_best,
            "测试结果": config.RESULT_DIR
        })
        
    finally:
        config.CLASSIFICATION_DIR = original_classification_dir
        config.RESULT_DIR = original_result_dir
        config.DATA_AUGMENTATION_DIR = original_data_aug_dir


if __name__ == '__main__':
    main()
