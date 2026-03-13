"""
MEDAL-Lite 干净数据训练+测试脚本 (重构版)
=========================================
使用干净标签（无噪声）训练分类器，复用主流程代码
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

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger
from MoudleCode.utils.logging_utils import (
    log_section_header, log_data_stats, log_final_summary
)
from MoudleCode.feature_extraction.backbone import build_backbone

from scripts.training.train import stage4_finetune_classifier
from scripts.training.common import (
    safe_makedirs,
    rng_fingerprint_short,
    seed_snapshot,
    add_finetune_backbone_cli_args,
    apply_finetune_backbone_override,
)
from scripts.training.data_loading import load_train_dataset
from scripts.testing.test import main as test_main

def main():
    parser = argparse.ArgumentParser(description="干净数据训练+测试")
    parser.add_argument('--use_ground_truth', action='store_true', help='使用真实标签')
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
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, 'logs'), name='clean_train_test')

    logger.info(f"🔧 RNG指纹(seed前): {rng_fp_before_seed}")
    logger.info(f"🔧 RNG指纹(seed后): {rng_fp_after_seed} ({seed_snapshot(config, args.seed)})")

    # 配置：仅保留与消融相关的差异，其余沿用主配置
    apply_finetune_backbone_override(args, config)

    run_tag = args.run_tag.strip() or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config.LABEL_CORRECTION_DIR, 'analysis', 'clean_only_runs', run_tag)
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
        log_section_header(logger, "🚀 干净数据训练+测试模式")
        logger.info(f'运行目录: {run_dir}')
        logger.info(f'时间戳: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info('')
        logger.info('📋 训练模式说明:')
        logger.info('  - 数据来源: 纯原始训练数据（无增强）')
        logger.info('  - 标签: 真实标签（无噪声）')
        logger.info('  - 标签矫正: 跳过')
        logger.info('  - 数据增强: 跳过')
        logger.info(f"  - 骨干微调: {'启用' if config.FINETUNE_BACKBONE else '关闭'}")
        logger.info('')
        
        # 输出配置
        config.log_stage_config(logger, "Stage 3")
        
        # 加载数据
        logger.info(f"🔧 RNG指纹(加载训练数据前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        X_train, y_train_true, _ = load_train_dataset(config, prefer_preprocessed=True, normalize=True)
        logger.info(f"🔧 RNG指纹(加载训练数据后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        if X_train is None:
            raise RuntimeError('训练数据集加载失败')

        y_corrected = y_train_true.astype(int)
        correction_weight = np.ones(len(y_corrected), dtype=np.float32)

        log_data_stats(logger, {
            "数据形状": f"{X_train.shape} (样本数×序列长度×特征维度)",
            "训练样本总数": len(X_train),
            "正常样本": int((y_corrected == 0).sum()),
            "恶意样本": int((y_corrected == 1).sum()),
            "数据类型": "原始序列（支持骨干微调）"
        }, "训练数据统计")
        
        # 加载骨干网络
        logger.info(f"🔧 RNG指纹(构建backbone前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        backbone = build_backbone(config, logger=logger)
        logger.info(f"🔧 RNG指纹(构建backbone后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        backbone_path = args.backbone_path if args.backbone_path else os.path.join(config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth')
        
        if os.path.exists(backbone_path) and not args.retrain_backbone:
            logger.info(f'✓ 加载骨干网络: {backbone_path}')
            logger.info(f"🔧 RNG指纹(加载backbone权重前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            try:
                state_dict = torch.load(backbone_path, map_location=config.DEVICE, weights_only=True)
            except TypeError:
                state_dict = torch.load(backbone_path, map_location=config.DEVICE)
            backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"🔧 RNG指纹(加载backbone权重后): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
            backbone.freeze()
        else:
            logger.warning('⚠ 使用随机初始化骨干网络')
            backbone.freeze()
        
        # Stage 4: 分类器训练
        logger.info(f"🔧 RNG指纹(Stage4调用前): {rng_fingerprint_short()} ({seed_snapshot(config, args.seed)})")
        stage4_finetune_classifier(
            backbone, X_train, y_corrected, correction_weight,
            config, logger, n_original=len(X_train), backbone_path=backbone_path,
            X_train_real=X_train
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


if __name__ == '__main__':
    main()
