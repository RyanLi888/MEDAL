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
import hashlib
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger
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

from scripts.training.train import stage1_pretrain_backbone, stage2_label_correction_and_augmentation, stage3_finetune_classifier
from scripts.testing.test import main as test_main


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


def _advance_rng_to_stage2_baseline(logger, args_seed: int) -> None:
    """Align RNG state to match the non-retrain pipeline right before Stage2.

    In the non-retrain run, RNG typically advances due to backbone initialization.
    We reproduce that consumption by building a throwaway backbone once.
    """
    logger.info(f"🔧 RNG对齐(Stage2基线) - 起点: {_rng_fingerprint_short()} ({_seed_snapshot(args_seed)})")
    _tmp = build_backbone(config, logger=logger)
    del _tmp
    logger.info(f"🔧 RNG对齐(Stage2基线) - 重放后: {_rng_fingerprint_short()} ({_seed_snapshot(args_seed)})")


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


def main():
    parser = argparse.ArgumentParser(description="数据增强训练+测试")
    parser.add_argument('--retrain_backbone', action='store_true', help='重新训练骨干网络')
    parser.add_argument('--backbone_path', type=str, default='', help='骨干网络路径')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_tag', type=str, default='')
    args = parser.parse_args()

    rng_fp_before_seed = _rng_fingerprint_short()
    set_seed(args.seed)
    rng_fp_after_seed = _rng_fingerprint_short()
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, 'logs'), name='augmented_train_test')

    logger.info(f"🔧 RNG指纹(seed前): {rng_fp_before_seed}")
    logger.info(f"🔧 RNG指纹(seed后): {rng_fp_after_seed} ({_seed_snapshot(args.seed)})")

    # 配置：使用最优参数
    config.USE_FOCAL_LOSS = True
    config.USE_BCE_LOSS = False
    config.USE_SOFT_F1_LOSS = False
    config.STAGE3_ONLINE_AUGMENTATION = False
    config.STAGE3_USE_ST_MIXUP = False
    config.FINETUNE_BACKBONE = False  # 特征空间数据不支持骨干微调
    config.FINETUNE_VAL_SPLIT = 0.0
    config.FINETUNE_ES_ALLOW_TRAIN_METRIC = True

    run_tag = args.run_tag.strip() or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config.LABEL_CORRECTION_DIR, 'analysis', 'augmented_runs', run_tag)
    _safe_makedirs(run_dir)
    
    # 隔离输出目录
    original_classification_dir = config.CLASSIFICATION_DIR
    original_result_dir = config.RESULT_DIR
    original_data_aug_dir = config.DATA_AUGMENTATION_DIR
    
    config.CLASSIFICATION_DIR = os.path.join(run_dir, 'classification')
    config.RESULT_DIR = os.path.join(run_dir, 'result')
    config.DATA_AUGMENTATION_DIR = os.path.join(run_dir, 'data_augmentation')
    
    for d in [config.CLASSIFICATION_DIR, config.RESULT_DIR, config.DATA_AUGMENTATION_DIR]:
        _safe_makedirs(d)
        _safe_makedirs(os.path.join(d, 'models'))
        _safe_makedirs(os.path.join(d, 'figures'))
        _safe_makedirs(os.path.join(d, 'logs'))

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
        logger.info('  - 骨干微调: 取决于混合训练模式')
        logger.info('')
        
        # 输出配置
        config.log_stage_config(logger, "Stage 2")
        config.log_stage_config(logger, "Stage 3")
        
        # 加载数据
        logger.info(f"🔧 RNG指纹(加载数据前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        X_train, y_train_true = _load_train_dataset()
        logger.info(f"🔧 RNG指纹(加载数据后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
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
            logger.info(f"🔧 RNG指纹(Stage1分支进入): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")

            logger.info(f"🔧 RNG指纹(Stage1-构建backbone前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            backbone = build_backbone(config, logger=logger)
            logger.info(f"🔧 RNG指纹(Stage1-构建backbone后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")

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
            logger.info(f"🔧 RNG指纹(Stage1-DataLoader创建前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            logger.info(f"🔧 RNG指纹(Stage1-DataLoader创建后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")

            backbone = backbone.to(config.DEVICE)
            logger.info(f"🔧 RNG指纹(Stage1训练前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            backbone, _pretrain_history = stage1_pretrain_backbone(backbone, train_loader, config, logger)
            logger.info(f"🔧 RNG指纹(Stage1训练后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")

            reseed_fp_before = _rng_fingerprint_short()
            set_seed(args.seed)
            reseed_fp_after = _rng_fingerprint_short()
            logger.info(
                f"🔧 RNG指纹(Stage1后重置seed前/后): {reseed_fp_before} -> {reseed_fp_after} ({_seed_snapshot(args.seed)})"
            )
            _advance_rng_to_stage2_baseline(logger, args.seed)

            default_backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth')
            if args.backbone_path:
                logger.info(f"🔧 RNG指纹(Stage1保存前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
                torch.save(backbone.state_dict(), args.backbone_path)
                backbone_path = args.backbone_path
                logger.info(f'✓ 已保存新骨干网络: {backbone_path}')
                logger.info(f"🔧 RNG指纹(Stage1保存后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            else:
                backbone_path = default_backbone_path
                logger.info(f'✓ 已保存新骨干网络: {backbone_path}')
            logger.info(f"🔧 RNG指纹(Stage1分支结束/进入Stage2前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        else:
            logger.info(f"🔧 RNG指纹(构建backbone前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
            backbone = build_backbone(config, logger=logger)
            logger.info(f"🔧 RNG指纹(构建backbone后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")

            if os.path.exists(backbone_path):
                logger.info(f'✓ 加载骨干网络: {backbone_path}')
                try:
                    state_dict = torch.load(backbone_path, map_location=config.DEVICE, weights_only=True)
                except TypeError:
                    state_dict = torch.load(backbone_path, map_location=config.DEVICE)
                backbone.load_state_dict(state_dict, strict=False)
                backbone.freeze()
            else:
                logger.warning('⚠ 使用随机初始化骨干网络')
                backbone.freeze()
        
        # Stage 2: 数据增强（跳过标签矫正）
        logger.info(f"🔧 RNG指纹(Stage2调用前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        Z_aug, y_aug, w_aug, correction_stats, tabddpm, n_original = stage2_label_correction_and_augmentation(
            backbone, X_train, y_corrected, y_corrected, config, logger,
            stage2_mode='clean_augment_only'
        )
        logger.info(f"🔧 RNG指纹(Stage2返回后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")

        # Stage 3: 分类器训练
        logger.info(f"🔧 RNG指纹(Stage3调用前): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")
        # 加载原始序列用于混合训练
        real_kept_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "real_kept_data.npz")
        X_real = None
        use_mixed_stream = bool(getattr(config, 'STAGE3_MIXED_STREAM', True))
        
        if os.path.exists(real_kept_path) and use_mixed_stream:
            logger.info(f'✓ 加载原始序列: {real_kept_path}')
            real_data = np.load(real_kept_path)
            X_real = real_data['X_real']
            logger.info(f'  - 原始序列形状: {X_real.shape}')
        else:
            use_mixed_stream = False
            logger.info('⚠️ 混合训练模式已禁用')

        stage3_finetune_classifier(
            backbone, Z_aug, y_aug, w_aug, config, logger,
            n_original=n_original, backbone_path=backbone_path,
            X_train_real=X_real, use_mixed_stream=use_mixed_stream
        )
        logger.info(f"🔧 RNG指纹(Stage3返回后): {_rng_fingerprint_short()} ({_seed_snapshot(args.seed)})")

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
