import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import json
import shutil
import argparse
from datetime import datetime

import numpy as np
import torch

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger
from MoudleCode.preprocessing.pcap_parser import load_dataset
from MoudleCode.feature_extraction.backbone import MicroBiMambaBackbone, build_backbone

try:
    from scripts.utils.preprocess import check_preprocessed_exists, load_preprocessed
    PREPROCESS_AVAILABLE = True
except Exception:
    PREPROCESS_AVAILABLE = False

from scripts.training.train import stage3_finetune_classifier
from scripts.testing.test import main as test_main


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_train_dataset():
    if PREPROCESS_AVAILABLE and check_preprocessed_exists('train'):
        X_train, y_train, _ = load_preprocessed('train')
        return X_train, y_train

    X_train, y_train, _ = load_dataset(
        benign_dir=config.BENIGN_TRAIN,
        malicious_dir=config.MALICIOUS_TRAIN,
        sequence_length=config.SEQUENCE_LENGTH,
    )
    return X_train, y_train


def _default_correction_npz_path() -> str:
    return os.path.join(config.LABEL_CORRECTION_DIR, 'analysis', 'correction_results.npz')


def _summarize_array(x: np.ndarray):
    x = np.asarray(x)
    if x.size == 0:
        return {
            'min': None,
            'max': None,
            'mean': None,
            'p25': None,
            'p50': None,
            'p75': None,
        }
    return {
        'min': float(np.min(x)),
        'max': float(np.max(x)),
        'mean': float(np.mean(x)),
        'p25': float(np.percentile(x, 25)),
        'p50': float(np.percentile(x, 50)),
        'p75': float(np.percentile(x, 75)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correction_npz', type=str, default='', help='Path to correction_results.npz')
    parser.add_argument('--use_ground_truth', action='store_true')
    parser.add_argument('--retrain_backbone', action='store_true')
    parser.add_argument('--backbone_path', type=str, default='', help='Path to backbone model (optional)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_tag', type=str, default='')
    args = parser.parse_args()

    set_seed(args.seed)
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, 'logs'), name='clean_only_train_test')

    # ----------------------------
    # Pure focal-loss mode (for clean-only overfit/debug experiments)
    # Keep gradients simple: no SoftF1 / no orthogonality / no consistency.
    # ----------------------------
    config.USE_FOCAL_LOSS = True
    config.USE_SOFT_F1_LOSS = False
    config.SOFT_F1_WEIGHT = 0.0
    config.SOFT_ORTH_WEIGHT_START = 0.0
    config.SOFT_ORTH_WEIGHT_END = 0.0
    config.CONSISTENCY_WEIGHT_START = 0.0
    config.CONSISTENCY_WEIGHT_END = 0.0

    # Enable backbone finetuning in clean-only experiments (do not overwrite original backbone file).
    config.FINETUNE_BACKBONE = True
    tmp_scope = str(getattr(config, 'FINETUNE_BACKBONE_SCOPE', 'projection')).lower()
    config.FINETUNE_BACKBONE_SCOPE = tmp_scope if tmp_scope in ('projection', 'all') else 'projection'

    # Use full training data in this mode; allow early-stopping on train metric.
    config.FINETUNE_VAL_SPLIT = 0.0
    config.FINETUNE_ES_ALLOW_TRAIN_METRIC = True

    run_tag = args.run_tag.strip() or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config.LABEL_CORRECTION_DIR, 'analysis', 'clean_only_runs', run_tag)
    _safe_makedirs(run_dir)
    _safe_makedirs(os.path.join(run_dir, 'models'))
    _safe_makedirs(os.path.join(run_dir, 'result_snapshot'))

    # Override output dirs to avoid overwriting the default training/test artifacts
    original_classification_dir = getattr(config, 'CLASSIFICATION_DIR', None)
    original_result_dir = getattr(config, 'RESULT_DIR', None)

    config.CLASSIFICATION_DIR = os.path.join(run_dir, 'classification')
    config.RESULT_DIR = os.path.join(run_dir, 'result')

    for d in [config.CLASSIFICATION_DIR, config.RESULT_DIR]:
        _safe_makedirs(d)
        _safe_makedirs(os.path.join(d, 'models'))
        _safe_makedirs(os.path.join(d, 'figures'))
        _safe_makedirs(os.path.join(d, 'logs'))

    try:

        logger.info('=' * 70)
        logger.info('🚀 干净数据训练+测试模式')
        logger.info('=' * 70)
        logger.info(f'运行目录: {run_dir}')
        logger.info(f'时间戳: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info('')
        
        # 显示训练模式
        if args.use_ground_truth:
            logger.info('📋 训练模式: 使用真实标签（无噪声）')
            logger.info('  - 数据来源: 原始训练集')
            logger.info('  - 标签: 真实标签（ground truth）')
            logger.info('  - 标签矫正: 跳过')
            logger.info('  - 数据增强: 跳过')
        else:
            logger.info('📋 训练模式: 使用标签矫正结果')
            logger.info(f'  - 矫正文件: {correction_npz}')
            logger.info('  - 标签: 矫正后的标签')
            logger.info('  - 数据增强: 跳过')
        
        logger.info('')
        logger.info('📁 输出目录（隔离）:')
        logger.info(f'  - 分类器: {config.CLASSIFICATION_DIR}')
        logger.info(f'  - 测试结果: {config.RESULT_DIR}')
        logger.info('')

        X_train, y_train_true = _load_train_dataset()
        if X_train is None:
            raise RuntimeError('Failed to load train dataset')

        if args.use_ground_truth:
            y_corrected = y_train_true.astype(int)
            action_mask = np.zeros(len(y_corrected), dtype=int)
            correction_weight = np.ones(len(y_corrected), dtype=np.float32)
            correction_npz = ''
        else:
            correction_npz = args.correction_npz.strip() or _default_correction_npz_path()
            if not os.path.exists(correction_npz):
                raise FileNotFoundError(f'correction_npz not found: {correction_npz}')
            logger.info(f'Correction npz: {correction_npz}')
            data = np.load(correction_npz, allow_pickle=True)
            y_corrected = data['y_corrected'].astype(int)
            action_mask = data['action_mask'].astype(int)
            correction_weight = data['correction_weight'].astype(np.float32)

            if len(X_train) != len(y_corrected):
                raise ValueError(f'Length mismatch: X_train={len(X_train)} vs y_corrected={len(y_corrected)}')

        keep_mask = action_mask != 2
        X_clean = X_train[keep_mask]
        y_clean = y_corrected[keep_mask]
        w_clean = correction_weight[keep_mask]

        stats = {
            'n_total': int(len(y_corrected)),
            'n_keep': int(np.sum(action_mask == 0)),
            'n_flip': int(np.sum(action_mask == 1)),
            'n_drop': int(np.sum(action_mask == 2)),
            'n_reweight': int(np.sum(action_mask == 3)),
            'n_train_used': int(len(X_clean)),
            'label_dist_corrected': {
                'benign': int(np.sum(y_clean == 0)),
                'malicious': int(np.sum(y_clean == 1)),
            },
            'weight_summary': _summarize_array(w_clean),
        }

        logger.info('📊 数据统计:')
        logger.info(f"  - 训练样本总数: {stats['n_train_used']}")
        logger.info(f"  - 正常样本: {stats['label_dist_corrected']['benign']}")
        logger.info(f"  - 恶意样本: {stats['label_dist_corrected']['malicious']}")
        
        if not args.use_ground_truth:
            logger.info('')
            logger.info('📝 标签矫正统计:')
            logger.info(f"  - 保持不变: {stats['n_keep']}")
            logger.info(f"  - 翻转标签: {stats['n_flip']}")
            logger.info(f"  - 丢弃样本: {stats['n_drop']}")
            logger.info(f"  - 重新加权: {stats['n_reweight']}")
        
        logger.info('')
        logger.info('🔧 骨干网络:')
        
        backbone = build_backbone(config, logger=logger)
        
        # 确定backbone路径：优先使用命令行参数，否则使用默认路径
        if args.backbone_path:
            backbone_path = args.backbone_path
            logger.info(f'  - 来源: 命令行指定')
        else:
            backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth')
            logger.info(f'  - 来源: 默认路径')
        
        logger.info(f'  - 路径: {backbone_path}')
        
        if os.path.exists(backbone_path) and not args.retrain_backbone:
            logger.info(f'  - 状态: ✓ 加载预训练模型')
            try:
                state_dict = torch.load(backbone_path, map_location=config.DEVICE, weights_only=True)
            except TypeError:
                state_dict = torch.load(backbone_path, map_location=config.DEVICE)
            try:
                backbone.load_state_dict(state_dict)
            except RuntimeError as e:
                logger.warning(f"⚠ 骨干网络检查点与当前结构不完全匹配，将使用 strict=False 加载: {e}")
                missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"  missing_keys: {missing}")
                if unexpected:
                    logger.warning(f"  unexpected_keys: {unexpected}")
            backbone.freeze()
        else:
            if args.retrain_backbone:
                logger.info('  - 状态: ⚠ 使用随机初始化（--retrain_backbone 指定）')
            else:
                logger.info(f'  - 状态: ⚠ 模型文件不存在，使用随机初始化')
            backbone.freeze()
        
        logger.info('')

        with open(os.path.join(run_dir, 'train_data_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        X_tr = X_clean
        y_tr = y_clean
        w_tr = w_clean

        logger.info('=' * 70)
        logger.info('🎯 Stage 3: 分类器训练')
        logger.info('=' * 70)
        logger.info('📥 输入数据:')
        logger.info(f'  - 训练样本: {len(X_tr)} 个')
        logger.info(f'  - 特征来源: 骨干网络提取')
        if args.use_ground_truth:
            logger.info(f'  - 标签来源: 真实标签（无噪声）')
        else:
            logger.info(f'  - 标签来源: 标签矫正结果')
        logger.info(f'  - 数据增强: 未使用')
        logger.info('')

        stage3_finetune_classifier(
            backbone,
            X_tr,
            y_tr,
            w_tr,
            config,
            logger,
            n_original=len(X_tr),
            backbone_path=backbone_path,
        )

        classifier_final = os.path.join(config.CLASSIFICATION_DIR, 'models', 'classifier_final.pth')
        classifier_best = os.path.join(config.CLASSIFICATION_DIR, 'models', 'classifier_best_f1.pth')
        history_npz = os.path.join(config.CLASSIFICATION_DIR, 'models', 'training_history.npz')

        logger.info('')
        logger.info('=' * 70)
        logger.info('🧪 测试评估')
        logger.info('=' * 70)
        logger.info('📥 使用模型:')
        logger.info(f'  - 骨干网络: {backbone_path}')
        logger.info(f'  - 分类器: {classifier_best}')
        logger.info('')
        
        # Prefer finetuned backbone if available (read from model metadata)
        meta_path = os.path.join(config.CLASSIFICATION_DIR, 'models', 'model_metadata.json')
        backbone_path_for_test = backbone_path
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                bp = meta.get('backbone_path', '')
                if isinstance(bp, str) and bp.strip():
                    backbone_path_for_test = bp
            except Exception:
                pass

        # 创建测试参数，传递骨干网络路径
        test_args = argparse.Namespace(backbone_path=backbone_path_for_test, classifier_path=classifier_best)
        test_main(test_args)

        result_models_dir = os.path.join(config.RESULT_DIR, 'models')
        result_figures_dir = os.path.join(config.RESULT_DIR, 'figures')

        summary = {
            'run_dir': run_dir,
            'correction_npz': correction_npz,
            'use_ground_truth': bool(args.use_ground_truth),
            'backbone_path': backbone_path,
            'classifier_final': classifier_final,
            'result_dir': config.RESULT_DIR,
            'train_data_stats': stats,
        }

        with open(os.path.join(run_dir, 'run_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info('')
        logger.info('=' * 70)
        logger.info('✅ 完成！')
        logger.info('=' * 70)
        logger.info('📁 输出文件:')
        logger.info(f'  - 运行目录: {run_dir}')
        logger.info(f'  - 分类器: {classifier_best}')
        logger.info(f'  - 测试结果: {result_models_dir}/')
        logger.info(f'  - 可视化: {result_figures_dir}/')
        logger.info(f'  - 运行摘要: {os.path.join(run_dir, "run_summary.json")}')
        logger.info('=' * 70)
    finally:
        # Restore config (best-effort) for safety if this script is imported elsewhere
        if original_classification_dir is not None:
            config.CLASSIFICATION_DIR = original_classification_dir
        if original_result_dir is not None:
            config.RESULT_DIR = original_result_dir


if __name__ == '__main__':
    main()
