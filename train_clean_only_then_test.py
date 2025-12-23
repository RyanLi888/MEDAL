import os
import sys
import json
import shutil
import argparse
from datetime import datetime

import numpy as np
import torch

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger
from MoudleCode.preprocessing.pcap_parser import load_dataset
from MoudleCode.feature_extraction.backbone import MicroBiMambaBackbone

try:
    from preprocess import check_preprocessed_exists, load_preprocessed
    PREPROCESS_AVAILABLE = True
except Exception:
    PREPROCESS_AVAILABLE = False

from train import stage3_finetune_classifier
from test import main as test_main


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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_tag', type=str, default='')
    args = parser.parse_args()

    set_seed(args.seed)
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, 'logs'), name='clean_only_train_test')

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
        logger.info('CLEAN-ONLY TRAINING + TESTING (no augmentation)')
        logger.info('=' * 70)
        logger.info(f'Run dir: {run_dir}')
        logger.info(f'Use ground truth: {bool(args.use_ground_truth)}')
        logger.info(f'Output classification dir (isolated): {config.CLASSIFICATION_DIR}')
        logger.info(f'Output result dir (isolated): {config.RESULT_DIR}')

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

        with open(os.path.join(run_dir, 'train_data_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info('Data stats saved: train_data_stats.json')
        logger.info(f"Training samples: {stats['n_train_used']} | benign={stats['label_dist_corrected']['benign']} malicious={stats['label_dist_corrected']['malicious']}")

        backbone = MicroBiMambaBackbone(config)
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth')
        if os.path.exists(backbone_path) and not args.retrain_backbone:
            logger.info(f'Loading backbone: {backbone_path}')
            backbone.load_state_dict(torch.load(backbone_path, map_location=config.DEVICE))
            backbone.freeze()
        else:
            logger.warning('Backbone checkpoint not found or --retrain_backbone specified; using randomly initialized backbone')
            backbone.freeze()

        X_tr = X_clean
        y_tr = y_clean
        w_tr = w_clean

        logger.info('=' * 70)
        logger.info('Stage 3 (classifier fine-tune) on corrected clean data')
        logger.info('=' * 70)

        stage3_finetune_classifier(
            backbone,
            X_tr,
            y_tr,
            w_tr,
            config,
            logger,
            n_original=len(X_tr),
        )

        classifier_final = os.path.join(config.CLASSIFICATION_DIR, 'models', 'classifier_final.pth')
        classifier_best = os.path.join(config.CLASSIFICATION_DIR, 'models', 'classifier_best_f1.pth')
        history_npz = os.path.join(config.CLASSIFICATION_DIR, 'models', 'training_history.npz')

        logger.info('=' * 70)
        logger.info('Testing')
        logger.info('=' * 70)

        test_main(argparse.Namespace())

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

        logger.info('=' * 70)
        logger.info('Done')
        logger.info(f'Run dir: {run_dir}')
        logger.info('=' * 70)
    finally:
        # Restore config (best-effort) for safety if this script is imported elsewhere
        if original_classification_dir is not None:
            config.CLASSIFICATION_DIR = original_classification_dir
        if original_result_dir is not None:
            config.RESULT_DIR = original_result_dir


if __name__ == '__main__':
    main()
