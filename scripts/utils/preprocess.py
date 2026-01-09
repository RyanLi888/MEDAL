"""
MEDAL-Lite 数据预处理脚本
========================

将PCAP文件预处理为.npy文件，供训练和测试直接使用。
预处理后的文件保存在 output/preprocessed/ 目录下。

使用方法:
    # 预处理所有数据（训练集+测试集）
    python preprocess.py
    
    # 仅预处理训练集
    python preprocess.py --train_only
    
    # 仅预处理测试集
    python preprocess.py --test_only
    
    # 强制重新预处理（覆盖已有文件）
    python preprocess.py --force

预处理后的文件:
    output/preprocessed/
    ├── train_X.npy      # 训练集特征 (N, 1024, 4)
    ├── train_y.npy      # 训练集标签 (N,)
    ├── train_files.npy  # 训练集文件名
    ├── test_X.npy       # 测试集特征 (M, 1024, 4)
    ├── test_y.npy       # 测试集标签 (M,)
    └── test_files.npy   # 测试集文件名
"""
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path when running as a script
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import argparse
import time
from datetime import datetime

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import setup_logger
from MoudleCode.preprocessing.pcap_parser import load_dataset

import logging

logger = setup_logger(os.path.join(config.OUTPUT_ROOT, "logs"), name='preprocess')


def get_preprocessed_dir():
    """获取预处理数据目录"""
    preprocessed_dir = os.path.join(config.OUTPUT_ROOT, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    return preprocessed_dir


def check_preprocessed_exists(data_type='train'):
    """
    检查预处理文件是否存在
    
    Args:
        data_type: 'train' 或 'test'
        
    Returns:
        bool: 是否存在完整的预处理文件
    """
    preprocessed_dir = get_preprocessed_dir()
    
    required_files = [
        f'{data_type}_X.npy',
        f'{data_type}_y.npy',
        f'{data_type}_files.npy'
    ]
    
    for f in required_files:
        if not os.path.exists(os.path.join(preprocessed_dir, f)):
            return False
    return True


def load_preprocessed(data_type='train'):
    """
    加载预处理后的数据
    
    Args:
        data_type: 'train' 或 'test'
        
    Returns:
        X: numpy array (N, L, 4)
        y: numpy array (N,)
        files: list of filenames
    """
    preprocessed_dir = get_preprocessed_dir()
    
    X = np.load(os.path.join(preprocessed_dir, f'{data_type}_X.npy'))
    y = np.load(os.path.join(preprocessed_dir, f'{data_type}_y.npy'))
    files = np.load(os.path.join(preprocessed_dir, f'{data_type}_files.npy'), allow_pickle=True)
    
    return X, y, files.tolist()


def normalize_burstsize_inplace(X: np.ndarray) -> np.ndarray:
    try:
        enabled = bool(getattr(config, 'BURSTSIZE_NORMALIZE', False))
    except Exception:
        enabled = False
    if not enabled:
        return X

    try:
        burst_idx = int(getattr(config, 'BURST_SIZE_INDEX', 2))
        vm_idx = int(getattr(config, 'VALID_MASK_INDEX', 3))
        denom = float(getattr(config, 'BURSTSIZE_NORM_DENOM', 1.0))
    except Exception:
        return X

    if denom <= 0:
        return X
    if X is None or getattr(X, 'ndim', 0) != 3:
        return X
    if burst_idx < 0 or burst_idx >= X.shape[-1] or vm_idx < 0 or vm_idx >= X.shape[-1]:
        return X

    mask = X[:, :, vm_idx] > 0.5
    X[:, :, burst_idx][mask] = X[:, :, burst_idx][mask] / denom
    return X


def save_preprocessed(X, y, files, data_type='train'):
    """
    保存预处理后的数据
    
    Args:
        X: numpy array (N, L, 4)
        y: numpy array (N,)
        files: list of filenames
        data_type: 'train' 或 'test'
    """
    preprocessed_dir = get_preprocessed_dir()
    
    np.save(os.path.join(preprocessed_dir, f'{data_type}_X.npy'), X)
    np.save(os.path.join(preprocessed_dir, f'{data_type}_y.npy'), y)
    np.save(os.path.join(preprocessed_dir, f'{data_type}_files.npy'), np.array(files, dtype=object))
    
    logger.info(f"✓ {data_type}数据已保存到 {preprocessed_dir}/")
    logger.info(f"  {data_type}_X.npy: {X.shape}")
    logger.info(f"  {data_type}_y.npy: {y.shape}")
    logger.info(f"  {data_type}_files.npy: {len(files)} 个文件名")


def preprocess_train(force=False):
    """
    预处理训练集
    
    Args:
        force: 是否强制重新预处理
        
    Returns:
        X, y, files
    """
    if not force and check_preprocessed_exists('train'):
        logger.info("✓ 训练集预处理文件已存在，跳过预处理")
        return load_preprocessed('train')
    
    logger.info("="*70)
    logger.info("📦 预处理训练集 PCAP 文件")
    logger.info("="*70)
    logger.info(f"正常流量目录: {config.BENIGN_TRAIN}")
    logger.info(f"恶意流量目录: {config.MALICIOUS_TRAIN}")
    logger.info("")
    
    start_time = time.time()
    
    X_train, y_train, train_files = load_dataset(
        benign_dir=config.BENIGN_TRAIN,
        malicious_dir=config.MALICIOUS_TRAIN,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    elapsed = time.time() - start_time
    
    if X_train is None:
        logger.error("❌ 训练集预处理失败!")
        return None, None, None
    
    logger.info("")
    logger.info(f"⏱️  训练集预处理完成，耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    logger.info(f"  数据形状: {X_train.shape}")
    logger.info(f"  正常样本: {(y_train==0).sum()}")
    logger.info(f"  恶意样本: {(y_train==1).sum()}")
    
    # 保存预处理结果
    save_preprocessed(X_train, y_train, train_files, 'train')
    
    return X_train, y_train, train_files


def preprocess_test(force=False):
    """
    预处理测试集
    
    Args:
        force: 是否强制重新预处理
        
    Returns:
        X, y, files
    """
    if not force and check_preprocessed_exists('test'):
        logger.info("✓ 测试集预处理文件已存在，跳过预处理")
        return load_preprocessed('test')
    
    logger.info("="*70)
    logger.info("📦 预处理测试集 PCAP 文件")
    logger.info("="*70)
    logger.info(f"正常流量目录: {config.BENIGN_TEST}")
    logger.info(f"恶意流量目录: {config.MALICIOUS_TEST}")
    logger.info("")
    logger.info("⚠️  测试集可能包含大文件，预计耗时较长...")
    logger.info("")
    
    start_time = time.time()
    
    X_test, y_test, test_files = load_dataset(
        benign_dir=config.BENIGN_TEST,
        malicious_dir=config.MALICIOUS_TEST,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    elapsed = time.time() - start_time
    
    if X_test is None:
        logger.error("❌ 测试集预处理失败!")
        return None, None, None
    
    logger.info("")
    logger.info(f"⏱️  测试集预处理完成，耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    logger.info(f"  数据形状: {X_test.shape}")
    logger.info(f"  正常样本: {(y_test==0).sum()}")
    logger.info(f"  恶意样本: {(y_test==1).sum()}")
    
    # 保存预处理结果
    save_preprocessed(X_test, y_test, test_files, 'test')
    
    return X_test, y_test, test_files


def main(args):
    """主函数"""
    logger.info("="*70)
    logger.info("🚀 MEDAL-Lite 数据预处理")
    logger.info("="*70)
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"强制重新预处理: {args.force}")
    logger.info("")
    
    total_start = time.time()
    
    # 预处理训练集
    if not args.test_only:
        X_train, y_train, train_files = preprocess_train(force=args.force)
        if X_train is None:
            return
    
    # 预处理测试集
    if not args.train_only:
        X_test, y_test, test_files = preprocess_test(force=args.force)
        if X_test is None:
            return
    
    total_elapsed = time.time() - total_start
    
    logger.info("")
    logger.info("="*70)
    logger.info("🎉 预处理完成!")
    logger.info("="*70)
    logger.info(f"总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    logger.info("")
    logger.info("📁 预处理文件位置:")
    logger.info(f"  {get_preprocessed_dir()}/")
    logger.info("")
    logger.info("💡 后续使用:")
    logger.info("  训练: python train.py")
    logger.info("  测试: python test.py")
    logger.info("  完整流程: python all_train_test.py")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEDAL-Lite 数据预处理")
    
    parser.add_argument("--train_only", action="store_true",
                       help="仅预处理训练集")
    parser.add_argument("--test_only", action="store_true",
                       help="仅预处理测试集")
    parser.add_argument("--force", action="store_true",
                       help="强制重新预处理（覆盖已有文件）")
    
    args = parser.parse_args()
    
    main(args)
