"""
CL+AUM 双重校验 - 快速测试脚本
================================

使用合成数据快速测试 CL+AUM 双重校验机制。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import logging
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from MoudleCode.label_correction import HybridCourt
from MoudleCode.utils.config import config
from MoudleCode.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 128,
    noise_rate: float = 0.3,
    random_state: int = 42
):
    """
    生成合成数据
    
    参数:
        n_samples: 样本数
        n_features: 特征维度
        noise_rate: 噪声率
        random_state: 随机种子
        
    返回:
        features: (n_samples, n_features)
        noisy_labels: (n_samples,)
        true_labels: (n_samples,)
    """
    logger.info("="*70)
    logger.info("生成合成数据")
    logger.info("="*70)
    
    # 生成二分类数据
    features, true_labels = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.1),
        n_classes=2,
        class_sep=2.0,
        random_state=random_state
    )
    
    logger.info(f"  ✓ 特征: {features.shape}")
    logger.info(f"  ✓ 标签: {true_labels.shape}")
    logger.info(f"  ✓ 类别分布: {np.bincount(true_labels)}")
    
    # 添加标签噪声
    noisy_labels = true_labels.copy()
    n_noise = int(n_samples * noise_rate)
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    noisy_labels[noise_indices] = 1 - noisy_labels[noise_indices]  # 翻转标签
    
    actual_noise_rate = (noisy_labels != true_labels).mean()
    logger.info(f"  ✓ 噪声率: {actual_noise_rate:.2%} ({(noisy_labels != true_labels).sum()}/{n_samples})")
    
    return features, noisy_labels, true_labels


def evaluate_correction(
    noisy_labels: np.ndarray,
    clean_labels: np.ndarray,
    true_labels: np.ndarray,
    action_mask: np.ndarray
):
    """
    评估矫正效果
    
    参数:
        noisy_labels: 噪声标签
        clean_labels: 矫正后的标签
        true_labels: 真实标签
        action_mask: 动作掩码 (0=Keep, 1=Flip, 2=Drop)
    """
    logger.info("\n" + "="*70)
    logger.info("评估矫正效果")
    logger.info("="*70)
    
    n_samples = len(true_labels)
    is_noise = (noisy_labels != true_labels)
    
    # 整体指标
    original_accuracy = (noisy_labels == true_labels).mean()
    corrected_accuracy = (clean_labels == true_labels).mean()
    improvement = corrected_accuracy - original_accuracy
    
    logger.info(f"\n  📊 整体指标:")
    logger.info(f"    原始准确率: {original_accuracy:.2%}")
    logger.info(f"    矫正准确率: {corrected_accuracy:.2%}")
    logger.info(f"    提升: {improvement:+.2%}")
    
    # 动作统计
    n_keep = (action_mask == 0).sum()
    n_flip = (action_mask == 1).sum()
    n_drop = (action_mask == 2).sum()
    
    logger.info(f"\n  📊 动作统计:")
    logger.info(f"    Keep: {n_keep:5d} ({100*n_keep/n_samples:.1f}%)")
    logger.info(f"    Flip: {n_flip:5d} ({100*n_flip/n_samples:.1f}%)")
    logger.info(f"    Drop: {n_drop:5d} ({100*n_drop/n_samples:.1f}%)")
    
    # Flip 详细分析
    if n_flip > 0:
        flip_mask = (action_mask == 1)
        flip_correct = ((clean_labels == true_labels) & flip_mask).sum()
        flip_wrong = n_flip - flip_correct
        
        # 真阳性：正确识别并翻转噪声
        true_positive = ((clean_labels == true_labels) & flip_mask & is_noise).sum()
        # 假阳性：错误翻转干净样本
        false_positive = flip_wrong
        # 假阴性：未识别的噪声
        false_negative = (is_noise & (action_mask == 0)).sum()
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / is_noise.sum() if is_noise.sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"\n  📊 Flip 详细分析:")
        logger.info(f"    翻转总数: {n_flip}")
        logger.info(f"    翻转正确: {flip_correct} ({100*flip_correct/n_flip:.1f}%)")
        logger.info(f"    翻转错误: {flip_wrong} ({100*flip_wrong/n_flip:.1f}%)")
        logger.info(f"    真阳性 (TP): {true_positive}")
        logger.info(f"    假阳性 (FP): {false_positive}")
        logger.info(f"    假阴性 (FN): {false_negative}")
        logger.info(f"    精确率 (Precision): {precision:.3f}")
        logger.info(f"    召回率 (Recall): {recall:.3f}")
        logger.info(f"    F1 分数: {f1:.3f}")
    
    # Keep 纯度
    keep_mask = (action_mask == 0)
    if keep_mask.sum() > 0:
        keep_correct = ((clean_labels == true_labels) & keep_mask).sum()
        keep_purity = keep_correct / keep_mask.sum()
        logger.info(f"\n  📊 Keep 纯度: {keep_purity:.2%} ({keep_correct}/{keep_mask.sum()})")


def main():
    """主函数"""
    # 设置日志
    setup_logger()
    
    logger.info("="*70)
    logger.info("CL+AUM 双重校验 - 快速测试")
    logger.info("="*70)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  设备: {device}")
    
    # 生成合成数据
    features, noisy_labels, true_labels = generate_synthetic_data(
        n_samples=1000,
        n_features=128,
        noise_rate=0.3,
        random_state=42
    )
    
    # 初始化 HybridCourt
    logger.info("\n" + "="*70)
    logger.info("初始化 HybridCourt")
    logger.info("="*70)

    config.NUM_CLASSES = 2
    config.FEATURE_DIM = int(features.shape[1])
    config.KNN_NEIGHBORS = 10
    config.DEVICE = device

    config.STAGE2_CL_THRESHOLD = 0.7
    config.STAGE2_AUM_THRESHOLD = 0.0
    config.AUM_EPOCHS = 20
    config.AUM_BATCH_SIZE = 128
    config.AUM_LR = 0.01
    config.STAGE2_KNN_PURITY_THRESHOLD = 0.8
    config.STAGE2_USE_DROP = False

    hybrid_court = HybridCourt(config)
    
    logger.info(f"  ✓ HybridCourt 初始化完成")
    
    # 执行 CL+AUM 双重校验
    logger.info("\n" + "="*70)
    logger.info("执行 CL+AUM 双重校验")
    logger.info("="*70)
    
    clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs = \
        hybrid_court.correct_labels(
            features=features,
            noisy_labels=noisy_labels,
            device=device,
            y_true=true_labels,
        )
    
    # 评估矫正效果
    evaluate_correction(noisy_labels, clean_labels, true_labels, action_mask)
    
    logger.info("\n" + "="*70)
    logger.info("测试完成！")
    logger.info("="*70)


if __name__ == '__main__':
    main()
