"""
评估指标计算工具
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def calculate_metrics(y_true, y_pred, labels=[0, 1]):
    """计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 类别标签列表
    
    Returns:
        dict: 包含各项指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels),
    }
    
    # 每个类别的指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {
        'precision': precision_per_class,
        'recall': recall_per_class,
        'f1_score': f1_per_class,
    }
    
    return metrics


def print_metrics(metrics, logger=None):
    """打印评估指标
    
    Args:
        metrics: 指标字典
        logger: 日志记录器
    """
    def _print(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    _print("=" * 80)
    _print("评估指标")
    _print("=" * 80)
    _print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    _print(f"精确率 (Precision): {metrics['precision']:.4f}")
    _print(f"召回率 (Recall):    {metrics['recall']:.4f}")
    _print(f"F1分数 (F1-Score):  {metrics['f1_score']:.4f}")
    _print("")
    _print("混淆矩阵:")
    _print(metrics['confusion_matrix'])
    _print("")
    _print("各类别指标:")
    _print(f"  类别0 (正常) - Precision: {metrics['per_class']['precision'][0]:.4f}, "
           f"Recall: {metrics['per_class']['recall'][0]:.4f}, "
           f"F1: {metrics['per_class']['f1_score'][0]:.4f}")
    _print(f"  类别1 (恶意) - Precision: {metrics['per_class']['precision'][1]:.4f}, "
           f"Recall: {metrics['per_class']['recall'][1]:.4f}, "
           f"F1: {metrics['per_class']['f1_score'][1]:.4f}")
    _print("=" * 80)


def calculate_label_correction_metrics(original_labels, noisy_labels, corrected_labels):
    """计算标签矫正效果指标
    
    Args:
        original_labels: 原始正确标签
        noisy_labels: 含噪声标签
        corrected_labels: 矫正后标签
    
    Returns:
        dict: 矫正效果指标
    """
    original_labels = np.array(original_labels)
    noisy_labels = np.array(noisy_labels)
    corrected_labels = np.array(corrected_labels)
    
    # 噪声样本索引
    noise_indices = np.where(original_labels != noisy_labels)[0]
    clean_indices = np.where(original_labels == noisy_labels)[0]
    
    # 矫正效果
    corrected_noise = np.sum(corrected_labels[noise_indices] == original_labels[noise_indices])
    preserved_clean = np.sum(corrected_labels[clean_indices] == original_labels[clean_indices])
    
    metrics = {
        'total_samples': len(original_labels),
        'noise_samples': len(noise_indices),
        'clean_samples': len(clean_indices),
        'noise_rate': len(noise_indices) / len(original_labels),
        'corrected_noise': corrected_noise,
        'noise_correction_rate': corrected_noise / len(noise_indices) if len(noise_indices) > 0 else 0,
        'preserved_clean': preserved_clean,
        'clean_preservation_rate': preserved_clean / len(clean_indices) if len(clean_indices) > 0 else 0,
        'overall_accuracy': np.sum(corrected_labels == original_labels) / len(original_labels),
    }
    
    return metrics


def print_correction_metrics(metrics, logger=None):
    """打印标签矫正指标
    
    Args:
        metrics: 矫正指标字典
        logger: 日志记录器
    """
    def _print(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    _print("=" * 80)
    _print("标签矫正效果")
    _print("=" * 80)
    _print(f"总样本数: {metrics['total_samples']}")
    _print(f"噪声样本数: {metrics['noise_samples']}")
    _print(f"干净样本数: {metrics['clean_samples']}")
    _print(f"噪声率: {metrics['noise_rate']:.2%}")
    _print("")
    _print(f"矫正的噪声样本数: {metrics['corrected_noise']}")
    _print(f"噪声矫正率: {metrics['noise_correction_rate']:.2%}")
    _print(f"保留的干净样本数: {metrics['preserved_clean']}")
    _print(f"干净样本保留率: {metrics['clean_preservation_rate']:.2%}")
    _print(f"矫正后总体准确率: {metrics['overall_accuracy']:.2%}")
    _print("=" * 80)

