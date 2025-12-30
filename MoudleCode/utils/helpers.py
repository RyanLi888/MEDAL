"""
Helper functions for MEDAL-Lite model
"""
import torch
import numpy as np
import random
import os
import logging
from datetime import datetime

from MoudleCode.utils.logging_utils import setup_logger


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inject_label_noise(labels, noise_rate, num_classes=2):
    """
    Inject symmetric label noise
    
    Args:
        labels: numpy array of original labels
        noise_rate: float, percentage of labels to flip (0-1)
        num_classes: int, number of classes
    
    Returns:
        noisy_labels: numpy array with injected noise
        noise_mask: boolean array indicating which labels were flipped
    """
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)
    
    # Create noise mask
    noise_indices = np.random.choice(n_samples, n_noisy, replace=False)
    noise_mask = np.zeros(n_samples, dtype=bool)
    noise_mask[noise_indices] = True
    
    # Create noisy labels
    noisy_labels = labels.copy()
    
    # Symmetric noise: flip to other classes uniformly
    for idx in noise_indices:
        current_label = labels[idx]
        other_labels = [l for l in range(num_classes) if l != current_label]
        noisy_labels[idx] = np.random.choice(other_labels)
    
    return noisy_labels, noise_mask


def calculate_metrics(y_true, y_pred, y_prob=None, positive_class=1):
    """
    Calculate classification metrics
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        y_prob: predicted probabilities (optional)
    
    Returns:
        dict of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix, roc_auc_score
    )
    
    # 论文口径：恶意类(positive_class)为正类的单类指标
    precision_pos = precision_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
    recall_pos = recall_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_pos': precision_pos,
        'recall_pos': recall_pos,
        'f1_pos': f1_pos,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Add AUC if probabilities are provided
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except:
            metrics['auc'] = 0.0
    
    return metrics


def print_metrics(metrics, logger=None):
    """Pretty print metrics"""
    msg = "\n" + "="*50 + "\n"
    msg += "Performance Metrics:\n"
    msg += "="*50 + "\n"
    msg += f"Accuracy:  {metrics['accuracy']:.4f}\n"
    msg += f"Precision (pos=1): {metrics['precision_pos']:.4f}\n"
    msg += f"Recall    (pos=1): {metrics['recall_pos']:.4f}\n"
    msg += f"F1 (pos=1):        {metrics['f1_pos']:.4f}  # 论文口径\n"
    msg += f"F1-Macro:          {metrics['f1_macro']:.4f}\n"
    
    if 'auc' in metrics:
        msg += f"AUC:       {metrics['auc']:.4f}\n"
    
    msg += "\nConfusion Matrix:\n"
    msg += str(metrics['confusion_matrix']) + "\n"
    msg += "="*50 + "\n"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return msg


def find_optimal_threshold(y_true, y_prob, metric='f1_binary', positive_class=1):
    """
    基于验证集自动寻找最优决策阈值
    
    核心指标：Binary F1-Score (针对恶意类，positive_class=1)
    F1 = 2 × (Precision × Recall) / (Precision + Recall)
    
    策略改进：使用 Precision-Recall 曲线直接寻找 F1 峰值
    - 之前的方法（约登指数）倾向于优化 Accuracy，牺牲恶意类召回率
    - 新方法直接最大化 Binary F1-Score，在 Precision 和 Recall 之间找到最佳平衡
    
    Args:
        y_true: (N,) 真实标签
        y_prob: (N, 2) 或 (N,) 预测概率
               - 如果是 (N, 2)，取 positive_class 列的概率
               - 如果是 (N,)，直接使用
        metric: str, 优化指标
                - 'f1_binary': Binary F1-Score (针对 positive_class)【默认，推荐】
                - 'youden': 约登指数 (TPR - FPR)
                - 'f1_macro': 宏平均F1
        positive_class: int, 正类标签（默认1，恶意流量）
    
    Returns:
        optimal_threshold: float, 最优阈值
        optimal_metric: float, 最优指标值
        threshold_metrics: dict, 所有阈值对应的指标值
    """
    from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score
    
    # 提取正类概率
    if len(y_prob.shape) == 2:
        if y_prob.shape[1] == 2:
            prob_positive = y_prob[:, positive_class]
        else:
            raise ValueError(f"y_prob shape {y_prob.shape} not supported")
    else:
        prob_positive = y_prob
    
    if metric == 'f1_binary':
        # 🚀 策略优化：使用 Precision-Recall 曲线直接寻找 F1 峰值
        precision, recall, thresholds = precision_recall_curve(y_true, prob_positive)
        
        # 计算每个阈值对应的 F1 Score
        # F1 = 2 * (P * R) / (P + R)
        # 注意：precision_recall_curve 返回的数组比 thresholds 多一个元素
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        
        # 找到 F1 最高的点
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # 构建返回的 metrics_dict（包含所有阈值点）
        metrics_dict = {
            'threshold': thresholds.tolist(),
            'precision': precision[:-1].tolist(),
            'recall': recall[:-1].tolist(),
            'f1_binary': f1_scores.tolist()
        }
        
        return best_threshold, best_f1, metrics_dict
    
    else:
        # 对于其他指标，使用原有的网格搜索方法
        thresholds = np.arange(0.01, 1.0, 0.01)
        metrics_dict = {
            'threshold': [],
            'youden_j': [],
            'tpr': [],
            'fpr': [],
            'f1_binary': [],
            'f1_macro': [],
            'precision': [],
            'recall': []
        }
        
        best_threshold = 0.5
        best_metric = -np.inf
        
        for threshold in thresholds:
            # 根据阈值生成预测
            y_pred = (prob_positive >= threshold).astype(int)
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            # 计算各项指标
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr
            
            # 约登指数
            youden_j = tpr - fpr
            
            # Binary F1分数
            f1_binary = f1_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
            
            # Macro F1分数
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            # 保存指标
            metrics_dict['threshold'].append(threshold)
            metrics_dict['youden_j'].append(youden_j)
            metrics_dict['tpr'].append(tpr)
            metrics_dict['fpr'].append(fpr)
            metrics_dict['f1_binary'].append(f1_binary)
            metrics_dict['f1_macro'].append(f1_macro)
            metrics_dict['precision'].append(precision)
            metrics_dict['recall'].append(recall)
            
            # 根据选择的指标找到最优阈值
            if metric == 'youden':
                current_metric = youden_j
            elif metric == 'f1_macro':
                current_metric = f1_macro
            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'f1_binary', 'youden', or 'f1_macro'")
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_threshold = threshold
        
        return best_threshold, best_metric, metrics_dict


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


class EarlyStopping:
    """Early stopping to stop training when validation performance stops improving"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: how many epochs to wait before stopping
            min_delta: minimum change to qualify as an improvement
            mode: 'max' or 'min' - whether higher or lower is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


def encode_tcp_flags(flags_dict):
    """
    Encode TCP flags to normalized float value
    
    Args:
        flags_dict: dict with keys 'SYN', 'FIN', 'RST', 'PSH', 'ACK'
    
    Returns:
        float in [0, 1]
    """
    syn = 1.0 if flags_dict.get('SYN', False) else 0.0
    fin = 1.0 if flags_dict.get('FIN', False) else 0.0
    rst = 1.0 if flags_dict.get('RST', False) else 0.0
    psh = 1.0 if flags_dict.get('PSH', False) else 0.0
    ack = 1.0 if flags_dict.get('ACK', False) else 0.0
    
    val = (syn * 16 + fin * 8 + rst * 4 + psh * 2 + ack) / 31.0
    return val

