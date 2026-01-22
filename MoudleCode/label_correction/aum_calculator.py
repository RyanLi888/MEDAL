"""
AUM (Area Under the Margin) Calculator
=======================================

AUM 是一种基于训练动态的噪声检测指标，通过追踪样本在训练过程中的 Margin 变化来识别噪声。

核心原理：
- Margin = logit[true_label] - max(logit[other_labels])
- 干净样本：Margin 在训练过程中持续增大（学得快且稳定）
- 噪声样本：Margin 波动大或持续为负（学得慢且不稳定）

参考文献：
Pleiss et al. "Identifying Mislabeled Data using the Area Under the Margin Ranking" (NeurIPS 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """
    轻量级线性分类头（探针）
    
    用于在冻结的骨干网络特征上训练，以计算 AUM 分数。
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        初始化线性探针
        
        参数:
            input_dim: 输入特征维度
            num_classes: 分类类别数
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征 (B, input_dim)
            
        返回:
            logits: (B, num_classes)
        """
        return self.fc(x)


class AUMCalculator:
    """
    AUM 计算器
    
    通过训练一个轻量级探针来追踪样本的 Margin 变化，计算 AUM 分数。
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        num_epochs: int = 30,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        device: str = 'cpu'
    ):
        """
        初始化 AUM 计算器
        
        参数:
            num_classes: 分类类别数
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            device: 计算设备
        """
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        self.probe = None
        self.margin_history = None  # (n_samples, n_epochs)
        
    def compute_margin(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Margin
        
        Margin = logit[true_label] - max(logit[other_labels])
        
        参数:
            logits: (B, num_classes) 模型输出
            labels: (B,) 标签
            
        返回:
            margins: (B,) 每个样本的 Margin
        """
        batch_size = logits.size(0)
        
        # 获取真实标签的 logit
        true_logits = logits[torch.arange(batch_size), labels]
        
        # 创建掩码，排除真实标签
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = False
        
        # 获取其他类别的最大 logit
        other_logits = logits.masked_fill(~mask, float('-inf'))
        max_other_logits = other_logits.max(dim=1)[0]
        
        # 计算 Margin
        margins = true_logits - max_other_logits
        
        return margins
    
    def fit(
        self,
        features: np.ndarray,
        noisy_labels: np.ndarray,
        verbose: bool = True
    ) -> np.ndarray:
        """
        训练探针并计算 AUM 分数
        
        参数:
            features: (n_samples, feature_dim) 特征矩阵
            noisy_labels: (n_samples,) 噪声标签
            verbose: 是否输出详细日志
            
        返回:
            aum_scores: (n_samples,) AUM 分数（越高越可能是干净样本）
        """
        n_samples, feature_dim = features.shape
        
        if verbose:
            logger.info("")
            logger.info("="*70)
            logger.info("AUM Calculator: 训练探针并计算 AUM 分数")
            logger.info("="*70)
            logger.info(f"  样本数: {n_samples}")
            logger.info(f"  特征维度: {feature_dim}")
            logger.info(f"  训练轮数: {self.num_epochs}")
            logger.info(f"  批次大小: {self.batch_size}")
        
        # 初始化探针
        self.probe = LinearProbe(feature_dim, self.num_classes).to(self.device)
        optimizer = optim.SGD(self.probe.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # 准备数据
        features_tensor = torch.FloatTensor(features).to(self.device)
        labels_tensor = torch.LongTensor(noisy_labels).to(self.device)
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化 Margin 历史记录
        self.margin_history = np.zeros((n_samples, self.num_epochs))
        
        # 训练循环
        self.probe.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_margins = np.zeros(n_samples)
            epoch_counts = np.zeros(n_samples)
            
            for batch_features, batch_labels in dataloader:
                # 前向传播
                logits = self.probe(batch_features)
                loss = criterion(logits, batch_labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 计算当前 epoch 所有样本的 Margin
            with torch.no_grad():
                all_logits = self.probe(features_tensor)
                all_margins = self.compute_margin(all_logits, labels_tensor)
                self.margin_history[:, epoch] = all_margins.cpu().numpy()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                avg_margin = self.margin_history[:, epoch].mean()
                logger.info(f"  Epoch {epoch+1}/{self.num_epochs} | Loss: {avg_loss:.4f} | Avg Margin: {avg_margin:.4f}")
        
        # 计算 AUM 分数（Margin 的平均值）
        aum_scores = self.margin_history.mean(axis=1)
        
        if verbose:
            logger.info("")
            logger.info("  ✓ AUM 计算完成")
            logger.info(f"    AUM 分数范围: [{aum_scores.min():.4f}, {aum_scores.max():.4f}]")
            logger.info(f"    AUM 分数均值: {aum_scores.mean():.4f}")
            logger.info(f"    AUM 分数标准差: {aum_scores.std():.4f}")
        
        return aum_scores
    
    def get_margin_history(self) -> Optional[np.ndarray]:
        """
        获取 Margin 历史记录
        
        返回:
            margin_history: (n_samples, n_epochs) 或 None
        """
        return self.margin_history
    
    def analyze_aum_distribution(
        self,
        aum_scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        noisy_labels: Optional[np.ndarray] = None
    ) -> dict:
        """
        分析 AUM 分数的分布特性
        
        参数:
            aum_scores: (n_samples,) AUM 分数
            y_true: (n_samples,) 真实标签（可选）
            noisy_labels: (n_samples,) 噪声标签（可选）
            
        返回:
            analysis: 分析结果字典
        """
        analysis = {
            'mean': float(aum_scores.mean()),
            'std': float(aum_scores.std()),
            'min': float(aum_scores.min()),
            'max': float(aum_scores.max()),
            'median': float(np.median(aum_scores)),
            'q25': float(np.percentile(aum_scores, 25)),
            'q75': float(np.percentile(aum_scores, 75))
        }
        
        # 如果有真实标签，分析干净样本和噪声样本的 AUM 分布
        if y_true is not None and noisy_labels is not None:
            is_noise = (y_true != noisy_labels)
            
            clean_aum = aum_scores[~is_noise]
            noise_aum = aum_scores[is_noise]
            
            analysis['clean_mean'] = float(clean_aum.mean()) if len(clean_aum) > 0 else 0.0
            analysis['clean_std'] = float(clean_aum.std()) if len(clean_aum) > 0 else 0.0
            analysis['noise_mean'] = float(noise_aum.mean()) if len(noise_aum) > 0 else 0.0
            analysis['noise_std'] = float(noise_aum.std()) if len(noise_aum) > 0 else 0.0
            
            # 计算相关性
            if len(noise_aum) > 0:
                # AUM 与是否噪声的相关性（期望负相关：噪声样本 AUM 低）
                correlation = np.corrcoef(aum_scores, is_noise.astype(int))[0, 1]
                analysis['correlation_with_noise'] = float(correlation)
        
        return analysis
