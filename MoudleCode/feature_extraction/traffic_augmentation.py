"""
Traffic-Specific Augmentation Engine
流量特异性增强引擎

实现三种符合网络流量物理规律的数据增强策略：
1. 时序非对齐裁剪 (Asynchronous Temporal Cropping)
2. 时序抖动 (Inter-Arrival Jitter)
3. 特征通道掩码 (Channel Masking)
"""
import torch
import torch.nn as nn
import numpy as np


class TrafficAugmentation:
    """
    流量特异性增强引擎
    
    为同一条流生成两个不同的"视图"，用于对比学习
    """
    
    def __init__(self, config):
        """
        Args:
            config: 配置对象
        """
        self.config = config
        self.seq_len = config.SEQUENCE_LENGTH  # 1024
        
        # 增强策略的概率
        self.crop_prob = float(getattr(config, 'AUG_CROP_PROB', 0.8))
        self.jitter_prob = float(getattr(config, 'AUG_JITTER_PROB', 0.6))
        self.mask_prob = float(getattr(config, 'AUG_CHANNEL_MASK_PROB', 0.5))
        
        # 裁剪参数
        self.crop_min_ratio = float(getattr(config, 'AUG_CROP_MIN_RATIO', 0.5))
        self.crop_max_ratio = float(getattr(config, 'AUG_CROP_MAX_RATIO', 0.9))
        
        # 抖动参数（针对Log-IAT维度）
        self.jitter_std = float(getattr(config, 'AUG_JITTER_STD', 0.1))
        
        # 掩码参数
        self.mask_ratio = float(getattr(config, 'AUG_CHANNEL_MASK_RATIO', 0.15))
        
    def __call__(self, x):
        """
        生成两个增强视图
        
        Args:
            x: (B, L, D) - 原始流量序列
            
        Returns:
            x_view1: (B, L, D) - 第一个视图
            x_view2: (B, L, D) - 第二个视图
        """
        x_view1 = self._augment_single_view(x)
        x_view2 = self._augment_single_view(x)
        
        return x_view1, x_view2
    
    def _augment_single_view(self, x):
        """
        生成单个增强视图
        
        Args:
            x: (B, L, D) - 原始序列
            
        Returns:
            x_aug: (B, L, D) - 增强后的序列
        """
        x_aug = x.clone()
        
        # 策略A: 时序非对齐裁剪
        if np.random.rand() < self.crop_prob:
            x_aug = self._temporal_crop(x_aug)
        
        # 策略B: 时序抖动
        if np.random.rand() < self.jitter_prob:
            x_aug = self._temporal_jitter(x_aug)
        
        # 策略C: 特征通道掩码
        if np.random.rand() < self.mask_prob:
            x_aug = self._channel_mask(x_aug)
        
        return x_aug
    
    def _temporal_crop(self, x):
        """
        策略A: 时序非对齐裁剪
        
        物理意义：
        - 强迫模型识别"局部特征"与"整体行为"的一致性
        - 无论截取握手阶段还是传输阶段，核心指纹应保持不变
        
        Args:
            x: (B, L, D)
            
        Returns:
            x_crop: (B, L, D) - 裁剪后补零到原长度
        """
        B, L, D = x.shape
        
        # 随机决定裁剪长度
        crop_ratio = np.random.uniform(self.crop_min_ratio, self.crop_max_ratio)
        crop_len = int(L * crop_ratio)
        
        # 随机决定起始位置
        start_idx = np.random.randint(0, L - crop_len + 1)
        
        # 裁剪并补零
        x_crop = torch.zeros_like(x)
        x_crop[:, :crop_len, :] = x[:, start_idx:start_idx+crop_len, :]
        
        return x_crop
    
    def _temporal_jitter(self, x):
        """
        策略B: 时序抖动
        
        物理意义：
        - 模拟真实网络环境中的网络抖动（Jitter）
        - 模型必须学会忽略微小的传输延迟，抓住核心的"节奏感"
        
        Args:
            x: (B, L, D)
            
        Returns:
            x_jitter: (B, L, D) - 对Log-IAT添加噪声
        """
        x_jitter = x.clone()
        
        # 只对Log-IAT维度（索引1）添加高斯噪声（仅对有效token）
        noise = torch.randn_like(x[:, :, 1]) * self.jitter_std
        if x.shape[-1] >= 6:
            valid = x[:, :, 5] > 0.5
            x_jitter[:, :, 1] = torch.where(valid, x[:, :, 1] + noise, x[:, :, 1])
        else:
            x_jitter[:, :, 1] = x[:, :, 1] + noise
        
        return x_jitter
    
    def _channel_mask(self, x):
        """
        策略C: 特征通道掩码
        
        物理意义：
        - 强迫模型在缺失部分信息的情况下还原流的身份
        - 例如：看不到TCP标志位时，仅靠包长和时间序列识别流
        
        Args:
            x: (B, L, D)
            
        Returns:
            x_mask: (B, L, D) - 随机掩码某个维度
        """
        x_mask = x.clone()
        B, L, D = x.shape
        
        # 随机选择要掩码的维度（不掩码Direction/ValidMask，因为它们是结构性特征）
        # 6维输入: [0: Length, 3: BurstSize, 4: CumulativeLen]
        # 旧版输入(<=5维): 退化到原始策略
        if D >= 6:
            maskable_dims = [0, 3, 4]
            valid = x[:, :, 5] > 0.5
        else:
            maskable_dims = [0, 3, 4]
            valid = None
        
        for b in range(B):
            # 每个样本随机选择一个维度掩码
            mask_dim = np.random.choice(maskable_dims)
            if valid is not None:
                x_mask[b, valid[b], mask_dim] = 0.0
            else:
                x_mask[b, :, mask_dim] = 0.0

        return x_mask


class DualViewAugmentation:
    """
    双视图增强管道
    
    为对比学习生成两个不同的视图
    """
    
    def __init__(self, config):
        self.augmentation = TrafficAugmentation(config)
    
    def __call__(self, x):
        """
        Args:
            x: (B, L, D) - 原始流量
            
        Returns:
            x_view1: (B, L, D)
            x_view2: (B, L, D)
        """
        return self.augmentation(x)
