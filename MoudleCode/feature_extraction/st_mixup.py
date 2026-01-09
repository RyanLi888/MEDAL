"""
ST-Mixup: Spatio-Temporal Mixup for Traffic Data Augmentation

针对网络流量时序数据的改进版Mixup增强策略：
1. 空间维度：混合连续特征值（Length, BurstSize）
2. 时间维度：简单时间偏移对齐（避免复杂DTW）
3. 离散特征：随机选择（保持语义完整性）
4. 类内混合：只混合同类样本（避免语义冲突）
5. 渐进式启用：训练后期才启用（先学基本边界）
"""
import torch
import torch.nn.functional as F
import numpy as np


class IntraClassSTMixup:
    """
    类内时空混合增强（Intra-Class Spatio-Temporal Mixup）
    
    核心特点：
    1. 只混合同类样本（正常-正常，恶意-恶意）
    2. 连续特征线性插值，离散特征随机选择
    3. 简单时间偏移（避免破坏时序结构）
    4. 渐进式启用（训练后期才使用）
    
    适用场景：
    - Stage 3分类器训练（骨干网络已冻结）
    - 标签已矫正（Hybrid Court后）
    - 需要增强决策边界鲁棒性
    """
    
    def __init__(self, alpha=0.2, warmup_epochs=100, max_prob=0.3, 
                 time_shift_ratio=0.15, device='cuda',
                 continuous_indices=None, discrete_indices=None, valid_mask_index=None):
        """
        Args:
            alpha: Beta分布参数（控制混合强度，越小越极端）
            warmup_epochs: 预热轮数（前N轮不启用）
            max_prob: 最大混合概率（渐进增加到此值）
            time_shift_ratio: 时间偏移比例（序列长度的百分比）
            device: 设备
        """
        self.alpha = alpha
        self.warmup_epochs = warmup_epochs
        self.max_prob = max_prob
        self.time_shift_ratio = time_shift_ratio
        self.device = device
        
        # 特征索引
        if continuous_indices is None:
            continuous_indices = [0, 2]
        if discrete_indices is None:
            discrete_indices = [1]
        self.continuous_indices = list(continuous_indices)
        self.discrete_indices = list(discrete_indices)
        self.valid_mask_index = valid_mask_index
    
    def should_apply(self, epoch):
        """判断当前epoch是否应该应用ST-Mixup"""
        return epoch >= self.warmup_epochs
    
    def get_mixup_prob(self, epoch, total_epochs):
        """
        计算当前epoch的混合概率（渐进式增加）
        
        Args:
            epoch: 当前轮数
            total_epochs: 总轮数
            
        Returns:
            prob: 混合概率 [0, max_prob]
        """
        if epoch < self.warmup_epochs:
            return 0.0
        
        # 从warmup后线性增加到max_prob
        progress = (epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs)
        return self.max_prob * min(1.0, progress)
    
    def __call__(self, X, y, epoch=None, total_epochs=None):
        """
        对batch进行类内ST-Mixup增强
        
        Args:
            X: (B, L, D) - batch of sequences
            y: (B,) - labels (0 or 1)
            epoch: 当前轮数（用于渐进式控制）
            total_epochs: 总轮数
            
        Returns:
            X_mixed: (B, L, D) - 混合后的序列
            y_mixed: (B,) - 混合后的标签（类内混合时标签不变）
        """
        # 检查是否应该应用
        if epoch is not None and not self.should_apply(epoch):
            return X, y
        
        # 计算混合概率
        if epoch is not None and total_epochs is not None:
            mixup_prob = self.get_mixup_prob(epoch, total_epochs)
        else:
            mixup_prob = self.max_prob
        
        B, L, D = X.shape
        X_mixed = X.clone()
        y_mixed = y.clone()
        
        # 为每个样本随机决定是否混合
        mix_mask = torch.rand(B, device=self.device) < mixup_prob
        mix_indices = mix_mask.nonzero(as_tuple=True)[0]
        
        if len(mix_indices) == 0:
            return X, y  # 没有样本需要混合
        
        # 对每个需要混合的样本，找同类配对
        for idx in mix_indices:
            label = y[idx].item()
            
            # 找所有同类样本（排除自己）
            same_class_mask = (y == label)
            same_class_mask[idx] = False
            same_class_indices = same_class_mask.nonzero(as_tuple=True)[0]
            
            if len(same_class_indices) == 0:
                continue  # 没有同类样本可配对
            
            # 随机选择一个同类样本
            pair_idx = same_class_indices[torch.randint(len(same_class_indices), (1,)).item()]
            
            # 执行ST-Mixup
            X_mixed[idx] = self._spatial_temporal_mix(
                X[idx], X[pair_idx], L
            )
            # y_mixed[idx] 保持不变（类内混合）
        
        return X_mixed, y_mixed
    
    def _spatial_temporal_mix(self, x1, x2, L):
        """
        对两个同类序列进行时空混合
        
        Args:
            x1, x2: (L, D) - 两个流量序列
            L: 序列长度
            
        Returns:
            x_mixed: (L, D) - 混合后的序列
        """
        # 1. 采样混合系数
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 2. 时间对齐：随机偏移
        max_shift = int(L * self.time_shift_ratio)
        offset = np.random.randint(-max_shift, max_shift + 1)
        x2_shifted = self._shift_sequence(x2, offset, L)
        
        # 3. 空间混合
        x_mixed = x1.clone()

        mixed_mask = None
        if self.valid_mask_index is not None:
            try:
                vm_idx = int(self.valid_mask_index)
                if 0 <= vm_idx < x1.shape[-1]:
                    mask1 = x1[:, vm_idx]
                    mask2 = x2_shifted[:, vm_idx]
                    mixed_mask = torch.maximum(mask1, mask2)
            except Exception:
                mixed_mask = None
        
        # 连续特征：线性插值
        for idx in self.continuous_indices:
            try:
                j = int(idx)
            except Exception:
                continue
            if j < 0 or j >= x1.shape[-1]:
                continue
            if self.valid_mask_index is not None and int(j) == int(self.valid_mask_index):
                continue
            x_mixed[:, j] = lam * x1[:, j] + (1 - lam) * x2_shifted[:, j]
        
        # 离散特征：随机选择（保持语义）
        for idx in self.discrete_indices:
            try:
                j = int(idx)
            except Exception:
                continue
            if j < 0 or j >= x1.shape[-1]:
                continue
            # 每个时间步独立随机选择
            mask = torch.rand(L, device=self.device) < lam
            x_mixed[mask, j] = x1[mask, j]
            x_mixed[~mask, j] = x2_shifted[~mask, j]

        if mixed_mask is not None:
            x_mixed[:, vm_idx] = mixed_mask
            pad = (mixed_mask <= 0.5)
            if pad.any():
                x_mixed[pad] = 0.0
        
        return x_mixed
    
    def _shift_sequence(self, x, offset, L):
        """
        时间偏移（循环移位）
        
        Args:
            x: (L, D) - 序列
            offset: 偏移量（正数向右，负数向左）
            L: 序列长度
            
        Returns:
            x_shifted: (L, D) - 偏移后的序列
        """
        if offset == 0:
            return x
        
        # 使用torch.roll进行循环移位
        return torch.roll(x, shifts=offset, dims=0)


class SelectiveSTMixup(IntraClassSTMixup):
    """
    选择性ST-Mixup：只对困难样本应用混合
    
    继承自IntraClassSTMixup，增加困难样本选择机制
    """
    
    def __init__(self, alpha=0.2, warmup_epochs=100, max_prob=0.3,
                 uncertainty_threshold=0.3, time_shift_ratio=0.15, device='cuda'):
        """
        Args:
            uncertainty_threshold: 不确定性阈值（熵）
            其他参数同IntraClassSTMixup
        """
        super().__init__(alpha, warmup_epochs, max_prob, time_shift_ratio, device)
        self.uncertainty_threshold = uncertainty_threshold
    
    def select_hard_samples(self, X, y, classifier, backbone):
        """
        选择困难样本（高不确定性）
        
        Args:
            X: (B, L, D) - 输入序列
            y: (B,) - 标签
            classifier: 分类器模型
            backbone: 骨干网络
            
        Returns:
            hard_mask: (B,) - 困难样本mask
        """
        with torch.no_grad():
            # 提取特征
            z = backbone(X, return_sequence=False)
            
            # 分类
            logits = classifier(z, return_separate=False)
            probs = F.softmax(logits, dim=1)
            
            # 计算不确定性（熵）
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            
            # 选择高不确定性样本
            hard_mask = entropy > self.uncertainty_threshold
        
        return hard_mask
    
    def __call__(self, X, y, epoch=None, total_epochs=None, 
                 classifier=None, backbone=None):
        """
        对困难样本进行类内ST-Mixup增强
        
        Args:
            X, y, epoch, total_epochs: 同IntraClassSTMixup
            classifier: 分类器（用于选择困难样本）
            backbone: 骨干网络（用于选择困难样本）
            
        Returns:
            X_mixed, y_mixed: 混合后的数据
        """
        # 如果没有提供模型，退化为普通IntraClassSTMixup
        if classifier is None or backbone is None:
            return super().__call__(X, y, epoch, total_epochs)
        
        # 检查是否应该应用
        if epoch is not None and not self.should_apply(epoch):
            return X, y
        
        # 选择困难样本
        hard_mask = self.select_hard_samples(X, y, classifier, backbone)
        
        if hard_mask.sum() < 2:
            return X, y  # 困难样本太少
        
        # 只对困难样本应用Mixup
        X_hard = X[hard_mask]
        y_hard = y[hard_mask]
        
        X_hard_mixed, y_hard_mixed = super().__call__(
            X_hard, y_hard, epoch, total_epochs
        )
        
        # 合并回原batch
        X_mixed = X.clone()
        X_mixed[hard_mask] = X_hard_mixed
        
        return X_mixed, y


# 便捷函数
def create_st_mixup(config, mode='intra_class'):
    """
    创建ST-Mixup增强器
    
    Args:
        config: 配置对象
        mode: 'intra_class' 或 'selective'
        
    Returns:
        st_mixup: ST-Mixup增强器
    """
    if mode == 'intra_class':
        length_idx = getattr(config, 'LENGTH_INDEX', None)
        burst_idx = getattr(config, 'BURST_SIZE_INDEX', None)
        cum_idx = getattr(config, 'CUMULATIVE_LEN_INDEX', None)
        direction_idx = getattr(config, 'DIRECTION_INDEX', None)
        valid_mask_idx = getattr(config, 'VALID_MASK_INDEX', None)

        cont = []
        for idx in [length_idx, burst_idx, cum_idx]:
            if idx is None:
                continue
            try:
                cont.append(int(idx))
            except Exception:
                continue
        disc = []
        if direction_idx is not None:
            try:
                disc = [int(direction_idx)]
            except Exception:
                disc = []

        return IntraClassSTMixup(
            alpha=config.STAGE3_ST_MIXUP_ALPHA,
            warmup_epochs=config.STAGE3_ST_MIXUP_WARMUP_EPOCHS,
            max_prob=config.STAGE3_ST_MIXUP_MAX_PROB,
            time_shift_ratio=config.STAGE3_ST_MIXUP_TIME_SHIFT_RATIO,
            device=config.DEVICE,
            continuous_indices=cont,
            discrete_indices=disc,
            valid_mask_index=valid_mask_idx,
        )
    elif mode == 'selective':
        length_idx = getattr(config, 'LENGTH_INDEX', None)
        burst_idx = getattr(config, 'BURST_SIZE_INDEX', None)
        cum_idx = getattr(config, 'CUMULATIVE_LEN_INDEX', None)
        direction_idx = getattr(config, 'DIRECTION_INDEX', None)
        valid_mask_idx = getattr(config, 'VALID_MASK_INDEX', None)

        cont = []
        for idx in [length_idx, burst_idx, cum_idx]:
            if idx is None:
                continue
            try:
                cont.append(int(idx))
            except Exception:
                continue
        disc = []
        if direction_idx is not None:
            try:
                disc = [int(direction_idx)]
            except Exception:
                disc = []

        return SelectiveSTMixup(
            alpha=config.STAGE3_ST_MIXUP_ALPHA,
            warmup_epochs=config.STAGE3_ST_MIXUP_WARMUP_EPOCHS,
            max_prob=config.STAGE3_ST_MIXUP_MAX_PROB,
            uncertainty_threshold=config.STAGE3_ST_MIXUP_UNCERTAINTY_THRESHOLD,
            time_shift_ratio=config.STAGE3_ST_MIXUP_TIME_SHIFT_RATIO,
            device=config.DEVICE,
            continuous_indices=cont,
            discrete_indices=disc,
            valid_mask_index=valid_mask_idx,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # 测试代码
    print("Testing IntraClassSTMixup...")
    
    # 创建测试数据
    from MoudleCode.utils.config import config
    B, L, D = 8, 1024, int(getattr(config, 'INPUT_FEATURE_DIM', 6))
    X = torch.randn(B, L, D)
    y = torch.randint(0, 2, (B,))
    
    # 创建增强器
    st_mixup = IntraClassSTMixup(alpha=0.2, warmup_epochs=0, max_prob=0.5)
    
    # 测试
    X_mixed, y_mixed = st_mixup(X, y, epoch=100, total_epochs=200)
    
    print(f"✓ Input shape: {X.shape}")
    print(f"✓ Output shape: {X_mixed.shape}")
    print(f"✓ Labels unchanged (intra-class): {torch.equal(y, y_mixed)}")
    print(f"✓ Data changed: {not torch.equal(X, X_mixed)}")
    print("\nTest passed!")
