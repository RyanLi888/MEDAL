"""
Time-Aware Instance Contrastive Learning
基于时序增强的Mamba实例对比学习

核心思想：
- 正样本：同一条流的不同物理扰动（延迟、丢包、截断）
- 负样本：任意两条不同的流（即使都是恶意的）
- 目标：在特征空间形成"群岛"效应，每个岛代表一种具体行为模式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for Instance Contrastive Learning
    
    数学直觉：
    - 在64维超球面上，将同一条流的两个分身像磁铁一样吸在一起
    - 将Batch内其他所有样本（2N-2个负样本）像同极磁铁一样排斥开
    """
    
    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: 温度系数τ
                - τ越小，对"相似"的定义越严苛
                - 特征空间里的簇（Cluster）会被压缩得越紧致
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        计算InfoNCE损失
        
        Args:
            z_i: (B, d) - 第一个视图的特征
            z_j: (B, d) - 第二个视图的特征
            
        Returns:
            loss: scalar
        """
        B = z_i.shape[0]

        # L2归一化到单位超球面
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 拼接所有特征：[z_i; z_j]
        z = torch.cat([z_i, z_j], dim=0)  # (2B, d)

        # 计算相似度矩阵 (logits)
        logits = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Mask self-similarity to avoid being selected as a candidate.
        # Use a large negative number instead of -inf for numerical safety.
        logits = logits.masked_fill(torch.eye(2 * B, device=z.device, dtype=torch.bool), -1e9)

        # Targets: for k in [0..B-1], positive index is B+k; for B+k, positive index is k
        targets = torch.arange(2 * B, device=z.device)
        targets = (targets + B) % (2 * B)

        # Standard cross entropy on (2B) classification problems
        loss = F.cross_entropy(logits, targets)
        return loss


class ProjectionHead(nn.Module):
    """
    投影头 (Projection Head)
    
    作用：
    - 保护Mamba提取的特征保留原始信息
    - 让"强行拉近距离"的变换发生在MLP的输出端
    - 训练完后，扔掉这个MLP，只保留Mamba
    """
    
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=32):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        """
        Args:
            z: (B, input_dim) - Mamba输出的特征
            
        Returns:
            h: (B, output_dim) - 投影后的特征（用于对比学习）
        """
        return self.mlp(z)




class InstanceContrastiveLearning(nn.Module):
    """
    实例对比学习模块
    
    整合：
    1. 双视图增强
    2. 共享Mamba编码器
    3. 投影头
    4. InfoNCE损失
    """
    
    def __init__(self, backbone, config):
        """
        Args:
            backbone: Micro-Bi-Mamba backbone
            config: 配置对象
        """
        super().__init__()
        
        self.backbone = backbone
        self.config = config

        # 只支持InfoNCE方法
        self.method = 'infonce'
        
        feature_dim = getattr(config, 'FEATURE_DIM', getattr(config, 'OUTPUT_DIM', config.MODEL_DIM))
        self.projection_head = ProjectionHead(input_dim=feature_dim, hidden_dim=128, output_dim=feature_dim)

        temperature = float(getattr(config, 'INFONCE_TEMPERATURE', getattr(config, 'SUPCON_TEMPERATURE', 0.1)))
        self.infonce_loss = InfoNCELoss(temperature=temperature)
    
    def forward(self, x_view1, x_view2, epoch: Optional[int] = None):
        """
        前向传播
        
        Args:
            x_view1: (B, L, D) - 第一个视图
            x_view2: (B, L, D) - 第二个视图
            
        Returns:
            loss: scalar - InfoNCE损失
            z_i: (B, d) - 第一个视图的特征（用于下游任务）
            z_j: (B, d) - 第二个视图的特征（用于下游任务）
        """
        try:
            seq_len = int(getattr(self.config, 'SEQUENCE_LENGTH', x_view1.shape[1]))
            effective_len = int(getattr(self.config, 'EFFECTIVE_SEQUENCE_LENGTH', seq_len))
            use_global = bool(getattr(self.config, 'USE_GLOBAL_STATS_TOKEN', False))
            has_global = bool(use_global and x_view1.dim() == 3 and x_view1.shape[1] == effective_len and effective_len > seq_len)
        except Exception:
            has_global = False
            seq_len = None

        if has_global and seq_len is not None:
            x_view1 = x_view1[:, :seq_len, :]
            x_view2 = x_view2[:, :seq_len, :]

        z_i = self.backbone(x_view1, return_sequence=False)
        z_j = self.backbone(x_view2, return_sequence=False)

        h_i = self.projection_head(z_i)
        h_j = self.projection_head(z_j)

        # 只使用InfoNCE损失
        loss = self.infonce_loss(h_i, h_j)
        return loss, z_i, z_j


class HybridPretrainingLoss(nn.Module):
    """
    MEDAL 混合预训练损失
    
    L_Total = L_SimMTM + λ_infonce * L_InfoNCE + λ_consistency * L_Consistency
    
    三项损失：
    1. L_SimMTM: 序列重构损失 - 理解流量语法
    2. L_InfoNCE: 实例对比损失 - 区分不同流的个体差异
    3. L_Consistency: 双流一致性损失 - 强迫内容流与结构流对齐（新增）
    
    设计哲学：
    - SimMTM: 保证模型知道"这是一个合法的TCP流"
    - InfoNCE: 保证模型知道"这个TCP流和那个TCP流完全不同"
    - Consistency: 保证模型知道"包长模式必须匹配时间节奏"（去噪核心）
    """

    def __init__(self, simmtm_loss, instance_contrastive, lambda_infonce=1.0):
        super().__init__()
        self.simmtm_loss = simmtm_loss
        self.instance_contrastive = instance_contrastive
        self.lambda_infonce = float(lambda_infonce)

    def forward(self, backbone, x_original, x_view1, x_view2, epoch: Optional[int] = None):
        loss_simmtm = self.simmtm_loss(backbone, x_original)
        loss_infonce, z_i, z_j = self.instance_contrastive(x_view1, x_view2, epoch=epoch)
        total_loss = loss_simmtm + self.lambda_infonce * loss_infonce

        loss_dict = {
            'total': total_loss.item(),
            'simmtm': loss_simmtm.item(),
            'infonce': loss_infonce.item(),
        }

        return total_loss, loss_dict
