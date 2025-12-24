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
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)
        
        # 创建正样本mask
        # 对于z_i[k]，正样本是z_j[k]（在位置B+k）
        # 对于z_j[k]，正样本是z_i[k]（在位置k）
        pos_mask = torch.zeros(2*B, 2*B, device=z.device)
        for k in range(B):
            pos_mask[k, B+k] = 1  # z_i[k] <-> z_j[k]
            pos_mask[B+k, k] = 1  # z_j[k] <-> z_i[k]
        
        # 创建负样本mask（排除自己和正样本）
        neg_mask = torch.ones(2*B, 2*B, device=z.device)
        neg_mask.fill_diagonal_(0)  # 排除自己
        neg_mask = neg_mask - pos_mask  # 排除正样本
        
        # 计算InfoNCE损失
        # log(exp(sim_pos) / sum(exp(sim_neg)))
        pos_sim = (sim_matrix * pos_mask).sum(dim=1)  # (2B,)
        neg_sim = torch.exp(sim_matrix) * neg_mask  # (2B, 2B)
        neg_sim = neg_sim.sum(dim=1)  # (2B,)
        
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sim + 1e-8))
        loss = loss.mean()
        
        return loss


class ProjectionHead(nn.Module):
    """
    投影头 (Projection Head)
    
    作用：
    - 保护Mamba提取的特征保留原始信息
    - 让"强行拉近距离"的变换发生在MLP的输出端
    - 训练完后，扔掉这个MLP，只保留Mamba
    """
    
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=64):
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
        
        # 投影头
        self.projection_head = ProjectionHead(
            input_dim=config.MODEL_DIM,
            hidden_dim=128,
            output_dim=config.MODEL_DIM
        )
        
        # InfoNCE损失
        self.infonce_loss = InfoNCELoss(temperature=config.SUPCON_TEMPERATURE)
    
    def forward(self, x_view1, x_view2):
        """
        前向传播
        
        Args:
            x_view1: (B, L, 5) - 第一个视图
            x_view2: (B, L, 5) - 第二个视图
            
        Returns:
            loss: scalar - InfoNCE损失
            z_i: (B, d) - 第一个视图的特征（用于下游任务）
            z_j: (B, d) - 第二个视图的特征（用于下游任务）
        """
        # 通过共享的Mamba编码器
        z_i = self.backbone(x_view1, return_sequence=False)  # (B, 64)
        z_j = self.backbone(x_view2, return_sequence=False)  # (B, 64)
        
        # 通过投影头（用于对比学习）
        h_i = self.projection_head(z_i)  # (B, 64)
        h_j = self.projection_head(z_j)  # (B, 64)
        
        # 计算InfoNCE损失
        loss = self.infonce_loss(h_i, h_j)
        
        return loss, z_i, z_j


class HybridPretrainingLoss(nn.Module):
    """
    混合预训练损失
    
    L_Total = L_SimMTM + λ * L_InfoNCE
    
    - L_SimMTM: 理解序列语法（保证模型知道"这是一个合法的TCP流"）
    - L_InfoNCE: 区分个体差异（保证模型知道"这个TCP流和那个TCP流完全不同"）
    """
    
    def __init__(self, simmtm_loss, instance_contrastive, lambda_infonce=1.0):
        """
        Args:
            simmtm_loss: SimMTM损失模块
            instance_contrastive: 实例对比学习模块
            lambda_infonce: InfoNCE损失的权重
        """
        super().__init__()
        
        self.simmtm_loss = simmtm_loss
        self.instance_contrastive = instance_contrastive
        self.lambda_infonce = lambda_infonce
    
    def forward(self, backbone, x_original, x_view1, x_view2):
        """
        计算混合损失
        
        Args:
            backbone: Micro-Bi-Mamba backbone
            x_original: (B, L, 5) - 原始输入（用于SimMTM）
            x_view1: (B, L, 5) - 第一个增强视图（用于InfoNCE）
            x_view2: (B, L, 5) - 第二个增强视图（用于InfoNCE）
            
        Returns:
            total_loss: scalar
            loss_dict: 各项损失的字典
        """
        # SimMTM损失
        loss_simmtm = self.simmtm_loss(backbone, x_original)
        
        # InfoNCE损失
        loss_infonce, z_i, z_j = self.instance_contrastive(x_view1, x_view2)
        
        # 总损失
        total_loss = loss_simmtm + self.lambda_infonce * loss_infonce
        
        loss_dict = {
            'total': total_loss.item(),
            'simmtm': loss_simmtm.item(),
            'infonce': loss_infonce.item()
        }
        
        return total_loss, loss_dict
