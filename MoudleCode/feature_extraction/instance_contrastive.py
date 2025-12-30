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


class PredictionHead(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=16, output_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SimSiamLoss(nn.Module):
    def forward(self, p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()


class NNCLRLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, t, queue):
        q = F.normalize(q, dim=1)
        t = F.normalize(t, dim=1)
        queue = F.normalize(queue, dim=1)

        logits_pos = (q * t).sum(dim=1, keepdim=True)
        logits_neg = torch.matmul(q, queue.t())
        logits = torch.cat([logits_pos, logits_neg], dim=1) / self.temperature
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)


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

        self.method = getattr(config, 'CONTRASTIVE_METHOD', 'infonce')
        
        feature_dim = getattr(config, 'FEATURE_DIM', getattr(config, 'OUTPUT_DIM', config.MODEL_DIM))
        self.projection_head = ProjectionHead(input_dim=feature_dim, hidden_dim=128, output_dim=feature_dim)

        self.predictor = None
        self.simsiam_loss = None
        self.nnclr_loss = None
        self.register_buffer('_nnclr_queue', torch.zeros(int(getattr(config, 'NNCLR_QUEUE_SIZE', 4096)), feature_dim))
        self.register_buffer('_nnclr_ptr', torch.zeros(1, dtype=torch.long))

        temperature = float(getattr(config, 'INFONCE_TEMPERATURE', getattr(config, 'SUPCON_TEMPERATURE', 0.1)))
        self.infonce_loss = InfoNCELoss(temperature=temperature)

        if self.method == 'simsiam':
            self.predictor = PredictionHead(input_dim=feature_dim, hidden_dim=16, output_dim=feature_dim)
            self.simsiam_loss = SimSiamLoss()
        elif self.method == 'nnclr':
            self.nnclr_loss = NNCLRLoss(temperature=temperature)
    
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
        z_i = self.backbone(x_view1, return_sequence=False)
        z_j = self.backbone(x_view2, return_sequence=False)

        h_i = self.projection_head(z_i)
        h_j = self.projection_head(z_j)

        if self.method == 'simsiam':
            p_i = self.predictor(h_i)
            p_j = self.predictor(h_j)
            loss = 0.5 * self.simsiam_loss(p_i, h_j) + 0.5 * self.simsiam_loss(p_j, h_i)
            return loss, z_i, z_j

        if self.method == 'nnclr':
            queue = self._nnclr_queue
            min_sim = float(getattr(self.config, 'NNCLR_MIN_SIMILARITY', 0.0))
            warmup_epochs = int(getattr(self.config, 'NNCLR_WARMUP_EPOCHS', 0))
            in_warmup = (epoch is not None) and (warmup_epochs > 0) and (epoch < warmup_epochs)

            if queue.numel() == 0 or torch.all(queue == 0) or in_warmup:
                # Fallback: use own augmented view as target (SimCLR-style positive)
                nn_i = h_j.detach()
                nn_j = h_i.detach()
            else:
                q = F.normalize(queue, dim=1)
                hi_n = F.normalize(h_i, dim=1)
                hj_n = F.normalize(h_j, dim=1)

                sim_j = torch.matmul(hj_n, q.t())
                best_j, idx_j = torch.max(sim_j, dim=1)
                nn_j = queue[idx_j].detach()
                if min_sim > 0:
                    fallback_j = best_j < min_sim
                    if fallback_j.any():
                        nn_j[fallback_j] = h_i.detach()[fallback_j]

                sim_i = torch.matmul(hi_n, q.t())
                best_i, idx_i = torch.max(sim_i, dim=1)
                nn_i = queue[idx_i].detach()
                if min_sim > 0:
                    fallback_i = best_i < min_sim
                    if fallback_i.any():
                        nn_i[fallback_i] = h_j.detach()[fallback_i]

            loss_ij = self.nnclr_loss(h_i, nn_j, queue)
            loss_ji = self.nnclr_loss(h_j, nn_i, queue)
            loss = 0.5 * (loss_ij + loss_ji)

            with torch.no_grad():
                k = queue.size(0)
                b = h_i.size(0)
                ptr = int(self._nnclr_ptr.item())
                feats = torch.cat([F.normalize(h_i.detach(), dim=1), F.normalize(h_j.detach(), dim=1)], dim=0)
                n = feats.size(0)
                if n >= k:
                    queue.copy_(feats[-k:])
                    ptr = 0
                else:
                    end = ptr + n
                    if end <= k:
                        queue[ptr:end] = feats
                    else:
                        first = k - ptr
                        queue[ptr:k] = feats[:first]
                        queue[0:end - k] = feats[first:]
                    ptr = (end) % k
                self._nnclr_ptr[0] = ptr

            return loss, z_i, z_j

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
    
    def forward(self, backbone, x_original, x_view1, x_view2, epoch: Optional[int] = None):
        """
        计算混合损失
        
        Args:
            backbone: Micro-Bi-Mamba backbone
            x_original: (B, L, D) - 原始输入（用于SimMTM）
            x_view1: (B, L, D) - 第一个增强视图（用于InfoNCE）
            x_view2: (B, L, D) - 第二个增强视图（用于InfoNCE）
            
        Returns:
            total_loss: scalar
            loss_dict: 各项损失的字典
        """
        # SimMTM损失
        loss_simmtm = self.simmtm_loss(backbone, x_original)
        
        # InfoNCE损失
        loss_infonce, z_i, z_j = self.instance_contrastive(x_view1, x_view2, epoch=epoch)
        
        # 总损失
        total_loss = loss_simmtm + self.lambda_infonce * loss_infonce
        
        loss_dict = {
            'total': total_loss.item(),
            'simmtm': loss_simmtm.item(),
            'infonce': loss_infonce.item()
        }
        
        return total_loss, loss_dict
