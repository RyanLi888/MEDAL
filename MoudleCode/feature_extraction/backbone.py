"""
Micro-Bi-Mamba Backbone for Feature Extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .mamba_block import MambaLayer


class TrafficHybridEmbedding(nn.Module):
    def __init__(self, d_model: int, direction_index: int = 2, input_dim: int = 6):
        super().__init__()
        self.direction_index = int(direction_index)
        self.input_dim = int(input_dim)

        cont_indices = [i for i in range(self.input_dim) if i != self.direction_index]
        self.register_buffer('_cont_indices', torch.tensor(cont_indices, dtype=torch.long), persistent=False)

        self.cont_proj = nn.Sequential(
            nn.Linear(len(cont_indices), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.dir_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)
        nn.init.normal_(self.dir_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cont = x.index_select(dim=-1, index=self._cont_indices)

        x_dir_raw = x[:, :, self.direction_index]
        dir_indices = (x_dir_raw > 0).to(dtype=torch.long)

        embed_cont = self.cont_proj(x_cont)
        embed_dir = self.dir_emb(dir_indices)

        return embed_cont + embed_dir


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        
        Returns:
            x + pe: (B, L, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class MicroBiMambaBackbone(nn.Module):
    """
    Micro-Bi-Mamba Backbone
    
    Bidirectional Mamba architecture for traffic feature extraction
    with frozen backbone after pre-training
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.output_dim = getattr(config, 'OUTPUT_DIM', getattr(config, 'FEATURE_DIM', config.MODEL_DIM))
        
        # Embedding layer
        input_dim = int(getattr(config, 'INPUT_FEATURE_DIM', 0))
        direction_index = getattr(config, 'DIRECTION_INDEX', None)
        if direction_index is not None:
            try:
                direction_index = int(direction_index)
            except Exception:
                direction_index = None

        if input_dim > 0 and direction_index is not None and 0 <= int(direction_index) < input_dim:
            self.embedding = TrafficHybridEmbedding(
                d_model=config.MODEL_DIM,
                direction_index=int(direction_index),
                input_dim=input_dim,
            )
        else:
            self.embedding = nn.Linear(config.INPUT_FEATURE_DIM, config.MODEL_DIM)
        # 使用 EFFECTIVE_SEQUENCE_LENGTH 以支持全局统计令牌（第1025行）
        max_seq_len = getattr(config, 'EFFECTIVE_SEQUENCE_LENGTH', config.SEQUENCE_LENGTH)
        self.pos_encoding = SinusoidalPositionalEncoding(config.MODEL_DIM, max_len=max_seq_len)
        self.embed_dropout = nn.Dropout(config.EMBEDDING_DROPOUT)
        
        # Forward Mamba layers
        self.forward_layers = nn.ModuleList([
            MambaLayer(
                d_model=config.MODEL_DIM,
                d_state=config.MAMBA_STATE_DIM,
                d_conv=config.MAMBA_CONV_KERNEL,
                expand=config.MAMBA_EXPANSION_FACTOR,
                dropout=config.MAMBA_DROPOUT
            )
            for _ in range(config.MAMBA_LAYERS)
        ])
        
        # Backward Mamba layers
        self.backward_layers = nn.ModuleList([
            MambaLayer(
                d_model=config.MODEL_DIM,
                d_state=config.MAMBA_STATE_DIM,
                d_conv=config.MAMBA_CONV_KERNEL,
                expand=config.MAMBA_EXPANSION_FACTOR,
                dropout=config.MAMBA_DROPOUT
            )
            for _ in range(config.MAMBA_LAYERS)
        ])
        
        # Projection layer to combine bidirectional features
        # Keep internal MODEL_DIM for sequence modeling, but output compressed features for downstream.
        self.projection = nn.Linear(config.MODEL_DIM * 2, self.output_dim)
        
        # Decoder for SimMTM (pre-training only)
        use_mlp_decoder = bool(getattr(config, 'SIMMTM_DECODER_USE_MLP', False))
        decoder_hidden_dim = int(getattr(config, 'SIMMTM_DECODER_HIDDEN_DIM', 64))
        if use_mlp_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(self.output_dim, decoder_hidden_dim),
                nn.GELU(),
                nn.Linear(decoder_hidden_dim, config.INPUT_FEATURE_DIM),
            )
        else:
            self.decoder = nn.Linear(self.output_dim, config.INPUT_FEATURE_DIM)
        
        self.frozen = False
    
    def forward(self, x, return_sequence=False):
        """
        Forward pass
        
        Args:
            x: (B, L, D_in) - input features
            return_sequence: If True, return sequence features; else return pooled features
            
        Returns:
            z: (B, output_dim) if return_sequence=False, else (B, L, output_dim)
        """
        valid_mask = None
        valid_mask_index = getattr(self.config, 'VALID_MASK_INDEX', None)
        if valid_mask_index is not None:
            try:
                if int(valid_mask_index) >= 0 and int(valid_mask_index) < x.shape[-1]:
                    valid_mask = x[:, :, int(valid_mask_index)]
            except Exception:
                valid_mask = None

        # Embedding
        x = self.embedding(x)  # (B, L, d_model)
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)
        
        # Forward pass
        x_fwd = x
        for layer in self.forward_layers:
            x_fwd = layer(x_fwd)
        
        # Backward pass (reverse sequence)
        x_bwd = torch.flip(x, dims=[1])
        for layer in self.backward_layers:
            x_bwd = layer(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])  # Reverse back
        
        # Combine bidirectional features
        if return_sequence:
            # Concatenate and project for sequence output
            x_combined = torch.cat([x_fwd, x_bwd], dim=-1)  # (B, L, 2*d_model)
            z = self.projection(x_combined)  # (B, L, d_model)
        else:
            # Average pooling and combine for single vector output
            if valid_mask is not None:
                m = (valid_mask > 0.5).to(dtype=x_fwd.dtype, device=x_fwd.device).unsqueeze(-1)  # (B, L, 1)
                denom = m.sum(dim=1).clamp_min(1e-6)
                z_fwd = (x_fwd * m).sum(dim=1) / denom
                z_bwd = (x_bwd * m).sum(dim=1) / denom
            else:
                z_fwd = torch.mean(x_fwd, dim=1)  # (B, d_model)
                z_bwd = torch.mean(x_bwd, dim=1)  # (B, d_model)
            z_cat = torch.cat([z_fwd, z_bwd], dim=-1)  # (B, 2*d_model)
            z = self.projection(z_cat)  # (B, d_model)
        
        return z
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
        self.frozen = True
    
    def unfreeze(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        self.frozen = False


class _DualStreamHybridEmbedding(nn.Module):
    """旧版：仅支持 Direction + BurstSize"""
    def __init__(self, d_model: int):
        super().__init__()
        self.cont_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.dir_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)
        nn.init.normal_(self.dir_emb.weight, std=0.02)

    def forward(self, x_dir: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        dir_indices = (x_dir > 0).to(dtype=torch.long)
        embed_cont = self.cont_proj(x_cont)
        embed_dir = self.dir_emb(dir_indices)
        return embed_cont + embed_dir


class _DualStreamStructureEmbedding(nn.Module):
    """
    MEDAL-Lite5 结构流 Embedding
    
    输入：Direction (离散) + BurstSize (连续)
    输出：融合后的结构特征向量
    
    设计：
    - Direction: Embedding 层（+1/-1 映射到向量）
    - BurstSize: 线性投影（1维 → d_model）
    - 融合：加法（保持特征独立性）
    """
    def __init__(self, d_model: int, cont_dim: int = 1):
        super().__init__()
        self.cont_proj = nn.Sequential(
            nn.Linear(cont_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.dir_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)
        nn.init.normal_(self.dir_emb.weight, std=0.02)

    def forward(self, x_dir: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_dir: (B, L) - Direction 特征 (+1/-1)
            x_cont: (B, L, cont_dim) - 连续特征 [BurstSize]
        
        Returns:
            embed: (B, L, d_model) - 融合后的结构特征
        """
        dir_indices = (x_dir > 0).to(dtype=torch.long)
        embed_cont = self.cont_proj(x_cont)
        embed_dir = self.dir_emb(dir_indices)
        return embed_cont + embed_dir


class DualStreamBiMambaBackbone(nn.Module):
    """
    MEDAL-Lite5 双流 Bi-Mamba 骨干网络
    
    双流物理切分：
    - Stream 1 (内容+时序流): Length + LogIAT - 学习"大包与小包的排列旋律" + "时间节奏"
    - Stream 2 (结构流): Direction + BurstSize - 学习"交互模式的配合"
    
    门控融合：动态调整两路特征权重
    InfoNCE 一致性：强迫双流特征对齐（通过 instance_contrastive 模块）
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = getattr(config, 'OUTPUT_DIM', getattr(config, 'FEATURE_DIM', config.MODEL_DIM))

        model_dim = int(getattr(config, 'MODEL_DIM', 32))
        d_len = max(1, model_dim // 2)
        d_struct = max(1, model_dim - d_len)

        self._d_len = int(d_len)
        self._d_struct = int(d_struct)

        self.fusion_type = str(getattr(config, 'MAMBA_FUSION_TYPE', 'gate')).strip().lower()

        # MEDAL-Lite5 特征索引
        self.length_index = getattr(config, 'LENGTH_INDEX', 0)
        self.log_iat_index = getattr(config, 'LOG_IAT_INDEX', getattr(config, 'IAT_INDEX', None))
        self.direction_index = getattr(config, 'DIRECTION_INDEX', None)
        self.burst_index = getattr(config, 'BURST_SIZE_INDEX', None)
        self.valid_mask_index = getattr(config, 'VALID_MASK_INDEX', None)

        # Stream 1: 内容+时序流 (Length + LogIAT)
        # 使用2维输入：Length (1维) + LogIAT (1维) = 2维
        self.len_embed = nn.Sequential(
            nn.Linear(2, self._d_len),  # 从1维改为2维：Length + LogIAT
            nn.LayerNorm(self._d_len),
            nn.GELU(),
        )
        # 使用 EFFECTIVE_SEQUENCE_LENGTH 以支持全局统计令牌（第1025行）
        max_seq_len = getattr(config, 'EFFECTIVE_SEQUENCE_LENGTH', config.SEQUENCE_LENGTH)
        self.len_pos = SinusoidalPositionalEncoding(self._d_len, max_len=max_seq_len)
        self.len_dropout = nn.Dropout(config.EMBEDDING_DROPOUT)

        # Stream 2: 结构流 (Direction + BurstSize)
        self.struct_embed = _DualStreamStructureEmbedding(self._d_struct, cont_dim=1)
        self.struct_pos = SinusoidalPositionalEncoding(self._d_struct, max_len=max_seq_len)
        self.struct_dropout = nn.Dropout(config.EMBEDDING_DROPOUT)

        self.len_forward_layers = nn.ModuleList([
            MambaLayer(
                d_model=self._d_len,
                d_state=config.MAMBA_STATE_DIM,
                d_conv=config.MAMBA_CONV_KERNEL,
                expand=config.MAMBA_EXPANSION_FACTOR,
                dropout=config.MAMBA_DROPOUT,
            )
            for _ in range(config.MAMBA_LAYERS)
        ])
        self.len_backward_layers = nn.ModuleList([
            MambaLayer(
                d_model=self._d_len,
                d_state=config.MAMBA_STATE_DIM,
                d_conv=config.MAMBA_CONV_KERNEL,
                expand=config.MAMBA_EXPANSION_FACTOR,
                dropout=config.MAMBA_DROPOUT,
            )
            for _ in range(config.MAMBA_LAYERS)
        ])
        self.len_proj = nn.Linear(self._d_len * 2, self._d_len)

        self.struct_forward_layers = nn.ModuleList([
            MambaLayer(
                d_model=self._d_struct,
                d_state=config.MAMBA_STATE_DIM,
                d_conv=config.MAMBA_CONV_KERNEL,
                expand=config.MAMBA_EXPANSION_FACTOR,
                dropout=config.MAMBA_DROPOUT,
            )
            for _ in range(config.MAMBA_LAYERS)
        ])
        self.struct_backward_layers = nn.ModuleList([
            MambaLayer(
                d_model=self._d_struct,
                d_state=config.MAMBA_STATE_DIM,
                d_conv=config.MAMBA_CONV_KERNEL,
                expand=config.MAMBA_EXPANSION_FACTOR,
                dropout=config.MAMBA_DROPOUT,
            )
            for _ in range(config.MAMBA_LAYERS)
        ])
        self.struct_proj = nn.Linear(self._d_struct * 2, self._d_struct)

        self.fusion_gate = None
        self.fusion_proj = None
        self.len_to_fused = None
        self.struct_to_fused = None

        if self.fusion_type in ('gate', 'gated', 'gated_fusion'):
            self.fusion_gate = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.Sigmoid(),
            )
            self.fusion_proj = nn.Linear(model_dim, model_dim)
        elif self.fusion_type in ('concat_project', 'concat', 'project'):
            self.fusion_proj = nn.Linear(model_dim, model_dim)
        elif self.fusion_type in ('average', 'avg', 'mean'):
            self.len_to_fused = nn.Linear(self._d_len, model_dim)
            self.struct_to_fused = nn.Linear(self._d_struct, model_dim)
        else:
            self.fusion_type = 'gate'
            self.fusion_gate = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.Sigmoid(),
            )
            self.fusion_proj = nn.Linear(model_dim, model_dim)
        self.projection = nn.Linear(model_dim, self.output_dim)

        use_mlp_decoder = bool(getattr(config, 'SIMMTM_DECODER_USE_MLP', False))
        decoder_hidden_dim = int(getattr(config, 'SIMMTM_DECODER_HIDDEN_DIM', 64))
        if use_mlp_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(self.output_dim, decoder_hidden_dim),
                nn.GELU(),
                nn.Linear(decoder_hidden_dim, config.INPUT_FEATURE_DIM),
            )
        else:
            self.decoder = nn.Linear(self.output_dim, config.INPUT_FEATURE_DIM)

        self.frozen = False

    def forward(self, x, return_sequence=False, return_dual_features=False, return_pre_fusion=False):
        """
        前向传播
        
        Args:
            x: (B, L, D) - 输入特征
            return_sequence: 是否返回序列特征
            return_dual_features: 是否返回双流特征（用于 InfoNCE 一致性，池化后）
            return_pre_fusion: 是否返回融合前的序列特征（用于 InfoNCE 一致性损失计算）
        
        Returns:
            如果 return_pre_fusion=True:
                h_len: (B, L, d_len) - 内容流序列特征（融合前）
                h_struct: (B, L, d_struct) - 结构流序列特征（融合前）
            如果 return_dual_features=True:
                z_content: (B, d_len) - 内容流特征（池化后）
                z_structure: (B, d_struct) - 结构流特征（池化后）
                z_fused: (B, output_dim) - 融合后特征
            否则:
                z: (B, output_dim) 或 (B, L, output_dim)
        """
        # 输入维度验证（MEDAL-Lite5应该是5维特征）
        expected_dim = getattr(self.config, 'INPUT_FEATURE_DIM', 5)
        if x.shape[-1] != expected_dim:
            import warnings
            warnings.warn(
                f"Input feature dimension mismatch: expected {expected_dim} (MEDAL-Lite5), "
                f"got {x.shape[-1]}. This may cause feature extraction errors.",
                UserWarning
            )
        
        valid_mask = None
        if self.valid_mask_index is not None:
            try:
                if int(self.valid_mask_index) >= 0 and int(self.valid_mask_index) < x.shape[-1]:
                    valid_mask = x[:, :, int(self.valid_mask_index)]
            except Exception:
                valid_mask = None

        gate = None
        if valid_mask is not None:
            gate = (valid_mask > 0.5).to(dtype=x.dtype, device=x.device).unsqueeze(-1)

        # ===== Stream 1: 内容+时序流 (Length + LogIAT) =====
        length_index = int(self.length_index) if self.length_index is not None else 0
        log_iat_index = int(self.log_iat_index) if self.log_iat_index is not None else None
        
        # 提取Length特征
        x_length = x[:, :, length_index:length_index + 1]  # (B, L, 1)
        
        # 提取LogIAT特征（如果存在）
        if log_iat_index is not None and 0 <= log_iat_index < x.shape[-1]:
            x_logiat = x[:, :, log_iat_index:log_iat_index + 1]  # (B, L, 1)
        else:
            # 如果LogIAT不存在，使用零填充
            x_logiat = torch.zeros_like(x_length)
        
        # 拼接Length和LogIAT: (B, L, 2)
        x_len = torch.cat([x_length, x_logiat], dim=-1)
        x_len = self.len_embed(x_len)
        x_len = self.len_pos(x_len)
        x_len = self.len_dropout(x_len)
        if gate is not None:
            x_len = x_len * gate

        # ===== Stream 2: 结构流 (Direction + BurstSize) =====
        x_dir = None
        x_burst = None
        
        # Extract Direction
        if self.direction_index is not None:
            try:
                di = int(self.direction_index)
                if 0 <= di < x.shape[-1]:
                    x_dir = x[:, :, di]
            except Exception:
                x_dir = None
        
        # Extract BurstSize
        if self.burst_index is not None:
            try:
                bi = int(self.burst_index)
                if 0 <= bi < x.shape[-1]:
                    x_burst = x[:, :, bi:bi + 1]
            except Exception:
                x_burst = None
        
        # Fallback to zeros if features are missing
        if x_dir is None:
            x_dir = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        if x_burst is None:
            x_burst = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)

        x_struct_cont = x_burst  # (B, L, 1)
        
        x_struct = self.struct_embed(x_dir, x_struct_cont)
        x_struct = self.struct_pos(x_struct)
        x_struct = self.struct_dropout(x_struct)
        if gate is not None:
            x_struct = x_struct * gate

        # ===== Bi-Mamba 编码 =====
        # Content stream
        x_len_fwd = x_len
        for layer in self.len_forward_layers:
            x_len_fwd = layer(x_len_fwd)
        x_len_bwd = torch.flip(x_len, dims=[1])
        for layer in self.len_backward_layers:
            x_len_bwd = layer(x_len_bwd)
        x_len_bwd = torch.flip(x_len_bwd, dims=[1])
        h_len = self.len_proj(torch.cat([x_len_fwd, x_len_bwd], dim=-1))

        # Structure stream
        x_struct_fwd = x_struct
        for layer in self.struct_forward_layers:
            x_struct_fwd = layer(x_struct_fwd)
        x_struct_bwd = torch.flip(x_struct, dims=[1])
        for layer in self.struct_backward_layers:
            x_struct_bwd = layer(x_struct_bwd)
        x_struct_bwd = torch.flip(x_struct_bwd, dims=[1])
        h_struct = self.struct_proj(torch.cat([x_struct_fwd, x_struct_bwd], dim=-1))

        # ===== 返回融合前的序列特征（用于 InfoNCE 一致性损失） =====
        if return_pre_fusion:
            return h_len, h_struct

        h = torch.cat([h_len, h_struct], dim=-1)  # (B, L, model_dim)
        if self.fusion_type == 'gate':
            fusion_gate = self.fusion_gate(h)  # (B, L, model_dim)
            h_fused = fusion_gate * h + (1.0 - fusion_gate) * self.fusion_proj(h)
        elif self.fusion_type == 'concat_project':
            h_fused = self.fusion_proj(h)
        elif self.fusion_type == 'average':
            h_fused = 0.5 * (self.len_to_fused(h_len) + self.struct_to_fused(h_struct))
        else:
            fusion_gate = self.fusion_gate(h)  # (B, L, model_dim)
            h_fused = fusion_gate * h + (1.0 - fusion_gate) * self.fusion_proj(h)

        # ===== 返回双流特征（用于 InfoNCE 一致性） =====
        if return_dual_features:
            # Pool each stream separately
            if valid_mask is not None:
                m = (valid_mask > 0.5).to(dtype=h.dtype, device=h.device).unsqueeze(-1)
                denom = m.sum(dim=1).clamp_min(1e-6)
                z_content = (h_len * m).sum(dim=1) / denom  # (B, d_len)
                z_structure = (h_struct * m).sum(dim=1) / denom  # (B, d_struct)
                z_fused_pooled = (h_fused * m).sum(dim=1) / denom
            else:
                z_content = torch.mean(h_len, dim=1)
                z_structure = torch.mean(h_struct, dim=1)
                z_fused_pooled = torch.mean(h_fused, dim=1)
            
            z_fused = self.projection(z_fused_pooled)
            return z_content, z_structure, z_fused

        # ===== 标准返回 =====
        if return_sequence:
            return self.projection(h_fused)

        if valid_mask is not None:
            m = (valid_mask > 0.5).to(dtype=h_fused.dtype, device=h_fused.device).unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1e-6)
            pooled = (h_fused * m).sum(dim=1) / denom
        else:
            pooled = torch.mean(h_fused, dim=1)
        return self.projection(pooled)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.frozen = False


def build_backbone(config, logger=None):
    if logger is not None:
        logger.info("🧠 Backbone arch: DualStreamBiMambaBackbone (fixed)")
    return DualStreamBiMambaBackbone(config)


class SimMTMLoss(nn.Module):
    """
    Masked multi-head reconstruction loss (SimMTM++)
    
    - 随机掩码 + 轻噪声
    - 连续特征用 MSE/Huber，离散特征用 BCE
    - 只在被掩码且有效的位置上计算
    """
    
    def __init__(
        self,
        config,
        mask_rate: float = 0.5,
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.config = config
        self.mask_rate = mask_rate
        self.noise_std = noise_std
    
    def forward(self, backbone, x_original):
        """
        Args:
            backbone: 微型 Bi-Mamba 编码器
            x_original: (B, L, D) 原始输入
        
        Returns:
            total_loss: scalar
        """
        B, L, D = x_original.shape
        device = x_original.device
        cfg = backbone.config

        seq_len = int(getattr(cfg, 'SEQUENCE_LENGTH', L))
        effective_len = int(getattr(cfg, 'EFFECTIVE_SEQUENCE_LENGTH', seq_len))
        use_global_stats = bool(getattr(cfg, 'USE_GLOBAL_STATS_TOKEN', False))
        has_global = bool(use_global_stats and L == effective_len and L > seq_len)
        pkt_len = seq_len if has_global else L

        # ===== masking & noise =====
        valid_mask_idx = getattr(cfg, 'VALID_MASK_INDEX', None)
        valid = None
        if valid_mask_idx is not None and 0 <= int(valid_mask_idx) < D:
            try:
                valid = x_original[:, :pkt_len, int(valid_mask_idx)] > 0.5
            except Exception:
                valid = None

        # True 表示要被掩码重构
        mask_pos_pkt = torch.rand(B, pkt_len, device=device) < self.mask_rate
        if valid is not None:
            mask_pos_pkt = mask_pos_pkt & valid
        mask_pos_pkt = mask_pos_pkt.unsqueeze(-1)  # (B, pkt_len, 1)

        x_masked = x_original.clone()
        x_masked[:, :pkt_len, :] = x_masked[:, :pkt_len, :] * (~mask_pos_pkt)
        if valid_mask_idx is not None and 0 <= int(valid_mask_idx) < D:
            x_masked[:, :pkt_len, int(valid_mask_idx)] = x_original[:, :pkt_len, int(valid_mask_idx)]

        # 应用噪声增强（特征特定或全局）
        use_feature_specific = getattr(cfg, 'USE_FEATURE_SPECIFIC_NOISE', False)
        noise_apply = (~mask_pos_pkt.squeeze(-1))
        if valid is not None:
            noise_apply = noise_apply & valid

        if use_feature_specific:
            # 特征特定噪声增强（优化版）
            # 1. 包长：0.1倍标准差的高斯噪声
            length_idx = getattr(cfg, 'LENGTH_INDEX', None)
            if length_idx is not None and 0 <= int(length_idx) < D:
                li = int(length_idx)
                length_noise_std = float(getattr(cfg, 'PRETRAIN_LENGTH_NOISE_STD', 0.1))
                # 计算包长的标准差（基于当前批次）
                length_std = x_original[:, :pkt_len, li].std(dim=-1, keepdim=True).clamp_min(1e-6)
                n = torch.randn_like(x_masked[:, :pkt_len, li]) * length_noise_std * length_std
                x_masked[:, :pkt_len, li] = torch.where(noise_apply, x_masked[:, :pkt_len, li] + n, x_masked[:, :pkt_len, li])
            
            # 2. IAT (LogIAT)：0.05倍的泊松噪声
            # 泊松噪声模拟网络延迟的随机性，适用于IAT特征
            iat_idx = getattr(cfg, 'LOG_IAT_INDEX', getattr(cfg, 'IAT_INDEX', None))
            if iat_idx is not None and 0 <= int(iat_idx) < D:
                ii = int(iat_idx)
                iat_poisson_lambda = float(getattr(cfg, 'PRETRAIN_IAT_POISSON_LAMBDA', 0.05))
                iat_values = x_original[:, :pkt_len, ii]
                # 泊松噪声：使用泊松分布生成噪声，强度为原始值的0.05倍
                # 对于log空间的IAT，我们使用乘性泊松噪声的近似
                # 泊松噪声强度 = lambda * |原始值|
                poisson_scale = iat_poisson_lambda * torch.abs(iat_values).clamp_min(1e-6)
                # 生成泊松随机数（lambda = poisson_scale），然后转换为加性噪声
                # 使用正态近似：泊松(lambda) ≈ N(lambda, sqrt(lambda))
                poisson_mean = poisson_scale
                poisson_std = torch.sqrt(poisson_scale.clamp_min(1e-6))
                poisson_noise = torch.randn_like(iat_values) * poisson_std + (poisson_mean - poisson_scale)
                # 应用噪声（保持log空间的连续性）
                x_masked[:, :pkt_len, ii] = torch.where(noise_apply, x_masked[:, :pkt_len, ii] + poisson_noise, x_masked[:, :pkt_len, ii])
            
            # 3. BurstSize：使用全局噪声（保持兼容性）
            burst_idx = getattr(cfg, 'BURST_SIZE_INDEX', None)
            if burst_idx is not None and 0 <= int(burst_idx) < D:
                bi = int(burst_idx)
                n = torch.randn_like(x_masked[:, :pkt_len, bi]) * self.noise_std
                x_masked[:, :pkt_len, bi] = torch.where(noise_apply, x_masked[:, :pkt_len, bi] + n, x_masked[:, :pkt_len, bi])
        else:
            # 全局噪声（向后兼容）
            if self.noise_std > 0:
                length_idx = getattr(cfg, 'LENGTH_INDEX', None)
                if length_idx is not None and 0 <= int(length_idx) < D:
                    li = int(length_idx)
                    n = torch.randn_like(x_masked[:, :pkt_len, li]) * self.noise_std
                    x_masked[:, :pkt_len, li] = torch.where(noise_apply, x_masked[:, :pkt_len, li] + n, x_masked[:, :pkt_len, li])

                burst_idx = getattr(cfg, 'BURST_SIZE_INDEX', None)
                if burst_idx is not None and 0 <= int(burst_idx) < D:
                    bi = int(burst_idx)
                    n = torch.randn_like(x_masked[:, :pkt_len, bi]) * self.noise_std
                    x_masked[:, :pkt_len, bi] = torch.where(noise_apply, x_masked[:, :pkt_len, bi] + n, x_masked[:, :pkt_len, bi])

        # ===== encode & decode =====
        z_seq = backbone(x_masked, return_sequence=True)  # (B, L, d)
        x_recon = backbone.decoder(z_seq)  # (B, L, D)

        # ===== per-feature losses =====
        mask_scalar = mask_pos_pkt.squeeze(-1).to(dtype=x_original.dtype)  # (B, pkt_len)

        def _masked_mean(loss_tensor, mask_tensor):
            denom = mask_tensor.sum().clamp_min(1e-6)
            return (loss_tensor * mask_tensor).sum() / denom

        total_components = {}
        recon_loss = 0.0

        length_idx = getattr(cfg, 'LENGTH_INDEX', None)
        if length_idx is not None and 0 <= int(length_idx) < D:
            lt = int(length_idx)
            loss_len = F.mse_loss(x_recon[:, :pkt_len, lt], x_original[:, :pkt_len, lt], reduction='none')
            loss_len = _masked_mean(loss_len, mask_scalar)
            total_components['length'] = loss_len
            recon_loss = recon_loss + float(getattr(cfg, 'PRETRAIN_LENGTH_WEIGHT', 1.0)) * loss_len

        burst_idx = getattr(cfg, 'BURST_SIZE_INDEX', None)
        if burst_idx is not None and 0 <= int(burst_idx) < D:
            bt = int(burst_idx)
            loss_burst = F.mse_loss(x_recon[:, :pkt_len, bt], x_original[:, :pkt_len, bt], reduction='none')
            loss_burst = _masked_mean(loss_burst, mask_scalar)
            total_components['burst'] = loss_burst
            recon_loss = recon_loss + float(getattr(cfg, 'PRETRAIN_BURST_WEIGHT', 1.0)) * loss_burst

        dir_idx = getattr(cfg, 'DIRECTION_INDEX', None)
        if dir_idx is not None and 0 <= int(dir_idx) < D:
            dt = int(dir_idx)
            target_dir = (x_original[:, :pkt_len, dt] > 0).to(dtype=x_original.dtype)
            loss_dir = F.binary_cross_entropy_with_logits(
                x_recon[:, :pkt_len, dt],
                target_dir,
                reduction='none'
            )
            loss_dir = _masked_mean(loss_dir, mask_scalar)
            total_components['direction'] = loss_dir
            recon_loss = recon_loss + float(getattr(cfg, 'PRETRAIN_DIRECTION_WEIGHT', 1.0)) * loss_dir

        # LogIAT特征重建损失（MSE，连续值）
        log_iat_idx = getattr(cfg, 'LOG_IAT_INDEX', getattr(cfg, 'IAT_INDEX', None))
        if log_iat_idx is not None and 0 <= int(log_iat_idx) < D:
            iat = int(log_iat_idx)
            loss_iat = F.mse_loss(x_recon[:, :pkt_len, iat], x_original[:, :pkt_len, iat], reduction='none')
            loss_iat = _masked_mean(loss_iat, mask_scalar)
            total_components['logiat'] = loss_iat
            recon_loss = recon_loss + float(getattr(cfg, 'PRETRAIN_LOG_IAT_WEIGHT', 1.0)) * loss_iat

        if valid_mask_idx is not None and 0 <= int(valid_mask_idx) < D:
            vm = int(valid_mask_idx)
            target_vm = x_original[:, :pkt_len, vm]
            loss_vm = F.binary_cross_entropy_with_logits(
                x_recon[:, :pkt_len, vm],
                target_vm,
                reduction='none'
            )
            loss_vm = _masked_mean(loss_vm, mask_scalar)
            total_components['validmask'] = loss_vm
            recon_loss = recon_loss + float(getattr(cfg, 'PRETRAIN_VALIDMASK_WEIGHT', 0.5)) * loss_vm

        total_loss = float(getattr(cfg, 'PRETRAIN_RECON_WEIGHT', 1.0)) * recon_loss

        # 缓存分项，便于外部调试
        self.last_loss_dict = {
            'total': float(total_loss.item()),
            **{k: float(v.item()) for k, v in total_components.items()}
        }

        return total_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: (B, d) - feature vectors
            labels: (B,) - labels
            
        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal (self-comparison)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss
