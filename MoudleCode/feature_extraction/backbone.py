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
        self.pos_encoding = SinusoidalPositionalEncoding(config.MODEL_DIM, max_len=config.SEQUENCE_LENGTH)
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


class DualStreamBiMambaBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = getattr(config, 'OUTPUT_DIM', getattr(config, 'FEATURE_DIM', config.MODEL_DIM))

        model_dim = int(getattr(config, 'MODEL_DIM', 32))
        d_len = max(1, model_dim // 2)
        d_struct = max(1, model_dim - d_len)

        self._d_len = int(d_len)
        self._d_struct = int(d_struct)

        self.length_index = getattr(config, 'LENGTH_INDEX', 0)
        self.direction_index = getattr(config, 'DIRECTION_INDEX', None)
        self.burst_index = getattr(config, 'BURST_SIZE_INDEX', None)
        self.valid_mask_index = getattr(config, 'VALID_MASK_INDEX', None)

        self.len_embed = nn.Sequential(
            nn.Linear(1, self._d_len),
            nn.LayerNorm(self._d_len),
            nn.GELU(),
        )
        self.len_pos = SinusoidalPositionalEncoding(self._d_len, max_len=config.SEQUENCE_LENGTH)
        self.len_dropout = nn.Dropout(config.EMBEDDING_DROPOUT)

        self.struct_embed = _DualStreamHybridEmbedding(self._d_struct)
        self.struct_pos = SinusoidalPositionalEncoding(self._d_struct, max_len=config.SEQUENCE_LENGTH)
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

    def forward(self, x, return_sequence=False):
        valid_mask = None
        if self.valid_mask_index is not None:
            try:
                if int(self.valid_mask_index) >= 0 and int(self.valid_mask_index) < x.shape[-1]:
                    valid_mask = x[:, :, int(self.valid_mask_index)]
            except Exception:
                valid_mask = None

        length_index = int(self.length_index) if self.length_index is not None else 0
        x_len = x[:, :, length_index:length_index + 1]
        x_len = self.len_embed(x_len)
        x_len = self.len_pos(x_len)
        x_len = self.len_dropout(x_len)

        x_dir = None
        x_burst = None
        if self.direction_index is not None:
            try:
                di = int(self.direction_index)
                if 0 <= di < x.shape[-1]:
                    x_dir = x[:, :, di]
            except Exception:
                x_dir = None
        if self.burst_index is not None:
            try:
                bi = int(self.burst_index)
                if 0 <= bi < x.shape[-1]:
                    x_burst = x[:, :, bi:bi + 1]
            except Exception:
                x_burst = None

        if x_dir is None:
            x_dir = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        if x_burst is None:
            x_burst = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)

        x_struct = self.struct_embed(x_dir, x_burst)
        x_struct = self.struct_pos(x_struct)
        x_struct = self.struct_dropout(x_struct)

        x_len_fwd = x_len
        for layer in self.len_forward_layers:
            x_len_fwd = layer(x_len_fwd)
        x_len_bwd = torch.flip(x_len, dims=[1])
        for layer in self.len_backward_layers:
            x_len_bwd = layer(x_len_bwd)
        x_len_bwd = torch.flip(x_len_bwd, dims=[1])
        h_len = self.len_proj(torch.cat([x_len_fwd, x_len_bwd], dim=-1))

        x_struct_fwd = x_struct
        for layer in self.struct_forward_layers:
            x_struct_fwd = layer(x_struct_fwd)
        x_struct_bwd = torch.flip(x_struct, dims=[1])
        for layer in self.struct_backward_layers:
            x_struct_bwd = layer(x_struct_bwd)
        x_struct_bwd = torch.flip(x_struct_bwd, dims=[1])
        h_struct = self.struct_proj(torch.cat([x_struct_fwd, x_struct_bwd], dim=-1))

        h = torch.cat([h_len, h_struct], dim=-1)
        gate = self.fusion_gate(h)
        h = gate * h + (1.0 - gate) * self.fusion_proj(h)

        if return_sequence:
            return self.projection(h)

        if valid_mask is not None:
            m = (valid_mask > 0.5).to(dtype=h.dtype, device=h.device).unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1e-6)
            pooled = (h * m).sum(dim=1) / denom
        else:
            pooled = torch.mean(h, dim=1)
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
    SimMTM: Masked Time-series Modeling Loss
    
    Implements the complete three-step process:
    1. Masking: Random masking of time steps
    2. Manifold Calibration: Compute batch center for alignment
    3. Reconstruction: Reconstruct masked positions
    """
    
    def __init__(self, mask_rate=0.5, manifold_weight=0.1):
        super().__init__()
        self.mask_rate = mask_rate
        self.manifold_weight = manifold_weight
    
    def forward(self, backbone, x_original):
        """
        Compute SimMTM loss with manifold calibration
        
        Args:
            backbone: Micro-Bi-Mamba backbone model
            x_original: (B, L, D_in) - original input
            
        Returns:
            loss: scalar
            loss_dict: dictionary with individual loss components
        """
        B, L, D = x_original.shape

        valid_mask_index = getattr(backbone.config, 'VALID_MASK_INDEX', None)
        valid = None
        if valid_mask_index is not None:
            try:
                if int(valid_mask_index) >= 0 and int(valid_mask_index) < D:
                    valid = x_original[:, :, int(valid_mask_index)] > 0.5
            except Exception:
                valid = None
        
        # Step 1: Create mask (randomly mask 50% of time steps)
        mask = torch.rand(B, L, device=x_original.device) < self.mask_rate
        if valid is not None:
            mask = mask & valid
        
        # Masked input
        x_masked = x_original.clone()
        x_masked[mask] = 0.0
        if valid_mask_index is not None and int(valid_mask_index) >= 0 and int(valid_mask_index) < D:
            x_masked[:, :, int(valid_mask_index)] = x_original[:, :, int(valid_mask_index)]
        
        # Get sequence features from masked input
        z_seq = backbone(x_masked, return_sequence=True)  # (B, L, d) where d=OUTPUT_DIM
        
        # Step 2: Manifold Calibration
        # Pool to get feature vector for each sample
        if valid is not None:
            m = valid.to(dtype=z_seq.dtype).unsqueeze(-1)  # (B, L, 1)
            denom = m.sum(dim=1).clamp_min(1e-6)
            z_pooled = (z_seq * m).sum(dim=1) / denom
        else:
            z_pooled = torch.mean(z_seq, dim=1)  # (B, d)
        
        # Compute batch center (theoretical center) for manifold calibration
        # Group by labels if available, otherwise use entire batch
        with torch.no_grad():
            z_batch_center = torch.mean(z_pooled, dim=0, keepdim=True)  # (1, d)
            z_batch_center = z_batch_center.expand(B, -1)  # (B, d)
        
        # Manifold alignment loss (align features to batch center)
        manifold_loss = F.mse_loss(z_pooled, z_batch_center.detach())
        
        # Step 3: Reconstruction loss
        # Decode to reconstruct original features
        x_recon = backbone.decoder(z_seq)  # (B, L, D_in)
        
        # Reconstruction loss (only on masked positions)
        mask_expanded = mask.unsqueeze(-1).expand_as(x_original)
        if valid_mask_index is not None and int(valid_mask_index) >= 0 and int(valid_mask_index) < D:
            mask_expanded[:, :, int(valid_mask_index)] = False
        if mask_expanded.sum() > 0:
            recon_loss = F.mse_loss(x_recon[mask_expanded], x_original[mask_expanded])
        else:
            recon_loss = torch.tensor(0.0, device=x_original.device)
        
        # Total SimMTM loss: reconstruction + manifold calibration
        total_loss = recon_loss + self.manifold_weight * manifold_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'manifold': manifold_loss.item()
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

