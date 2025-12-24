"""
Micro-Bi-Mamba Backbone for Feature Extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .mamba_block import MambaLayer


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
        self.decoder = nn.Linear(self.output_dim, config.INPUT_FEATURE_DIM)
        
        self.frozen = False
    
    def forward(self, x, return_sequence=False):
        """
        Forward pass
        
        Args:
            x: (B, L, 5) - input features
            return_sequence: If True, return sequence features; else return pooled features
            
        Returns:
            z: (B, output_dim) if return_sequence=False, else (B, L, output_dim)
        """
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
            x_original: (B, L, 5) - original input
            
        Returns:
            loss: scalar
            loss_dict: dictionary with individual loss components
        """
        B, L, D = x_original.shape
        
        # Step 1: Create mask (randomly mask 50% of time steps)
        mask = torch.rand(B, L, device=x_original.device) < self.mask_rate
        
        # Masked input
        x_masked = x_original.clone()
        x_masked[mask] = 0.0
        
        # Get sequence features from masked input
        z_seq = backbone(x_masked, return_sequence=True)  # (B, L, 64)
        
        # Step 2: Manifold Calibration
        # Pool to get feature vector for each sample
        z_pooled = torch.mean(z_seq, dim=1)  # (B, 64)
        
        # Compute batch center (theoretical center) for manifold calibration
        # Group by labels if available, otherwise use entire batch
        with torch.no_grad():
            z_batch_center = torch.mean(z_pooled, dim=0, keepdim=True)  # (1, 64)
            z_batch_center = z_batch_center.expand(B, -1)  # (B, 64)
        
        # Manifold alignment loss (align features to batch center)
        manifold_loss = F.mse_loss(z_pooled, z_batch_center.detach())
        
        # Step 3: Reconstruction loss
        # Decode to reconstruct original features
        x_recon = backbone.decoder(z_seq)  # (B, L, 5)
        
        # Reconstruction loss (only on masked positions)
        mask_expanded = mask.unsqueeze(-1).expand_as(x_original)
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

