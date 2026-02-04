"""
Dual-Stream Co-Detection Module
Implements soft-orthogonal dual-stream MLP for robust classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DualStreamMLP(nn.Module):
    """
    Dual-Stream MLP with Soft-Orthogonal Constraints
    
    Two parallel MLP classifiers with dynamic loss weighting
    Architecture: 32 → 16 → 8 → 2 (optimized for small datasets <1000 samples)
    Enhanced with LayerNorm for better feature scaling
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        # Use OUTPUT_DIM (backbone output dimension) for feature dimension
        # This is the dimension of features extracted by the backbone
        feature_dim = getattr(config, 'FEATURE_DIM', getattr(config, 'OUTPUT_DIM', config.MODEL_DIM))
        
        # 优化建议1: LayerNorm替代BatchNorm（更适合特征向量）
        # LayerNorm对特征向量的每个样本独立归一化，不依赖batch统计
        # 这对Mamba输出的抽象特征向量更合适
        self.ln_input = nn.LayerNorm(feature_dim)
        
        # 保留BatchNorm作为备选（可通过配置切换）
        self.bn_input = nn.BatchNorm1d(feature_dim)
        self.use_layernorm = getattr(config, 'USE_LAYERNORM_IN_CLASSIFIER', True)

        self.mlp_a = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, config.CLASSIFIER_OUTPUT_DIM)
        )

        self.mlp_b = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, config.CLASSIFIER_OUTPUT_DIM)
        )
        
        # Initialize with different random seeds
        self._init_weights_differently()
    
    def _init_weights_differently(self):
        """Initialize two MLPs with different random initializations"""
        # MLP A - default initialization
        for m in self.mlp_a.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # MLP B - different initialization
        for m in self.mlp_b.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)
    
    def forward(self, z, return_separate=False):
        """
        Forward pass
        
        Args:
            z: (B, 32) - features from backbone (32-dim feature vector)
            return_separate: bool - if True, return separate logits
            
        Returns:
            If return_separate:
                logits_a: (B, 2)
                logits_b: (B, 2)
            Else:
                logits_avg: (B, 2)
        """
        # 关键改进：先对输入特征进行BatchNorm
        # 防御性处理：如果误传入 (B, L, D)，先做 mean pooling 回到 (B, D)
        if z.dim() == 3:
            z = z.mean(dim=1)
        if z.dim() != 2:
            raise ValueError(f"DualStreamMLP expects z of shape (B, D) (or (B, L, D)), got {tuple(z.shape)}")
        
        # 优化建议1: 使用LayerNorm（推荐）或BatchNorm
        if self.use_layernorm:
            z = self.ln_input(z)  # LayerNorm: 对每个样本独立归一化
        else:
            z = self.bn_input(z)  # BatchNorm: 依赖batch统计
        
        logits_a = self.mlp_a(z)
        logits_b = self.mlp_b(z)
        
        if return_separate or self.training:
            return logits_a, logits_b
        else:
            # Average for inference
            logits_avg = (logits_a + logits_b) / 2.0
            return logits_avg
    
    def get_first_layer_weights(self):
        """Get first layer weights for orthogonality computation"""
        # 注意：现在第一层是Linear，第二层是BN
        w_a = self.mlp_a[0].weight  # (hidden_dim, input_dim)
        w_b = self.mlp_b[0].weight  # (hidden_dim, input_dim)
        return w_a, w_b

class DualStreamLoss(nn.Module):
    """
    Dual-Stream Loss with Co-teaching Support
    
    Supports:
    - Focal Loss for both streams
    - Co-teaching: mutual sample selection between two streams
    - Dynamic sample selection rate
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.focal_alpha = float(getattr(config, 'FOCAL_ALPHA', 0.5))
        self.focal_gamma = float(getattr(config, 'FOCAL_GAMMA', 2.0))
        
        # Co-teaching parameters
        self.use_co_teaching = getattr(config, 'USE_CO_TEACHING', False)
        self.co_teaching_warmup = getattr(config, 'CO_TEACHING_WARMUP_EPOCHS', 10)
        self.co_teaching_select_rate = getattr(config, 'CO_TEACHING_SELECT_RATE', 0.7)
        self.co_teaching_dynamic = getattr(config, 'CO_TEACHING_DYNAMIC_RATE', True)
        self.co_teaching_noise_rate = getattr(config, 'CO_TEACHING_NOISE_RATE', 0.3)
        
        # Track last logged epoch to avoid duplicate logging
        self._last_logged_epoch = -1
        
        if self.use_co_teaching:
            logger.info("✓ Co-teaching enabled:")
            logger.info(f"  - Warmup epochs: {self.co_teaching_warmup}")
            logger.info(f"  - Base select rate: {self.co_teaching_select_rate}")
            logger.info(f"  - Dynamic rate: {self.co_teaching_dynamic}")
            if self.co_teaching_dynamic:
                logger.info(f"  - Assumed noise rate: {self.co_teaching_noise_rate}")

    def focal_loss(self, logits, labels):
        """Focal Loss computation"""
        ce_loss = self.ce_loss(logits, labels)
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha * labels + (1 - self.focal_alpha) * (1 - labels)
        alpha_t = alpha_t.to(labels.device)
        return alpha_t * (1 - pt) ** self.focal_gamma * ce_loss

    def get_select_rate(self, epoch, total_epochs):
        """
        Calculate dynamic selection rate for co-teaching
        
        Strategy:
        - Warmup phase: gradually decrease from 1.0 to target rate
        - Stable phase: use fixed target rate
        """
        if not self.co_teaching_dynamic:
            return self.co_teaching_select_rate
        
        if epoch < self.co_teaching_warmup:
            # Warmup: linearly decrease from 1.0
            ratio = epoch / self.co_teaching_warmup
            select_rate = 1.0 - self.co_teaching_noise_rate * ratio
        else:
            # Stable: use target rate (1.0 - noise_rate)
            select_rate = 1.0 - self.co_teaching_noise_rate
        
        # Ensure select_rate is within reasonable bounds
        select_rate = max(0.5, min(1.0, select_rate))
        return select_rate

    def forward(self, model, z, labels, sample_weights, epoch, total_epochs):
        """
        Forward pass with optional co-teaching
        
        Args:
            model: DualStreamMLP model
            z: (B, D) - input features
            labels: (B,) - ground truth labels
            sample_weights: (B,) - sample weights (can be None)
            epoch: current epoch (0-indexed)
            total_epochs: total training epochs
            
        Returns:
            total_loss: scalar loss
            loss_dict: dictionary of loss components
        """
        # Get separate logits from two streams
        logits_a, logits_b = model(z, return_separate=True)

        # Ensure sample_weights is on correct device
        if sample_weights is None:
            sample_weights = torch.ones_like(labels, dtype=torch.float32, device=z.device)
        else:
            sample_weights = sample_weights.to(device=z.device, dtype=torch.float32)

        # Compute per-sample losses
        loss_a_per_sample = self.focal_loss(logits_a, labels)  # (B,)
        loss_b_per_sample = self.focal_loss(logits_b, labels)  # (B,)
        
        # Co-teaching: mutual sample selection
        if self.use_co_teaching and epoch >= self.co_teaching_warmup:
            B = labels.size(0)
            select_rate = self.get_select_rate(epoch, total_epochs)
            num_select = max(1, int(B * select_rate))
            
            # Stream A selects samples for Stream B
            # Select samples with smallest loss from A's perspective
            _, idx_a_select = torch.topk(loss_a_per_sample, num_select, largest=False)
            loss_b_selected = loss_b_per_sample[idx_a_select]
            weights_b_selected = sample_weights[idx_a_select]
            
            # Stream B selects samples for Stream A
            # Select samples with smallest loss from B's perspective
            _, idx_b_select = torch.topk(loss_b_per_sample, num_select, largest=False)
            loss_a_selected = loss_a_per_sample[idx_b_select]
            weights_a_selected = sample_weights[idx_b_select]
            
            # Apply sample weights to selected samples
            sup_loss_a = (loss_a_selected * weights_a_selected).mean()
            sup_loss_b = (loss_b_selected * weights_b_selected).mean()
            
            # Log co-teaching info (only once per epoch to avoid spam)
            if epoch != self._last_logged_epoch and epoch % 50 == 0:
                self._last_logged_epoch = epoch
                logger.info(f"  [Co-teaching] Epoch {epoch}: select_rate={select_rate:.3f}, "
                          f"num_select={num_select}/{B}")
        else:
            # Standard training: use all samples
            sup_loss_a = (loss_a_per_sample * sample_weights).mean()
            sup_loss_b = (loss_b_per_sample * sample_weights).mean()

        # Combine losses from both streams
        loss_sup = 0.5 * (sup_loss_a + sup_loss_b)
        total_loss = loss_sup

        loss_dict = {
            'total': total_loss.item(),
            'supervision': loss_sup.item(),
            'stream_a': sup_loss_a.item(),
            'stream_b': sup_loss_b.item(),
        }

        return total_loss, loss_dict



class MEDAL_Classifier(nn.Module):
    """
    Complete MEDAL Classifier
    Combines frozen backbone with dual-stream MLP
    """
    
    def __init__(self, backbone, config):
        super().__init__()
        
        self.backbone = backbone
        self.dual_mlp = DualStreamMLP(config)
        self.config = config
        
        # 优化建议1: 在backbone输出后添加LayerNorm（可选）
        # 这有助于稳定特征分布，特别是当backbone在不同阶段训练时
        feature_dim = getattr(config, 'FEATURE_DIM', getattr(config, 'OUTPUT_DIM', config.MODEL_DIM))
        self.use_backbone_layernorm = getattr(config, 'USE_BACKBONE_LAYERNORM', False)
        if self.use_backbone_layernorm:
            self.backbone_ln = nn.LayerNorm(feature_dim)
        else:
            self.backbone_ln = None
        
        # By default, keep backbone frozen. Can be overridden in Stage 3 by enabling
        # config.FINETUNE_BACKBONE and selectively unfreezing parameters.
        if not getattr(self.config, 'FINETUNE_BACKBONE', False):
            self.backbone.freeze()
    
    def forward(self, x, return_features=False, return_separate=False):
        """
        Forward pass
        
        Args:
            x: (B, L, D) - input sequences OR (B, D) - pre-extracted features
            return_features: bool - if True, also return backbone features
            return_separate: bool - if True, return separate logits from two streams
            
        Returns:
            logits: (B, 2) or ((B, 2), (B, 2)) if return_separate
            z: (B, 32) if return_features (32-dim feature vector from backbone)
        """
        # Determine if input is already features (2D) or sequences (3D)
        is_2d_input = (isinstance(x, torch.Tensor) and x.dim() == 2)
        
        # Extract features
        if is_2d_input:
            # Input is already features, use directly
            z = x
        else:
            # Input is sequences, extract features using backbone
            if getattr(self.config, 'FINETUNE_BACKBONE', False):
                z = self.backbone(x, return_sequence=False)
            else:
                with torch.no_grad():
                    z = self.backbone(x, return_sequence=False)
        
        # 优化建议1: 在backbone输出后应用LayerNorm（如果启用）
        if self.use_backbone_layernorm and self.backbone_ln is not None:
            z = self.backbone_ln(z)
        
        # Classification
        if return_separate or self.training:
            logits_a, logits_b = self.dual_mlp(z, return_separate=True)
            
            if return_features:
                return (logits_a, logits_b), z
            else:
                return logits_a, logits_b
        else:
            logits = self.dual_mlp(z, return_separate=False)
            
            if return_features:
                return logits, z
            else:
                return logits
    
    def predict(self, x, threshold=None):
        """
        Predict labels and probabilities
        
        Args:
            x: (B, L, D) - input sequences
            threshold: float - decision threshold for malicious class (default: 0.5)
                       If None, uses config.MALICIOUS_THRESHOLD or 0.5
            
        Returns:
            predictions: (B,) - predicted labels
            probabilities: (B, 2) - predicted probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self(x, return_separate=False)
            probs = F.softmax(logits, dim=1)
            
            # Use threshold for malicious class if provided
            if threshold is None:
                threshold = getattr(self.config, 'MALICIOUS_THRESHOLD', 0.5)
            
            # Apply threshold: if malicious probability > threshold, predict malicious
            preds = (probs[:, 1] > threshold).long()
        
        return preds, probs

