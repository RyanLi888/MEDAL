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
    Enhanced with BatchNorm for better feature scaling (对比学习特征幅度较小)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        # Use OUTPUT_DIM (backbone output dimension) for feature dimension
        # This is the dimension of features extracted by the backbone
        feature_dim = getattr(config, 'FEATURE_DIM', getattr(config, 'OUTPUT_DIM', config.MODEL_DIM))
        
        # 关键改进：对比学习特征需要BN层来拉伸幅度
        self.bn_input = nn.BatchNorm1d(feature_dim)

        self.mlp_a = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, config.CLASSIFIER_OUTPUT_DIM)
        )

        self.mlp_b = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, config.CLASSIFIER_OUTPUT_DIM)
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
            z: (B, 64) - features from backbone
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
        z = self.bn_input(z)
        
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.focal_alpha = float(getattr(config, 'FOCAL_ALPHA', 0.5))
        self.focal_gamma = float(getattr(config, 'FOCAL_GAMMA', 2.0))

    def focal_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        pt = torch.exp(-ce_loss)
        alpha_t = self.focal_alpha * labels + (1 - self.focal_alpha) * (1 - labels)
        alpha_t = alpha_t.to(labels.device)
        return alpha_t * (1 - pt) ** self.focal_gamma * ce_loss

    def forward(self, model, z, labels, sample_weights, epoch, total_epochs):
        logits_a, logits_b = model(z, return_separate=True)

        if sample_weights is None:
            sample_weights = torch.ones_like(labels, dtype=torch.float32, device=z.device)
        else:
            sample_weights = sample_weights.to(device=z.device, dtype=torch.float32)

        loss_a_per_sample = self.focal_loss(logits_a, labels)
        loss_b_per_sample = self.focal_loss(logits_b, labels)

        sup_loss_a = (loss_a_per_sample * sample_weights).mean()
        sup_loss_b = (loss_b_per_sample * sample_weights).mean()
        loss_sup = 0.5 * (sup_loss_a + sup_loss_b)
        total_loss = loss_sup

        loss_dict = {
            'total': total_loss.item(),
            'supervision': loss_sup.item(),
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
            z: (B, 64) if return_features
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

