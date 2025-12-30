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


class SoftF1Loss(nn.Module):
    """
    Soft F1 Loss: 直接优化 F1-Score (Binary)
    
    在训练时直接最大化 F1-Score，而不是优化 CrossEntropy (Accuracy)
    适用于二分类任务，特别是类别不平衡的入侵检测场景
    """
    
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 2) - 未经过 softmax 的 logits
            targets: (B,) - 0 或 1 的标签
        
        Returns:
            loss: scalar - 1 - F1 (最小化这个值等于最大化 F1)
        """
        # 获取恶意类（positive class, label=1）的概率
        probs = F.softmax(logits, dim=1)[:, 1]  # (B,)
        
        # 将 targets 转换为 float
        targets_float = targets.float()
        
        # 计算 TP, FP, FN
        tp = (probs * targets_float).sum()
        fp = (probs * (1 - targets_float)).sum()
        fn = ((1 - probs) * targets_float).sum()
        
        # 计算 Precision 和 Recall
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        
        # 计算 F1
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)
        
        # 返回 1 - F1 作为损失（最小化损失 = 最大化 F1）
        return 1 - f1


class DualStreamLoss(nn.Module):
    """
    Comprehensive loss for dual-stream training
    Includes soft-orthogonality, consistency, co-teaching,
    focal loss, margin loss, and Soft F1 Loss for better class separation
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.soft_f1_loss = SoftF1Loss()
    
        # Class weights for imbalanced classification
        # Weight malicious samples more heavily to handle class imbalance
        class_weight = torch.tensor([
            config.CLASS_WEIGHT_BENIGN,      # Weight for benign (class 0)
            config.CLASS_WEIGHT_MALICIOUS    # Weight for malicious (class 1)
        ], dtype=torch.float32)
        self.register_buffer('class_weight', class_weight)
        
        # Focal Loss parameters
        self.focal_alpha = getattr(config, 'FOCAL_ALPHA', 0.25)
        self.focal_gamma = getattr(config, 'FOCAL_GAMMA', 2.0)
        
        # Margin Loss parameters (ArcFace-style)
        self.margin_m = getattr(config, 'MARGIN_M', 0.5)  # Margin value
        self.margin_s = getattr(config, 'MARGIN_S', 30.0)  # Scale factor
        
        # Label Smoothing
        self.label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.1)

        self.consistency_temperature = float(getattr(config, 'CONSISTENCY_TEMPERATURE', 2.0))
        self.consistency_warmup_epochs = int(getattr(config, 'CONSISTENCY_WARMUP_EPOCHS', 0))
    
    def soft_orthogonality_loss(self, w_a, w_b):
        """
        Soft orthogonality loss to encourage different perspectives
        
        Args:
            w_a: (D_out, D_in) - weights of MLP A
            w_b: (D_out, D_in) - weights of MLP B
            
        Returns:
            loss: scalar
        """
        # Normalize weight vectors
        w_a_norm = F.normalize(w_a, p=2, dim=1)  # (D_out, D_in)
        w_b_norm = F.normalize(w_b, p=2, dim=1)  # (D_out, D_in)
        
        # Compute cosine similarity matrix
        similarity = torch.mm(w_a_norm, w_b_norm.T)  # (D_out, D_out)
        
        # Minimize squared similarity
        loss = torch.sum(similarity ** 2)
        
        return loss
    
    def consistency_loss(self, logits_a, logits_b):
        """
        Consistency loss to encourage agreement on predictions
        
        Args:
            logits_a: (B, 2)
            logits_b: (B, 2)
            
        Returns:
            loss: scalar
        """
        t = self.consistency_temperature
        if t <= 0:
            t = 1.0

        log_p_a = F.log_softmax(logits_a / t, dim=1)
        log_p_b = F.log_softmax(logits_b / t, dim=1)
        p_a = torch.exp(log_p_a)
        p_b = torch.exp(log_p_b)

        kl_ab = F.kl_div(log_p_a, p_b, reduction='batchmean')
        kl_ba = F.kl_div(log_p_b, p_a, reduction='batchmean')
        loss = 0.5 * (kl_ab + kl_ba) * (t * t)

        return loss

    def _label_smoothing_ce(self, logits, labels):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.label_smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)

        loss = -(smooth_labels * log_probs).sum(dim=1)
        return loss
    
    def focal_loss(self, logits, labels):
        """
        Focal Loss: focuses on hard examples to improve class separation
        
        Args:
            logits: (B, 2)
            labels: (B,)
            
        Returns:
            loss: (B,) - per-sample focal loss
        """
        ce_loss = self.ce_loss(logits, labels)
        pt = torch.exp(-ce_loss)  # Probability of true class
        
        # Alpha weighting
        alpha_t = self.focal_alpha * labels + (1 - self.focal_alpha) * (1 - labels)
        alpha_t = alpha_t.to(labels.device)
        
        # Focal loss
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss
    
    def margin_loss(self, logits, labels, features):
        """
        Margin Loss (ArcFace-style): adds margin between classes for better separation
        Conservative implementation: adds small margin without aggressive scaling
        
        Args:
            logits: (B, 2)
            labels: (B,)
            features: (B, D) - input features to classifier (not used, kept for compatibility)
            
        Returns:
            loss: scalar
        """
        # Apply margin only to the correct class logits (conservative approach)
        margin_logits = logits.clone()
        
        # Create one-hot encoding for labels
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # Add margin only to correct class (pushes correct class logits higher)
        margin_logits = margin_logits + self.margin_m * one_hot
        
        # Apply scale factor (if MARGIN_S != 1.0, scale; otherwise keep original scale)
        # Note: MARGIN_S=1.0 means no scaling, avoiding probability compression
        if self.margin_s != 1.0:
            margin_logits = margin_logits * self.margin_s
        
        # Compute cross-entropy with margin
        loss = F.cross_entropy(margin_logits, labels, reduction='mean')
        
        return loss
    
    def label_smoothing_loss(self, logits, labels):
        """
        Label Smoothing Cross-Entropy: reduces overconfidence
        
        Args:
            logits: (B, 2)
            labels: (B,)
            
        Returns:
            loss: (B,) - per-sample smoothed loss
        """
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Create smoothed labels
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.label_smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute loss
        loss = -torch.sum(smooth_labels * log_probs, dim=1)
        
        return loss
    
    def co_teaching_loss(self, logits_a, logits_b, labels, select_rate=0.7, use_focal=True):
        """Co-teaching: each network selects small-loss samples for the other."""
        batch_size = labels.shape[0]
        n_select = max(1, int(batch_size * select_rate))
        
        if use_focal:
            loss_a_per_sample = self.focal_loss(logits_a, labels)
            loss_b_per_sample = self.focal_loss(logits_b, labels)
        else:
            loss_a_per_sample = self.ce_loss(logits_a, labels)
            loss_b_per_sample = self.ce_loss(logits_b, labels)
        
        # A selects for B
        _, indices_a = torch.topk(loss_a_per_sample, n_select, largest=False)
        loss_b = loss_b_per_sample[indices_a].mean()
        
        # B selects for A
        _, indices_b = torch.topk(loss_b_per_sample, n_select, largest=False)
        loss_a = loss_a_per_sample[indices_b].mean()
        
        return loss_a, loss_b
    
    def forward(self, model, z, labels, sample_weights, epoch, total_epochs):
        logits_a, logits_b = model(z, return_separate=True)

        use_logit_margin = bool(getattr(self.config, 'USE_LOGIT_MARGIN', False))
        logit_margin_m = float(getattr(self.config, 'LOGIT_MARGIN_M', 0.0))
        logit_margin_warmup = int(getattr(self.config, 'LOGIT_MARGIN_WARMUP_EPOCHS', 0))
        margin_scale = 0.0
        if use_logit_margin and logit_margin_m > 0:
            one_hot = torch.zeros_like(logits_a)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            if logit_margin_warmup > 0:
                margin_scale = min(1.0, float(epoch + 1) / float(logit_margin_warmup))
            else:
                margin_scale = 1.0
            logits_a = logits_a - (one_hot * (logit_margin_m * margin_scale))
            logits_b = logits_b - (one_hot * (logit_margin_m * margin_scale))
        
        if sample_weights is None:
            sample_weights = torch.ones_like(labels, dtype=torch.float32, device=z.device)
        else:
            sample_weights = sample_weights.to(device=z.device, dtype=torch.float32)
        
        use_co_teaching = bool(getattr(self.config, 'USE_CO_TEACHING', False))
        co_teaching_warmup = int(getattr(self.config, 'CO_TEACHING_WARMUP_EPOCHS', 0))
        co_teaching_select_rate = float(getattr(self.config, 'CO_TEACHING_SELECT_RATE', 0.7))
        co_teaching_min_w = float(getattr(self.config, 'CO_TEACHING_MIN_SAMPLE_WEIGHT', 0.0))
        use_focal = bool(getattr(self.config, 'USE_FOCAL_LOSS', False))
        
        selected_for_a = None
        selected_for_b = None
        
        if use_co_teaching and epoch >= co_teaching_warmup:
            eligible = sample_weights >= co_teaching_min_w
            if eligible.any():
                logits_a_e = logits_a[eligible]
                logits_b_e = logits_b[eligible]
                labels_e = labels[eligible]
                loss_a_ct, loss_b_ct = self.co_teaching_loss(
                    logits_a_e, logits_b_e, labels_e,
                    select_rate=co_teaching_select_rate,
                    use_focal=use_focal
                )
                loss_sup_base = 0.5 * (loss_a_ct + loss_b_ct)
                loss_sup = loss_sup_base
                selected_for_a = eligible.nonzero(as_tuple=False).squeeze(1)
                selected_for_b = eligible.nonzero(as_tuple=False).squeeze(1)
            else:
                use_co_teaching = False
        
        if not use_co_teaching:
            # Balanced sampling already handled by the DataLoader; do not add class-weight compensation here.
            if use_focal:
                loss_a_per_sample = self.focal_loss(logits_a, labels)
                loss_b_per_sample = self.focal_loss(logits_b, labels)
            else:
                loss_a_per_sample = self._label_smoothing_ce(logits_a, labels)
                loss_b_per_sample = self._label_smoothing_ce(logits_b, labels)
            
            sup_loss_a = (loss_a_per_sample * sample_weights).mean()
            sup_loss_b = (loss_b_per_sample * sample_weights).mean()
            loss_sup_base = 0.5 * (sup_loss_a + sup_loss_b)
            loss_sup = loss_sup_base
        
        use_soft_f1 = bool(getattr(self.config, 'USE_SOFT_F1_LOSS', False))
        soft_f1_weight = float(getattr(self.config, 'SOFT_F1_WEIGHT', 0.0)) if use_soft_f1 else 0.0
        loss_f1 = torch.tensor(0.0, device=z.device)
        if use_soft_f1 and soft_f1_weight > 0:
            loss_f1_a = self.soft_f1_loss(logits_a, labels)
            loss_f1_b = self.soft_f1_loss(logits_b, labels)
            loss_f1 = 0.5 * (loss_f1_a + loss_f1_b)
            loss_sup = (1.0 - soft_f1_weight) * loss_sup_base + soft_f1_weight * loss_f1
        
        w_a, w_b = model.get_first_layer_weights()
        loss_orth = self.soft_orthogonality_loss(w_a, w_b)
        loss_con = self.consistency_loss(logits_a, logits_b)
        
        lambda_orth = self.config.get_dynamic_weight(
            epoch, total_epochs,
            self.config.SOFT_ORTH_WEIGHT_START,
            self.config.SOFT_ORTH_WEIGHT_END
        )
        
        lambda_con = self.config.get_dynamic_weight(
            epoch, total_epochs,
            self.config.CONSISTENCY_WEIGHT_START,
            self.config.CONSISTENCY_WEIGHT_END
        )

        if self.consistency_warmup_epochs > 0:
            warmup_scale = min(1.0, float(epoch + 1) / float(self.consistency_warmup_epochs))
            lambda_con = lambda_con * warmup_scale
        
        use_margin_loss = bool(getattr(self.config, 'USE_MARGIN_LOSS', False))
        margin_weight = float(getattr(self.config, 'MARGIN_LOSS_WEIGHT', 0.0)) if use_margin_loss else 0.0
        margin_w_start = float(getattr(self.config, 'MARGIN_LOSS_WEIGHT_START', 0.0))
        margin_w_end = float(getattr(self.config, 'MARGIN_LOSS_WEIGHT_END', margin_weight))
        lambda_margin = 0.0
        loss_margin = torch.tensor(0.0, device=z.device)
        if use_margin_loss and margin_weight > 0:
            lambda_margin = float(self.config.get_dynamic_weight(epoch, total_epochs, margin_w_start, margin_w_end))
            loss_margin_a = self.margin_loss(logits_a, labels, z)
            loss_margin_b = self.margin_loss(logits_b, labels, z)
            loss_margin = 0.5 * (loss_margin_a + loss_margin_b)

        total_loss = loss_sup + lambda_orth * loss_orth + lambda_con * loss_con + lambda_margin * loss_margin
        
        loss_dict = {
            'total': total_loss.item(),
            'supervision': loss_sup.item(),
            'supervision_base': loss_sup_base.item(),
            'soft_f1': loss_f1.item() if use_soft_f1 else 0.0,
            'orthogonality': loss_orth.item(),
            'consistency': loss_con.item(),
            'margin': loss_margin.item() if (use_margin_loss and margin_weight > 0) else 0.0,
            'use_logit_margin': float(bool(use_logit_margin and logit_margin_m > 0)),
            'logit_margin_m': float(logit_margin_m) if (use_logit_margin and logit_margin_m > 0) else 0.0,
            'logit_margin_scale': float(margin_scale) if (use_logit_margin and logit_margin_m > 0) else 0.0,
            'lambda_orth': lambda_orth,
            'lambda_con': lambda_con,
            'lambda_margin': float(lambda_margin),
            'soft_f1_weight': soft_f1_weight if use_soft_f1 else 0.0,
            'use_co_teaching': float(bool(use_co_teaching and epoch >= co_teaching_warmup)),
            'co_teaching_select_rate': float(co_teaching_select_rate),
            'co_teaching_min_sample_weight': float(co_teaching_min_w),
            'co_teaching_n_selected': int(selected_for_a.numel()) if selected_for_a is not None else 0
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
            x: (B, L, D) - input sequences
            return_features: bool - if True, also return backbone features
            return_separate: bool - if True, return separate logits from two streams
            
        Returns:
            logits: (B, 2) or ((B, 2), (B, 2)) if return_separate
            z: (B, 64) if return_features
        """
        # Extract features
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

