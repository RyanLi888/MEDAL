"""
Hybrid Court: Label Noise Correction Module
两阶段标签矫正策略 (CL + AUM + KNN):

阶段1: Keep和Flip决策
    - 使用CL（静态特征度量）、AUM（动态训练度量）和KNN（邻居投票）进行决策
    - 支持保守模式（Conservative）和激进模式（Aggressive）

阶段2: 保守补刀/救援（可选）
    - LateFlip: Phase1保持但Stage2指标显示应翻转的样本
    - UndoFlip: Phase1翻转但Stage2指标显示应撤销的样本

核心原则: 使用CL、AUM和KNN三重校验进行标签矫正
"""
import numpy as np
import torch
import torch.nn as nn
import copy
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from cleanlab.filter import find_label_issues
import logging

from .hybrid_court_cl_aum import correct_labels_cl_aum

logger = logging.getLogger(__name__)

class CLProjectionHead(nn.Module):
    """
    优化建议2: 为CL创建独立的投影头
    专门用于计算CL的置信度和KNN距离，与分类MLP分离
    设计: 较小的MLP (例如 256 -> 128 -> 64)
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.projection = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),  # 使用LayerNorm而非BatchNorm
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim),
        )
    
    def forward(self, x):
        """
        Args:
        x: (B, input_dim) - Mamba输出的特征向量
        Returns:
        z: (B, output_dim) - 投影后的特征，用于CL和KNN
        """
        return self.projection(x)


class ConfidentLearning:
    """Confident Learning (CL) - Probabilistic Diagnosis with Projection Head"""
    
    def __init__(self, n_folds=5, use_projection_head=True, feature_dim=32, device='cpu', config=None):
        self.n_folds = n_folds
        self.use_projection_head = use_projection_head
        self.device = device
        self.confident_joint = None
        self.thresholds = None
        
        # 优化建议2: 创建独立的投影头
        if use_projection_head:
            self.projection_head = CLProjectionHead(
                input_dim=feature_dim,
                hidden_dim=128,
                output_dim=64
            ).to(device)
            # 优化建议2: 投影头训练模式配置
            # 默认冻结（推理模式），如果需要训练可通过配置启用
            if config is not None:
                self.projection_trainable = getattr(config, 'CL_PROJECTION_TRAINABLE', False)
            else:
                self.projection_trainable = False
            if not self.projection_trainable:
                self.projection_head.eval()  # 推理模式（冻结）
                for param in self.projection_head.parameters():
                    param.requires_grad = False
            else:
                self.projection_head.train()  # 训练模式
        else:
            self.projection_head = None
    
    def fit_predict(self, features, labels):
        """
        Args:
        features: (N, D) - Mamba提取的特征向量
        labels: (N,) - 噪声标签
        """
        n_samples = len(labels)
        n_classes = len(np.unique(labels))
        
        # 优化建议2: 如果使用投影头，先投影特征
        if self.use_projection_head and self.projection_head is not None:
            features_tensor = torch.FloatTensor(features).to(self.device)
            with torch.no_grad():
                features_projected = self.projection_head(features_tensor).cpu().numpy()
            features_for_cl = features_projected
        else:
            features_for_cl = features
        
        pred_probs = np.zeros((n_samples, n_classes))
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(features_for_cl):
            X_train, X_val = features_for_cl[train_idx], features_for_cl[val_idx]
            y_train = labels[train_idx]
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            pred_probs[val_idx] = clf.predict_proba(X_val)

            label_issues = find_label_issues(labels=labels, pred_probs=pred_probs)
        suspected_noise_mask = label_issues.astype(bool)
        pred_labels = np.argmax(pred_probs, axis=1)
        
        self.last_suspected_noise = suspected_noise_mask
        self.last_pred_labels = pred_labels
        self.last_pred_probs = pred_probs
        
        logger.info(f"CL: Identified {suspected_noise_mask.sum()} / {n_samples} suspected noise")
        return suspected_noise_mask, pred_labels, pred_probs


class KNNSemanticVoting:
    """KNN: Semantic Voting"""
    
    def __init__(self, k=20, metric='euclidean', weight_eps=1e-6):
        self.k = k
        self.metric = metric
        self.weight_eps = weight_eps
        self.knn = None
    
    def fit(self, features):
        self.knn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric)
        self.knn.fit(features)
    
    def predict_semantic_label(self, features, labels):
        distances, indices = self.knn.kneighbors(features)
        n_samples = len(labels)
        neighbor_labels = np.zeros(n_samples, dtype=int)
        neighbor_consistency = np.zeros(n_samples)
        
        for i in range(n_samples):
            neighbor_idx = indices[i]
            neighbor_d = distances[i]
            # 排除自身
            if neighbor_idx[0] == i and neighbor_d[0] <= self.weight_eps:
                neighbor_idx = neighbor_idx[1:self.k + 1]
                neighbor_d = neighbor_d[1:self.k + 1]
            else:
                neighbor_idx = neighbor_idx[:self.k]
                neighbor_d = neighbor_d[:self.k]
            
            neighbor_y = labels[neighbor_idx]
            w = 1.0 / (neighbor_d + self.weight_eps)
            vote_0 = w[neighbor_y == 0].sum()
            vote_1 = w[neighbor_y == 1].sum()
            total_w = vote_0 + vote_1
            
            neighbor_labels[i] = 0 if vote_0 > vote_1 else 1
            neighbor_consistency[i] = max(vote_0, vote_1) / total_w if total_w > 0 else 0.0
        
        self.last_neighbor_labels = neighbor_labels
        self.last_neighbor_consistency = neighbor_consistency
        return neighbor_labels, neighbor_consistency


class HybridCourt:
    """
    Hybrid Court v15: CL + AUM + KNN 三重校验标签矫正策略
    
    核心原则: 使用CL（静态特征）+ AUM（动态训练稳定性）+ KNN（语义投票）进行标签矫正
    
    融合决策逻辑：
    - Rule 1: 确信干净 (High Confidence Clean)
      * IF (CL置信度 > threshold) AND (AUM > 0) → Keep (Tier 1)
      * 解释: 长得像（CL高），而且学得快（AUM正）。这是绝对的干净样本。
    
    - Rule 2: 疑似噪声 (Dirty Candidates)
      * IF (CL置信度 < threshold) OR (AUM < 0) → Dirty Candidate
      * 解释: 只要有一个指标觉得它不对劲，就把它列入嫌疑名单。
    
    - Rule 3: 拯救/翻转 (Correction - 仅针对 Dirty Candidate)
      * IF KNN Purity > 0.8 (且指向同一新类别) → Flip (Tier 2)
      * ELSE → Drop (Tier 3)
    """
    
    def __init__(self, config):
        self.config = config
        # 优化建议2: 为CL创建独立的投影头
        use_cl_projection = getattr(config, 'USE_CL_PROJECTION_HEAD', True)
        feature_dim = getattr(config, 'FEATURE_DIM', getattr(config, 'OUTPUT_DIM', 32))
        # 自动检测设备
        import torch
        device = getattr(config, 'DEVICE', None)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 如果device是torch.device对象，转换为字符串
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device
        
        self.cl = ConfidentLearning(
            n_folds=config.CL_K_FOLD,
            use_projection_head=use_cl_projection,
            feature_dim=feature_dim,
            device=device,
            config=config
        )
        
        self.knn = KNNSemanticVoting(
            k=config.KNN_NEIGHBORS,
            metric=getattr(config, 'KNN_METRIC', 'euclidean')
        )

    def correct_labels(self, features, noisy_labels, device='cpu', y_true=None):
        """
        两阶段标签矫正 (CL + KNN，去除MADE)
        
        阶段1: Keep和Flip决策
            - Keep条件:
              * 恶意标签(label=1): CL认为不是噪声 且 KNN一致性为high (>= high_threshold)
              * 正常标签(label=0): CL认为不是噪声 且 KNN一致性 > 0.55
            - Flip条件:
              * CL认为是噪声 且 KNN不支持当前标签 (neighbor_labels != current_label)
        
        阶段2: 重新计算CL和KNN（使用阶段1翻转后的新标签），Flip决策
            - Flip条件:
              * CL认为是噪声 且 KNN不支持当前标签 (neighbor_labels != current_label)
        
        KNN一致性等级：
        - Low: < medium_threshold
        - Medium: >= medium_threshold 且 < high_threshold
        - High: >= high_threshold
        """
        cl_threshold = float(getattr(self.config, 'STAGE2_CL_THRESHOLD', 0.7))
        aum_threshold = float(getattr(self.config, 'STAGE2_AUM_THRESHOLD', 0.0))
        aum_epochs = int(getattr(self.config, 'AUM_EPOCHS', 30))
        aum_batch_size = int(getattr(self.config, 'AUM_BATCH_SIZE', 128))
        aum_lr = float(getattr(self.config, 'AUM_LR', 0.01))
        knn_purity_threshold = float(getattr(self.config, 'STAGE2_KNN_PURITY_THRESHOLD', 0.8))
        use_drop = bool(getattr(self.config, 'STAGE2_USE_DROP', False))

        # Phase1 / Phase2 策略阈值（可通过 config 覆盖）
        phase1_aggressive = bool(getattr(self.config, 'PHASE1_AGGRESSIVE', False))
        phase1_aggressive_malicious_aum_threshold = float(getattr(self.config, 'PHASE1_AGGRESSIVE_MALICIOUS_AUM_THRESHOLD', 0.05))
        phase1_aggressive_malicious_cl_threshold = float(getattr(self.config, 'PHASE1_AGGRESSIVE_MALICIOUS_CL_THRESHOLD', 0.6))
        phase1_aggressive_malicious_knn_cons_threshold = float(getattr(self.config, 'PHASE1_AGGRESSIVE_MALICIOUS_KNN_CONS_THRESHOLD', 0.6))
        phase1_aggressive_benign_aum_threshold = float(getattr(self.config, 'PHASE1_AGGRESSIVE_BENIGN_AUM_THRESHOLD', -0.05))
        phase1_aggressive_benign_knn_threshold = float(getattr(self.config, 'PHASE1_AGGRESSIVE_BENIGN_KNN_THRESHOLD', 0.55))

        phase1_malicious_aum_threshold = float(getattr(self.config, 'PHASE1_MALICIOUS_AUM_THRESHOLD', 0.0))
        phase1_malicious_knn_threshold = float(getattr(self.config, 'PHASE1_MALICIOUS_KNN_THRESHOLD', 0.7))
        phase1_malicious_cl_low = float(getattr(self.config, 'PHASE1_MALICIOUS_CL_LOW', 0.5))
        phase1_benign_aum_threshold = float(getattr(self.config, 'PHASE1_BENIGN_AUM_THRESHOLD', -0.5))
        phase1_benign_knn_threshold = float(getattr(self.config, 'PHASE1_BENIGN_KNN_THRESHOLD', 0.7))

        phase2_enable = bool(getattr(self.config, 'PHASE2_ENABLE', False))
        phase2_independent = bool(getattr(self.config, 'PHASE2_INDEPENDENT', True))
        # Phase2独立翻转策略参数
        phase2_malicious_aum_threshold = float(getattr(self.config, 'PHASE2_MALICIOUS_AUM_THRESHOLD', 0.05))
        phase2_malicious_cl_threshold = float(getattr(self.config, 'PHASE2_MALICIOUS_CL_THRESHOLD', 0.65))
        phase2_malicious_knn_cons_threshold = float(getattr(self.config, 'PHASE2_MALICIOUS_KNN_CONS_THRESHOLD', 0.55))
        phase2_benign_aum_threshold = float(getattr(self.config, 'PHASE2_BENIGN_AUM_THRESHOLD', -0.2))
        phase2_benign_knn_threshold = float(getattr(self.config, 'PHASE2_BENIGN_KNN_THRESHOLD', 0.6))
        # 旧策略参数（兼容）
        phase2_late_flip_aum_threshold = float(getattr(self.config, 'PHASE2_LATE_FLIP_AUM_THRESHOLD', -0.5))
        phase2_late_flip_knn_threshold = float(getattr(self.config, 'PHASE2_LATE_FLIP_KNN_THRESHOLD', 0.65))
        phase2_late_flip_cl_threshold = float(getattr(self.config, 'PHASE2_LATE_FLIP_CL_THRESHOLD', 0.4))
        phase2_undo_flip_aum_threshold = float(getattr(self.config, 'PHASE2_UNDO_FLIP_AUM_THRESHOLD', -0.8))
        phase2_undo_flip_cl_threshold = float(getattr(self.config, 'PHASE2_UNDO_FLIP_CL_THRESHOLD', 0.25))
        phase2_undo_flip_use_and = bool(getattr(self.config, 'PHASE2_UNDO_FLIP_USE_AND', False))
        phase2_undo_flip_p1_aum_hesitant = float(getattr(self.config, 'PHASE2_UNDO_FLIP_P1_AUM_HESITANT', -0.2))
        phase2_undo_flip_p1_aum_strong = float(getattr(self.config, 'PHASE2_UNDO_FLIP_P1_AUM_STRONG', -0.5))
        phase2_undo_flip_p2_aum_weak = float(getattr(self.config, 'PHASE2_UNDO_FLIP_P2_AUM_WEAK', 1.5))
        recompute_stage2_metrics = bool(getattr(self.config, 'RECOMPUTE_STAGE2_METRICS', True))

        clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs = correct_labels_cl_aum(
            self,
            features=np.asarray(features),
            noisy_labels=np.asarray(noisy_labels),
            device=str(device),
            y_true=y_true,
            cl_threshold=cl_threshold,
            aum_threshold=aum_threshold,
            aum_epochs=aum_epochs,
            aum_batch_size=aum_batch_size,
            aum_lr=aum_lr,
            knn_purity_threshold=knn_purity_threshold,
            use_drop=use_drop,
            phase1_aggressive=phase1_aggressive,
            phase1_aggressive_malicious_aum_threshold=phase1_aggressive_malicious_aum_threshold,
            phase1_aggressive_malicious_cl_threshold=phase1_aggressive_malicious_cl_threshold,
            phase1_aggressive_malicious_knn_cons_threshold=phase1_aggressive_malicious_knn_cons_threshold,
            phase1_aggressive_benign_aum_threshold=phase1_aggressive_benign_aum_threshold,
            phase1_aggressive_benign_knn_threshold=phase1_aggressive_benign_knn_threshold,
            phase1_malicious_aum_threshold=phase1_malicious_aum_threshold,
            phase1_malicious_knn_threshold=phase1_malicious_knn_threshold,
            phase1_malicious_cl_low=phase1_malicious_cl_low,
            phase1_benign_aum_threshold=phase1_benign_aum_threshold,
            phase1_benign_knn_threshold=phase1_benign_knn_threshold,
            phase2_enable=phase2_enable,
            phase2_independent=phase2_independent,
            phase2_malicious_aum_threshold=phase2_malicious_aum_threshold,
            phase2_malicious_cl_threshold=phase2_malicious_cl_threshold,
            phase2_malicious_knn_cons_threshold=phase2_malicious_knn_cons_threshold,
            phase2_benign_aum_threshold=phase2_benign_aum_threshold,
            phase2_benign_knn_threshold=phase2_benign_knn_threshold,
            phase2_late_flip_aum_threshold=phase2_late_flip_aum_threshold,
            phase2_late_flip_knn_threshold=phase2_late_flip_knn_threshold,
            phase2_late_flip_cl_threshold=phase2_late_flip_cl_threshold,
            phase2_undo_flip_aum_threshold=phase2_undo_flip_aum_threshold,
            phase2_undo_flip_cl_threshold=phase2_undo_flip_cl_threshold,
            phase2_undo_flip_use_and=phase2_undo_flip_use_and,
            phase2_undo_flip_p1_aum_hesitant=phase2_undo_flip_p1_aum_hesitant,
            phase2_undo_flip_p1_aum_strong=phase2_undo_flip_p1_aum_strong,
            phase2_undo_flip_p2_aum_weak=phase2_undo_flip_p2_aum_weak,
            recompute_stage2_metrics=recompute_stage2_metrics,
        )

        return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs

    def _log_final_stats(self, n_samples, tier_info, action_mask, weights, logger):
        """输出最终统计"""
        logger.info("\n" + "="*70)
        logger.info("三阶段标签矫正 - 最终结果汇总")
        logger.info("="*70)
        
        tier_counts = {}
        for t in tier_info:
            tier_counts[t] = tier_counts.get(t, 0) + 1
        
        tier_order = [
            'Tier 1: Core', 'Tier 2: Flip', 'Tier 3: Keep',
            'Tier 4: Reweight', 'Dropped'
        ]
        
        weight_map = {
            'Tier 1: Core': weights['TIER_1_CORE'],
            'Tier 2: Flip': weights['TIER_2_FLIP'],
            'Tier 3: Keep': weights['TIER_3_KEEP_HI'],
            'Tier 4: Reweight': 0.5,
            'Dropped': 0.0
        }
        
        logger.info("\n📊 分级统计:")
        for tier in tier_order:
            if tier in tier_counts:
                count = tier_counts[tier]
                weight = weight_map.get(tier, 0)
                logger.info(f"  {tier:30s}: {count:5d} ({100*count/n_samples:5.1f}%) | w={weight:.2f}")
        
        n_keep = (action_mask == 0).sum()
        n_flip = (action_mask == 1).sum()
        n_drop = (action_mask == 2).sum()
        n_rew = (action_mask == 3).sum()
        
        logger.info("\n📊 动作统计:")
        logger.info(f"  Keep:     {n_keep:5d} ({100*n_keep/n_samples:.1f}%)")
        logger.info(f"  Flip:     {n_flip:5d} ({100*n_flip/n_samples:.1f}%)")
        logger.info(f"  Drop:     {n_drop:5d} ({100*n_drop/n_samples:.1f}%)")
        logger.info(f"  Reweight: {n_rew:5d} ({100*n_rew/n_samples:.1f}%)")

    def _compute_tier_purity(self, clean_labels, y_true, tier_info, correction_weight, logger):
        """计算各Tier纯度"""
        logger.info("\n📊 各Tier纯度分析:")
        
        tier_stats = {}
        for i, tier in enumerate(tier_info):
            if tier not in tier_stats:
                tier_stats[tier] = {'count': 0, 'correct': 0, 'weight': correction_weight[i]}
            tier_stats[tier]['count'] += 1
            if clean_labels[i] == y_true[i]:
                tier_stats[tier]['correct'] += 1
        
        tier_order = [
            'Tier 1: Core', 'Tier 2: Flip', 'Tier 3: Keep',
            'Tier 4: Reweight', 'Dropped'
        ]
        
        role_map = {
            'Tier 1: Core': '[定海神针] 绝对纯净基石',
            'Tier 2: Flip': '[强力纠错] 高质量翻转',
            'Tier 3: Keep': '[难例精华] 优质保持',
            'Tier 4: Reweight': '[长尾数据] 降权使用',
            'Dropped': '[已丢弃] 无法拯救'
        }
        
        total_w_correct = 0
        total_w_count = 0
        
        logger.info("-" * 90)
        logger.info(f"{'Tier':<30s} | {'权重':>6s} | {'样本数':>8s} | {'纯度':>8s} | {'含噪数':>8s} | 角色")
        logger.info("-" * 90)

        for tier in tier_order:
            if tier in tier_stats:
                s = tier_stats[tier]
                purity = 100 * s['correct'] / s['count'] if s['count'] > 0 else 0
                noise = s['count'] - s['correct']
                logger.info(f"{tier:<30s} | {s['weight']:>6.2f} | {s['count']:>8d} | {purity:>7.1f}% | {noise:>8d} | {role_map.get(tier, '')}")
                
                if 'Dropped' not in tier:
                    total_w_correct += s['correct'] * s['weight']
                    total_w_count += s['count'] * s['weight']
        
        logger.info("-" * 90)
        if total_w_count > 0:
            logger.info(f"📈 加权纯度: {100 * total_w_correct / total_w_count:.2f}%")
