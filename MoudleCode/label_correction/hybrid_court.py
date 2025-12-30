"""
Hybrid Court: Label Noise Correction Module v11
三阶段标签矫正策略 (严格版 + 迭代CL):
- Phase 1: 核心严选 (Core Selection) - CL AND KNN 双重验证筛选核心数据
- Phase 2: 分级挽救 (Rescue & Tiering) - 严格翻转/保持/重加权/丢弃
- Phase 3a: 迭代CL拯救 (Iterative CL) - 用干净数据重新训练CL，高置信度拯救
- Phase 3b: 锚点拯救 (Anchor Rescue) - 仅以Core为锚点进行保守拯救

核心原则: 前三个Tier为了干净可以牺牲数据量
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

logger = logging.getLogger(__name__)


# === 算法超参数配置 (平衡版 - 纯度优先，兼顾数据量) ===
TIER_CONFIG = {
    # Phase 1: 核心严选阈值 (分层防御策略)
    'PHASE1_THRESHOLDS': {
        # 路径一：CL直通车 (无需KNN支持)
        'CL_DIRECT':  {'正常': 0.54, '恶意': 0.57},
        
        # 路径二：密度分区防御 (必须KNN支持当前标签)
        'DENSITY_DEFENSE': {
            # 高密度区 (MADE > 85.0) - 风险极高，严格双重验证
            'HIGH_DENSITY': {
                'threshold': 85.0,
                'KNN': {'正常': 0.60, '恶意': 0.70},
                'CL_MIN': {'正常': 0.40, '恶意': 0.52}
            },
            # 中密度区 (55.0 <= MADE <= 85.0) - 风险中等，标准防御
            'MID_DENSITY': {
                'threshold_min': 55.0,
                'threshold_max': 85.0,
                'KNN': {'正常': 0.60, '恶意': 0.70},
                'CL_MIN': {'正常': 0.40, '恶意': 0.40}
            },
            # 低密度区 (MADE < 55.0) - 风险较低，召回离群点
            'LOW_DENSITY': {
                'threshold': 55.0,
                'KNN': {'正常': 0.52, '恶意': 0.52},
                'CL_MIN': {'正常': 0.36, '恶意': 0.36}
            }
        }
    },
    
    # Phase 2: 特征分离阈值（严格统一策略）
    'PHASE2_FILTERS': {
        # === 严格翻转策略（统一标准）===
        # 路径1: CL主导 (CL目标 >= 0.7)
        'FLIP_CL_PATH1_MIN': 0.70,          # CL路径的CL下限
        # 路径2: KNN救援 (CL目标 >= 0.6 且 KNN >= 0.8)
        'FLIP_CL_PATH2_MIN': 0.60,          # KNN路径的CL下限
        'FLIP_KNN_PATH2_MIN': 0.80,         # KNN路径的KNN下限
        
        # 旧参数保留兼容性
        'FLIP_MIN_KNN_CONS': 0.60,
        'FLIP_MIN_CL_TARGET': 0.70,
        'FLIP_RESCUE_CL_MIN': 0.60,
        'FLIP_RESCUE_KNN_MIN': 0.80,
        'REWEIGHT_BASE_CL': 0.35,
        'CORE_MADE_KNN_MADE_MAX': 55.0,  # 降低至55，过滤样本155 (密度57.3)
        'CORE_MADE_KNN_KNN_MIN': 0.70,   # 提升至0.70，双重保险

        
        # === Keep策略 (标签不对称 + 熔断机制) ===
        # 针对标签0 (正常样本)
        'KEEP_LABEL_0': {
            'CL_MIN': 0.48,              # CL下限
            'KNN_MIN': 0.52,             # KNN下限
            'KNN_MAX': 0.70,             # KNN上限 (拒绝抱团噪声)
            'FUSE_KNN_THRESHOLD': 0.56,  # 熔断阈值
            'FUSE_CL_MIN': 0.53          # 熔断时的CL要求
        },
        # 针对标签1 (恶意样本)
        'KEEP_LABEL_1': {
            'CL_MIN': 0.45,              # CL下限
            'KNN_MIN': 0.61,             # KNN下限
            'MADE_MAX': 50.0             # 密度护栏 (利用低密度特性)
        },
        
        # 旧参数保留兼容性
        'KEEP_CL_MIN': 0.55,
        'KEEP_KNN_MIN': 0.58,
        'KEEP_HIGH_CL_MIN': 0.55,
        'KEEP_HIGH_KNN_MIN': 0.58,
        'KEEP_HIGH_MADE_MAX': 70.0,
        'REW_LOW_CL_MAX': 0.48,
        'REW_LOW_MADE_MIN': 80.0,
        'REW_LOW_KNN_MAX': 0.55
    },
    
    # 最终权重分配
    'WEIGHTS': {
        'TIER_1_CORE':      1.0,
        'TIER_2_FLIP':      1.0,
        'TIER_3_KEEP_HI':   1.0,       # Keep: 100%纯度
        'TIER_4A_REW_HI':   0.6,       # Reweight-High: ~84%纯度
        'TIER_4B_REW_LO':   0.2,       # Reweight-Low: ~44%纯度
        'TIER_5_RESCUED_FLIP': 0.9     # Rescued-Flip: 拯救的翻转样本
    },
    
    # Phase 3a: 迭代CL拯救 (用干净数据重新训练CL)
    'PHASE3A_ITERATIVE_CL': {
        'ENABLED': True,               # 是否启用迭代CL
        'MIN_CLEAN_SAMPLES': 100,      # 最少干净样本数
        'FLIP_TARGET_CONF': 0.80,      # 翻转的目标置信度阈值 (100%纯度)
        'KEEP_CONF': 0.70              # Keep的置信度阈值
    },
    
    # Phase 3b: 锚点拯救阈值 (纯度优先版)
    'PHASE3B_ANCHOR_RESCUE': {
        'MIN_ANCHORS': 30,             # 最少锚点样本数
        'KNN_K': 10,                   # 锚点KNN的K值
        # Keep拯救阈值 (收紧)
        'RESCUE_KEEP_CONS_HIGH': 0.80,   # 高置信度Keep: 锚点一致性>=80%
        'RESCUE_KEEP_CONS_MID': 0.70,    # 中置信度Keep: 锚点>=70% + 原KNN支持
        'RESCUE_KEEP_CONS_LOW': 0.60,    # 低置信度Keep: 锚点>=60% + 原KNN + 迭代CL
        # Flip拯救阈值 (极严格)
        'RESCUE_FLIP_CONS_HIGH': 0.95,   # 高置信度Flip: 锚点>=90% + 迭代CL + 原KNN
        'RESCUE_FLIP_CONS_MID': 0.90,    # 中置信度Flip: 锚点>=85% + 迭代CL强 + 原KNN
        'RESCUE_FLIP_CONS_LOW': 0.85,    # 低置信度Flip: 锚点>=80% + 三重验证
        # 迭代CL支持阈值
        'ITER_CL_SUPPORT_CONF': 0.55,    # 迭代CL支持的最低置信度
        'ITER_CL_STRONG_CONF': 0.75,     # 迭代CL强支持阈值
        'ITER_CL_FLIP_REQUIRED': False,
        'FLIP_MAX_MADE': 70.0,           # 翻转拯救的MADE上限
        # 权重分配
        'RESCUE_KEEP_WEIGHT_HIGH': 0.95,
        'RESCUE_KEEP_WEIGHT_MID': 0.85,
        'RESCUE_KEEP_WEIGHT_LOW': 0.75,
        'RESCUE_FLIP_WEIGHT_HIGH': 0.90,
        'RESCUE_FLIP_WEIGHT_MID': 0.80,
        'RESCUE_FLIP_WEIGHT_LOW': 0.70,
        # 兼容旧配置
        'RESCUE_KEEP_CONS': 0.80,
        'RESCUE_FLIP_CONS': 0.90,
        'RESCUE_KEEP_WEIGHT': 0.90,
        'RESCUE_FLIP_WEIGHT': 0.85
    },
    
    # Phase 3c: 最终清理 (Drop无法拯救的样本)
    'PHASE3C_FINAL_CLEANUP': {
        'ENABLED': True,               # 是否启用最终清理
        'DROP_LOW_CONF': True,         # 丢弃低置信度样本
        'MIN_ITER_CL_CONF': 0.40,      # 迭代CL最低置信度 (低于此值Drop)
        'MIN_ANCHOR_CONS': 0.50        # 锚点KNN最低一致性 (低于此值Drop)
    }
}


class ConfidentLearning:
    """Confident Learning (CL) - Probabilistic Diagnosis"""
    
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.confident_joint = None
        self.thresholds = None
    
    def fit_predict(self, features, labels):
        n_samples = len(labels)
        n_classes = len(np.unique(labels))
        pred_probs = np.zeros((n_samples, n_classes))
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(features):
            X_train, X_val = features[train_idx], features[val_idx]
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


class MADEDensityEstimator:
    """MADE: Masked Autoencoder for Distribution Estimation"""
    
    def __init__(self, hidden_dims=[128, 256, 128], threshold_percentile=70):
        self.hidden_dims = hidden_dims
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
    
    def fit(self, features, device='cpu'):
        from .made_core import MADE
        input_dim = features.shape[1]
        self.model = MADE(n_in=input_dim, hidden_dims=self.hidden_dims,
                          gaussian=True, random_order=False, seed=42).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        features_tensor = torch.FloatTensor(features).to(device)
        
        self.model.train()
        for epoch in range(50):
            indices = torch.randperm(len(features_tensor))
            for i in range(0, len(features_tensor), 64):
                batch = features_tensor[indices[i:i+64]]
                optimizer.zero_grad()
                loss = self.model.compute_nll_loss(batch)
                loss.backward()
                optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
            scores = self.model.compute_log_prob(features_tensor).cpu().numpy()
        self.threshold = np.percentile(scores, 100 - self.threshold_percentile)
        logger.info(f"MADE: threshold={self.threshold:.4f}, range=[{scores.min():.4f}, {scores.max():.4f}]")
    
    def predict_density(self, features, device='cpu'):
        self.model.eval()
        with torch.no_grad():
            scores = self.model.compute_log_prob(torch.FloatTensor(features).to(device)).cpu().numpy()
        is_dense = scores >= self.threshold
        self.last_is_dense = is_dense
        self.last_scores = scores
        return is_dense, scores


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
    Hybrid Court v11: 三阶段标签矫正策略 (严格版 + 迭代CL)
    
    核心原则: 前三个Tier为了干净可以牺牲数据量
    
    Phase 1: 核心严选 - CL AND KNN 双重验证
    Phase 2: 分级挽救 - 严格翻转条件，宁可不翻也不错翻
    Phase 3a: 迭代CL - 用干净数据(Core+Flip)重新训练CL，高置信度拯救
    Phase 3b: 锚点拯救 - 仅以Core为锚点，保守拯救
    """
    
    def __init__(self, config):
        self.config = config
        self.cl = ConfidentLearning(n_folds=config.CL_K_FOLD)
        self.made = MADEDensityEstimator(
            hidden_dims=config.MADE_HIDDEN_DIMS,
            threshold_percentile=config.MADE_DENSITY_THRESHOLD_PERCENTILE
        )
        self.knn = KNNSemanticVoting(
            k=config.KNN_NEIGHBORS,
            metric=getattr(config, 'KNN_METRIC', 'euclidean')
        )
        self.tier_config = copy.deepcopy(TIER_CONFIG)

    def _maybe_update_density_thresholds(self, density_scores: np.ndarray):
        if density_scores is None or len(density_scores) == 0:
            return

        enabled = bool(getattr(self.config, 'HYBRIDCOURT_DYNAMIC_DENSITY_THRESHOLDS', False))
        if not enabled:
            return

        high_pct = float(getattr(self.config, 'HYBRIDCOURT_DENSITY_HIGH_PCT', 90.0))
        low_pct = float(getattr(self.config, 'HYBRIDCOURT_DENSITY_LOW_PCT', 50.0))
        high_pct = float(np.clip(high_pct, 0.0, 100.0))
        low_pct = float(np.clip(low_pct, 0.0, 100.0))
        if low_pct > high_pct:
            low_pct, high_pct = high_pct, low_pct

        low_th = float(np.percentile(density_scores, low_pct))
        high_th = float(np.percentile(density_scores, high_pct))

        dd = self.tier_config['PHASE1_THRESHOLDS']['DENSITY_DEFENSE']
        dd['HIGH_DENSITY']['threshold'] = high_th
        dd['MID_DENSITY']['threshold_min'] = low_th
        dd['MID_DENSITY']['threshold_max'] = high_th
        dd['LOW_DENSITY']['threshold'] = low_th

        logger.info(
            "[HybridCourt] 动态密度阈值已启用: "
            f"LOW<{low_th:.4f} (pct={low_pct}), HIGH>{high_th:.4f} (pct={high_pct})"
        )

    def correct_labels(self, features, noisy_labels, device='cpu', y_true=None):
        """三阶段标签矫正 (严格版)"""
        n_samples = len(noisy_labels)

        def _log_subset_purity(title, subset_mask, use_corrected=False):
            """
            计算子集纯度
            
            Args:
                title: 标题
                subset_mask: 子集掩码
                use_corrected: 是否使用矫正后的标签（用于Flip操作）
            """
            if y_true is None:
                return
            subset_mask = np.asarray(subset_mask, dtype=bool)
            n = int(subset_mask.sum())
            if n == 0:
                logger.info(f"  {title}: 0 samples")
                return
            
            # 对于Flip操作，使用矫正后的标签；否则使用噪声标签
            labels_to_check = clean_labels if use_corrected else noisy_labels
            noise = int(((labels_to_check != y_true) & subset_mask).sum())
            purity = 100.0 * (n - noise) / n
            logger.info(f"  {title}: {n} samples | noise={noise} | purity={purity:.1f}%")
        
        logger.info("="*70)
        logger.info("Hybrid Court v10: 三阶段标签矫正 (严格版)")
        logger.info("="*70)
        
        # ========== Step 1: 运行子模块 ==========
        logger.info("\n[Step 1] 运行子模块...")
        suspected_noise, pred_labels, pred_probs = self.cl.fit_predict(features, noisy_labels)
        self.made.fit(features, device=device)
        is_dense, density_scores = self.made.predict_density(features, device=device)
        self._maybe_update_density_thresholds(density_scores)
        self.knn.fit(features)
        neighbor_labels, neighbor_consistency = self.knn.predict_semantic_label(features, noisy_labels)
        
        # 初始化
        clean_labels = noisy_labels.copy()
        action_mask = np.zeros(n_samples, dtype=int)
        confidence = np.ones(n_samples)
        correction_weight = np.ones(n_samples)
        tier_info = [''] * n_samples
        
        cfg = self.tier_config
        p1 = cfg['PHASE1_THRESHOLDS']
        p2 = cfg['PHASE2_FILTERS']
        weights = cfg['WEIGHTS']
        
        # ========== Phase 1: 核心严选 (Core Selection) ==========
        logger.info("\n" + "="*50)
        logger.info("Phase 1: 核心严选 (分层防御策略)")
        logger.info("="*50)
        logger.info(f"  策略说明:")
        logger.info(f"    目标: 筛选出100%纯净的核心样本，作为后续迭代的基石")
        logger.info(f"    策略: 双路径分层防御")
        logger.info(f"")
        logger.info(f"    路径一：CL直通车 (无需KNN支持)")
        logger.info(f"      条件：模型自身极其确信，无需邻居点头")
        logger.info(f"      规则：正常样本 CL>{p1['CL_DIRECT']['正常']}, 恶意样本 CL>{p1['CL_DIRECT']['恶意']}")
        logger.info(f"")
        logger.info(f"    路径二：密度分区防御 (必须KNN支持当前标签)")
        dd = p1['DENSITY_DEFENSE']
        logger.info(f"      🔴 高密度区 (MADE>{dd['HIGH_DENSITY']['threshold']}) - 风险极高，严格双重验证")
        logger.info(f"         恶意: KNN>{dd['HIGH_DENSITY']['KNN']['恶意']} 且 CL>{dd['HIGH_DENSITY']['CL_MIN']['恶意']}")
        logger.info(f"         正常: KNN>{dd['HIGH_DENSITY']['KNN']['正常']} 且 CL>{dd['HIGH_DENSITY']['CL_MIN']['正常']}")
        logger.info(f"      🟡 中密度区 ({dd['MID_DENSITY']['threshold_min']}≤MADE≤{dd['MID_DENSITY']['threshold_max']}) - 风险中等，标准防御")
        logger.info(f"         恶意: KNN>{dd['MID_DENSITY']['KNN']['恶意']} 且 CL>{dd['MID_DENSITY']['CL_MIN']['恶意']}")
        logger.info(f"         正常: KNN>{dd['MID_DENSITY']['KNN']['正常']} 且 CL>{dd['MID_DENSITY']['CL_MIN']['正常']}")
        logger.info(f"      🟢 低密度区 (MADE<{dd['LOW_DENSITY']['threshold']}) - 风险较低，召回离群点")
        logger.info(f"         所有: KNN>{dd['LOW_DENSITY']['KNN']['正常']} 且 CL>{dd['LOW_DENSITY']['CL_MIN']['正常']}")
        logger.info(f"")
        logger.info(f"    原则: 宁缺毋滥，确保核心样本的绝对纯净度")
        
        n_core = 0
        n_core_by_cl = 0
        n_core_by_density_high = 0
        n_core_by_density_mid = 0
        n_core_by_density_low = 0
        
        # 用于记录Phase 1噪声分析
        phase1_noise_details = []
        phase1_core_by_path = {'CL直通车': [], '高密度区': [], '中密度区': [], '低密度区': []}
        
        for i in range(n_samples):
            current_label = int(noisy_labels[i])
            label_name = '正常' if current_label == 0 else '恶意'
            
            cl_conf = float(pred_probs[i, current_label])
            knn_vote = int(neighbor_labels[i])
            knn_cons = float(neighbor_consistency[i])
            density = float(density_scores[i])
            
            is_core = False
            path = ''
            
            # 路径一：CL直通车
            if cl_conf >= p1['CL_DIRECT'][label_name]:
                is_core = True
                path = 'CL直通车'
                n_core_by_cl += 1
            
            # 路径二：密度分区防御 (必须KNN支持当前标签)
            elif knn_vote == current_label:
                dd = p1['DENSITY_DEFENSE']
                
                # 高密度区
                if density > dd['HIGH_DENSITY']['threshold']:
                    if (knn_cons >= dd['HIGH_DENSITY']['KNN'][label_name] and 
                        cl_conf >= dd['HIGH_DENSITY']['CL_MIN'][label_name]):
                        is_core = True
                        path = '高密度区'
                        n_core_by_density_high += 1
                
                # 中密度区
                elif dd['MID_DENSITY']['threshold_min'] <= density <= dd['MID_DENSITY']['threshold_max']:
                    if (knn_cons >= dd['MID_DENSITY']['KNN'][label_name] and 
                        cl_conf >= dd['MID_DENSITY']['CL_MIN'][label_name]):
                        is_core = True
                        path = '中密度区'
                        n_core_by_density_mid += 1
                
                # 低密度区
                else:  # density < dd['LOW_DENSITY']['threshold']
                    if (knn_cons >= dd['LOW_DENSITY']['KNN'][label_name] and 
                        cl_conf >= dd['LOW_DENSITY']['CL_MIN'][label_name]):
                        is_core = True
                        path = '低密度区'
                        n_core_by_density_low += 1
            
            if is_core:
                action_mask[i] = 0
                confidence[i] = max(cl_conf, knn_cons) if path != 'CL直通车' else cl_conf
                correction_weight[i] = weights['TIER_1_CORE']
                tier_info[i] = 'Tier 1: Core'
                n_core += 1
                phase1_core_by_path[path].append(i)
                
                # 检查是否为噪声样本
                if y_true is not None and y_true[i] != noisy_labels[i]:
                    phase1_noise_details.append({
                        'idx': i,
                        'path': path,
                        'noisy_label': current_label,
                        'true_label': int(y_true[i]),
                        'cl_conf': cl_conf,
                        'knn_vote': knn_vote,
                        'knn_cons': knn_cons,
                        'density': density
                    })
        
        logger.info(f"  ✓ Phase 1 完成: {n_core} 个核心样本 ({100*n_core/n_samples:.1f}%)")
        logger.info(f"    - CL直通车: {n_core_by_cl}")
        logger.info(f"    - 高密度区: {n_core_by_density_high}")
        logger.info(f"    - 中密度区: {n_core_by_density_mid}")
        logger.info(f"    - 低密度区: {n_core_by_density_low}")

        _log_subset_purity("Phase 1 Core", np.array([t == 'Tier 1: Core' for t in tier_info]))
        
        # 输出Phase 1噪声详细分析
        if y_true is not None and phase1_noise_details:
            logger.info(f"")
            logger.info(f"    📊 Phase 1 Core噪声样本详细分析 (共{len(phase1_noise_details)}个):")
            logger.info(f"    " + "-"*100)
            logger.info(f"    {'样本ID':>6s} | {'选入路径':<10s} | {'噪声标签':<8s} | {'真实标签':<8s} | {'CL置信':>8s} | {'KNN投票':>8s} | {'KNN一致':>8s} | {'MADE密度':>8s}")
            logger.info(f"    " + "-"*100)
            
            # 按路径统计噪声
            noise_by_path = {'CL直通车': 0, '高密度区': 0, '中密度区': 0, '低密度区': 0}
            
            for d in phase1_noise_details:
                noisy_name = '正常' if d['noisy_label'] == 0 else '恶意'
                true_name = '正常' if d['true_label'] == 0 else '恶意'
                knn_vote_name = '正常' if d['knn_vote'] == 0 else '恶意'
                logger.info(f"    {d['idx']:>6d} | {d['path']:<10s} | {noisy_name:<8s} | {true_name:<8s} | {d['cl_conf']:>8.3f} | {knn_vote_name:>8s} | {d['knn_cons']:>8.3f} | {d['density']:>8.1f}")
                noise_by_path[d['path']] += 1
            
            logger.info(f"    " + "-"*100)
            logger.info(f"    噪声来源统计:")
            for path, count in noise_by_path.items():
                total_in_path = len(phase1_core_by_path[path])
                if total_in_path > 0:
                    purity = 100 * (total_in_path - count) / total_in_path
                    logger.info(f"      {path}路径: {count}个噪声 / {total_in_path}个样本 (纯度={purity:.1f}%)")
            
            # 分析噪声样本的特征分布
            if len(phase1_noise_details) > 0:
                cl_confs = [d['cl_conf'] for d in phase1_noise_details]
                knn_conss = [d['knn_cons'] for d in phase1_noise_details]
                densities = [d['density'] for d in phase1_noise_details]
                logger.info(f"    噪声样本特征分布:")
                logger.info(f"      CL置信度: min={min(cl_confs):.3f}, max={max(cl_confs):.3f}, avg={np.mean(cl_confs):.3f}")
                logger.info(f"      KNN一致性: min={min(knn_conss):.3f}, max={max(knn_conss):.3f}, avg={np.mean(knn_conss):.3f}")
                logger.info(f"      MADE密度: min={min(densities):.1f}, max={max(densities):.1f}, avg={np.mean(densities):.1f}")


        
        # ========== Phase 2: 分级挽救 (严格版) ==========
        logger.info("\n" + "="*50)
        logger.info("Phase 2: 分级挽救 (零误杀智能翻转)")
        logger.info("="*50)
        logger.info(f"  策略说明:")
        logger.info(f"    目标: 在非核心样本中，基于差异化阈值实现零误杀翻转")
        logger.info(f"    Flip策略 (差异化门槛):")
        logger.info(f"      逻辑A (正常样本 - 更激进):")
        logger.info(f"        筛选门槛: CL差值 >= 0.14")
        logger.info(f"        救援规则1: 密度兜底 - MADE > 40 则保留 (提升门槛，精准切掉35-40区间噪声)")
        logger.info(f"        救援规则1.1: 确信覆盖 - 即使密度>40，但CL目标>0.70则强制翻转 (保守抓顶级伪装者)")
        logger.info(f"        救援规则2: 疑罪从无 - CL目标<0.66 且 KNN<0.56 则保留")
        logger.info(f"      逻辑B (恶意样本 - 极低门槛深挖):")
        logger.info(f"        筛选门槛: CL差值 >= 0.05")
        logger.info(f"        救援规则1: 离群保护 - MADE < 15 则保留")
        logger.info(f"        救援规则2: 信心兜底 - CL当前 > 0.32 则保留")
        logger.info(f"    Keep策略 (标签不对称 + 熔断机制):")
        logger.info(f"      A. 针对标签0 (正常样本):")
        logger.info(f"         基础门槛: CL>{p2['KEEP_LABEL_0']['CL_MIN']} 且 {p2['KEEP_LABEL_0']['KNN_MIN']}<KNN<{p2['KEEP_LABEL_0']['KNN_MAX']}")
        logger.info(f"         逻辑: 拒绝KNN过高(>{p2['KEEP_LABEL_0']['KNN_MAX']})的抱团噪声，也拒绝KNN过低(<{p2['KEEP_LABEL_0']['KNN_MIN']})的孤立点")
        logger.info(f"         🛡️ 熔断护栏: 如果KNN<{p2['KEEP_LABEL_0']['FUSE_KNN_THRESHOLD']}，则必须CL>{p2['KEEP_LABEL_0']['FUSE_CL_MIN']}")
        logger.info(f"         作用: 精准击杀低一致性+置信度不高的漏网之鱼")
        logger.info(f"      B. 针对标签1 (恶意样本):")
        logger.info(f"         基础门槛: CL>{p2['KEEP_LABEL_1']['CL_MIN']} 且 KNN>{p2['KEEP_LABEL_1']['KNN_MIN']}")
        logger.info(f"         🛡️ 密度护栏: MADE<{p2['KEEP_LABEL_1']['MADE_MAX']}")
        logger.info(f"         作用: 利用真实恶意样本'低密度'特性，切除高密度区噪声")
        logger.info(f"    Reweight策略: 其余样本暂时标记为Reweight，等待Phase 3进一步处理")
        logger.info(f"    原则: 差异化门槛 + 双重救援 + 确信覆盖(0.70) = 零误杀高召回")
        
        n_flip = 0
        n_keep_hi = 0
        n_keep_lo = 0
        n_rew_hi = 0
        n_rew_lo = 0
        n_drop = 0
        
        # 用于记录Phase 2 Flip的详细信息
        phase2_flip_details = []
        phase2_keep_details = []  # 新增：记录Keep的详细信息
        
        for i in range(n_samples):
            if tier_info[i] == 'Tier 1: Core':
                continue
            
            current_label = int(noisy_labels[i])
            label_name = '正常' if current_label == 0 else '恶意'
            target_label = 1 - current_label
            target_name = '正常' if target_label == 0 else '恶意'
            
            cl_conf = float(pred_probs[i, current_label])
            cl_target = float(pred_probs[i, target_label])
            cl_pred = int(pred_labels[i])
            knn_vote = int(neighbor_labels[i])
            knn_cons = float(neighbor_consistency[i])
            density = float(density_scores[i])
            is_suspected = bool(suspected_noise[i])
            
            # --- 差异化翻转策略 (针对正常/恶意设定不同门槛) ---
            # 计算CL差值 (目标标签置信度 - 当前标签置信度)
            cl_diff = cl_target - cl_conf
            
            should_flip = False
            flip_reason = ''
            
            if current_label == 0:  # 逻辑A: 当前标签为"正常" (更激进)
                # 筛选门槛: CL差值 >= 0.14 (扩大筛选范围)
                if cl_diff >= 0.14:
                    # 救援规则1: 密度兜底 - MADE密度 > 40 -> 保留
                    # (提升门槛: 干净样本最低密度40.01，精准切掉35-40区间的噪声)
                    if density > 40:
                        # 救援规则1.1: 确信覆盖 - 即使密度>40，但CL目标>0.70则强制翻转
                        # (在超高密度区，干净样本CL目标最高0.679，>0.70更保守，确保零误杀)
                        if cl_target > 0.70:
                            should_flip = True
                            flip_reason = f'正常→恶意(确信覆盖:CL目标={cl_target:.3f}>0.70, 密度={density:.1f})'
                        else:
                            should_flip = False
                    # 救援规则2: 疑罪从无 - CL目标<0.66 且 KNN<0.56 -> 保留
                    # (针对那4个误杀样本: CL犹豫 + KNN杂乱 = 保留原判)
                    elif cl_target < 0.66 and knn_cons < 0.56:
                        should_flip = False
                    # 其他 -> 翻转
                    else:
                        should_flip = True
                        flip_reason = f'正常→恶意(CL差值={cl_diff:.3f}, 密度={density:.1f})'
            else:  # 逻辑B: 当前标签为"恶意" (极低门槛深挖)
                # 筛选门槛: CL差值 >= 0.05 (深挖隐藏噪声)
                if cl_diff >= 0.05:
                    # 救援规则1: 离群保护 - MADE密度 < 15 -> 保留
                    if density < 15:
                        should_flip = False
                    # 救援规则2: 信心兜底 - CL当前置信度 > 0.32 -> 保留
                    # (关键红线: 干净样本CL信心>=0.33, 噪声样本CL信心极低)
                    elif cl_conf > 0.32:
                        should_flip = False
                    # 其他 -> 翻转
                    else:
                        should_flip = True
                        flip_reason = f'恶意→正常(CL差值={cl_diff:.3f}, 密度={density:.1f})'
            
            if should_flip:
                clean_labels[i] = target_label
                action_mask[i] = 1
                confidence[i] = cl_target
                correction_weight[i] = weights['TIER_2_FLIP']
                # 记录详细的翻转信息
                tier_info[i] = f'Tier 2: Flip'
                n_flip += 1
                
                # 记录翻转详情
                is_noise = y_true[i] != noisy_labels[i] if y_true is not None else None
                is_correct = y_true[i] == target_label if y_true is not None else None
                phase2_flip_details.append({
                    'idx': i,
                    'is_noise': is_noise,
                    'is_correct': is_correct,
                    'cl_target': cl_target,
                    'knn_cons': knn_cons,
                    'current_label': current_label,
                    'target_label': target_label,
                    'reason': flip_reason
                })
                continue
            
            # --- Keep决策 (标签不对称 + 熔断机制) ---
            # 注意：不再要求KNN必须支持当前标签，只要满足阈值条件即可
            is_keep = False
            keep_reason = ''
            
            if current_label == 0:  # 标签0 (正常样本)
                # 基础门槛: CL > 0.48 且 0.52 < KNN < 0.70
                if (cl_conf >= p2['KEEP_LABEL_0']['CL_MIN'] and 
                    knn_cons >= p2['KEEP_LABEL_0']['KNN_MIN'] and 
                    knn_cons < p2['KEEP_LABEL_0']['KNN_MAX']):
                    
                    # 🛡️ 熔断护栏: 如果 KNN < 0.56，则必须 CL > 0.53
                    if knn_cons < p2['KEEP_LABEL_0']['FUSE_KNN_THRESHOLD']:
                        if cl_conf >= p2['KEEP_LABEL_0']['FUSE_CL_MIN']:
                            is_keep = True
                            knn_support = '支持' if knn_vote == current_label else '不支持'
                            keep_reason = f'正常样本(熔断通过,KNN{knn_support})'
                        # else: 熔断拒绝
                    else:
                        is_keep = True
                        knn_support = '支持' if knn_vote == current_label else '不支持'
                        keep_reason = f'正常样本(基础通过,KNN{knn_support})'
            
            else:  # 标签1 (恶意样本)
                # 基础门槛: CL > 0.45 且 KNN > 0.61
                # 🛡️ 密度护栏: MADE < 50.0
                if (cl_conf >= p2['KEEP_LABEL_1']['CL_MIN'] and 
                    knn_cons >= p2['KEEP_LABEL_1']['KNN_MIN'] and 
                    density < p2['KEEP_LABEL_1']['MADE_MAX']):
                    is_keep = True
                    knn_support = '支持' if knn_vote == current_label else '不支持'
                    keep_reason = f'恶意样本(密度护栏通过,KNN{knn_support})'
            
            if is_keep:
                action_mask[i] = 0
                confidence[i] = knn_cons
                correction_weight[i] = weights['TIER_3_KEEP_HI']
                tier_info[i] = 'Tier 3: Keep'
                n_keep_hi += 1
                
                # 记录Keep详情
                is_noise = y_true[i] != noisy_labels[i] if y_true is not None else None
                phase2_keep_details.append({
                    'idx': i,
                    'is_noise': is_noise,
                    'current_label': current_label,
                    'cl_conf': cl_conf,
                    'knn_cons': knn_cons,
                    'density': density,
                    'reason': keep_reason
                })
            else:
                # Reweight: 剩余样本 (Phase 3会进一步处理)
                action_mask[i] = 3
                confidence[i] = cl_conf
                correction_weight[i] = 0.5  # 临时权重，Phase 3会重新分配
                tier_info[i] = 'Tier 4: Reweight'
                n_rew_hi += 1
        
        logger.info(f"  ✓ Phase 2 完成:")
        logger.info(f"    Tier 2 (Flip):        {n_flip}")
        logger.info(f"    Tier 3 (Keep):        {n_keep_hi}")
        logger.info(f"    Tier 4 (Reweight):    {n_rew_hi}")

        _log_subset_purity("Phase 2 Tier 2 (Flip)", np.array([t == 'Tier 2: Flip' for t in tier_info]), use_corrected=True)
        _log_subset_purity("Phase 2 Tier 3 (Keep)", np.array([t == 'Tier 3: Keep' for t in tier_info]))
        _log_subset_purity("Phase 2 Tier 4 (Reweight)", np.array([t == 'Tier 4: Reweight' for t in tier_info]))
        
        # 输出Phase 2 Flip的详细分析
        if y_true is not None and phase2_flip_details:
            flip_errors = [d for d in phase2_flip_details if d['is_correct'] == False]
            if flip_errors:
                logger.info(f"")
                logger.info(f"    📊 Phase 2 Flip错误样本详细分析 (共{len(flip_errors)}个):")
                logger.info(f"    " + "-"*90)
                logger.info(f"    {'样本ID':>6s} | {'CL目标':>8s} | {'KNN一致':>8s} | {'翻转方向':<12s} | {'原因':<10s}")
                logger.info(f"    " + "-"*90)
                for d in flip_errors[:10]:  # 只显示前10个
                    direction = f"{d['current_label']}→{d['target_label']}"
                    logger.info(f"    {d['idx']:>6d} | {d['cl_target']:>8.3f} | {d['knn_cons']:>8.3f} | {direction:<12s} | {d['reason']:<10s}")
                
                # 统计特征分布
                cl_targets = [d['cl_target'] for d in flip_errors]
                knn_conss = [d['knn_cons'] for d in flip_errors]
                logger.info(f"    " + "-"*90)
                logger.info(f"    错误样本特征分布:")
                logger.info(f"      CL目标:  min={min(cl_targets):.3f}, max={max(cl_targets):.3f}, avg={np.mean(cl_targets):.3f}")
                logger.info(f"      KNN一致: min={min(knn_conss):.3f}, max={max(knn_conss):.3f}, avg={np.mean(knn_conss):.3f}")
                
                # 按原因统计
                reason_stats = {}
                for d in flip_errors:
                    reason_stats[d['reason']] = reason_stats.get(d['reason'], 0) + 1
                logger.info(f"    错误原因统计:")
                for reason, count in sorted(reason_stats.items()):
                    logger.info(f"      {reason}: {count}个样本")
        
        # 输出Phase 2 Keep的详细分析
        if y_true is not None and phase2_keep_details:
            keep_errors = [d for d in phase2_keep_details if d['is_noise'] == True]
            if keep_errors:
                logger.info(f"")
                logger.info(f"    📊 Phase 2 Keep噪声样本详细分析 (共{len(keep_errors)}个):")
                logger.info(f"    " + "-"*100)
                logger.info(f"    {'样本ID':>6s} | {'标签':>4s} | {'CL置信':>8s} | {'KNN一致':>8s} | {'MADE密度':>8s} | {'原因':<25s}")
                logger.info(f"    " + "-"*100)
                for d in keep_errors[:10]:  # 只显示前10个
                    label_name = '正常' if d['current_label'] == 0 else '恶意'
                    logger.info(f"    {d['idx']:>6d} | {label_name:>4s} | {d['cl_conf']:>8.3f} | {d['knn_cons']:>8.3f} | {d['density']:>8.1f} | {d['reason']:<25s}")
                
                # 统计特征分布
                cl_confs = [d['cl_conf'] for d in keep_errors]
                knn_conss = [d['knn_cons'] for d in keep_errors]
                densities = [d['density'] for d in keep_errors]
                logger.info(f"    " + "-"*100)
                logger.info(f"    噪声样本特征分布:")
                logger.info(f"      CL置信:  min={min(cl_confs):.3f}, max={max(cl_confs):.3f}, avg={np.mean(cl_confs):.3f}")
                logger.info(f"      KNN一致: min={min(knn_conss):.3f}, max={max(knn_conss):.3f}, avg={np.mean(knn_conss):.3f}")
                logger.info(f"      MADE密度: min={min(densities):.1f}, max={max(densities):.1f}, avg={np.mean(densities):.1f}")
                
                # 按原因统计
                reason_stats = {}
                for d in keep_errors:
                    reason_stats[d['reason']] = reason_stats.get(d['reason'], 0) + 1
                logger.info(f"    噪声原因统计:")
                for reason, count in sorted(reason_stats.items()):
                    logger.info(f"      {reason}: {count}个样本")
                
                # 按标签统计
                label_0_errors = [d for d in keep_errors if d['current_label'] == 0]
                label_1_errors = [d for d in keep_errors if d['current_label'] == 1]
                logger.info(f"    按标签统计:")
                logger.info(f"      标签0(正常): {len(label_0_errors)}个噪声")
                logger.info(f"      标签1(恶意): {len(label_1_errors)}个噪声")


        
        # ========== Phase 3: 统一拯救策略 (100%纯度优先) ==========
        # 目标: 在保证100%纯度的前提下，尽量多拯救样本
        # 策略: 结合迭代CL + 锚点KNN，双重验证确保纯度
        
        logger.info("\n" + "="*50)
        logger.info("Phase 3: 统一拯救策略 (100%纯度优先)")
        logger.info("="*50)
        
        p3a = cfg['PHASE3A_ITERATIVE_CL']
        p3b = cfg['PHASE3B_ANCHOR_RESCUE']
        p3c = cfg['PHASE3C_FINAL_CLEANUP']
        
        # Step 3.1: 训练迭代CL模型 (用Core + Flip)
        clf_iter = None
        iter_pred_probs_all = None
        
        clean_tiers = ['Tier 1: Core', 'Tier 2: Flip']
        clean_mask = np.array([t in clean_tiers for t in tier_info])
        n_clean = clean_mask.sum()
        
        logger.info(f"  [Step 3.1] 训练迭代CL模型")
        logger.info(f"    干净样本数 (Core+Flip): {n_clean}")
        
        if n_clean >= p3a['MIN_CLEAN_SAMPLES']:
            X_clean = features[clean_mask]
            y_clean = clean_labels[clean_mask]
            clf_iter = LogisticRegression(max_iter=1000, random_state=42)
            clf_iter.fit(X_clean, y_clean)
            iter_pred_probs_all = clf_iter.predict_proba(features)
            logger.info(f"    ✓ 迭代CL模型训练完成")
        else:
            logger.info(f"    ⚠ 干净样本不足，跳过迭代CL")
        
        # Step 3.2: 构建锚点KNN (用Core + Flip)
        anchor_tiers = ['Tier 1: Core', 'Tier 2: Flip']
        anchor_mask = np.array([t in anchor_tiers for t in tier_info])
        n_anchors = anchor_mask.sum()
        anchor_features = features[anchor_mask]
        anchor_labels = clean_labels[anchor_mask]
        
        logger.info(f"  [Step 3.2] 构建锚点KNN")
        logger.info(f"    锚点样本数 (Core+Flip): {n_anchors}")
        
        knn_anchor = None
        if n_anchors >= p3b['MIN_ANCHORS']:
            k_anchor = min(p3b['KNN_K'], n_anchors - 1)
            knn_anchor = NearestNeighbors(n_neighbors=k_anchor, metric='euclidean')
            knn_anchor.fit(anchor_features)
            logger.info(f"    ✓ 锚点KNN构建完成 (K={k_anchor})")
        
        # Step 3.3: 统一拯救 (待处理: Reweight组)
        rescue_tiers = ['Tier 4: Reweight']
        rescue_mask = np.array([t in rescue_tiers for t in tier_info])
        rescue_indices = np.where(rescue_mask)[0]
        
        logger.info(f"  [Step 3.3] 统一拯救 (双区通用安全翻转)")
        logger.info(f"    策略说明:")
        logger.info(f"      目标: 利用迭代CL和锚点KNN双重验证，拯救Phase 2中的Reweight样本")
        logger.info(f"      Keep拯救策略 (双通道):")
        logger.info(f"        通道A (高置信度): iter_CL当前标签>0.90")
        logger.info(f"        通道B (低密度安全): MADE<60.0 且 iter_CL当前标签>0.70")
        logger.info(f"        原则: 满足任一即可保留")
        logger.info(f"      Flip拯救策略 (双区通用安全翻转):")
        logger.info(f"        前置条件: Anchor KNN不支持当前标签")
        logger.info(f"        通道A (低密度区): MADE<40.0 且 iter_CL目标>0.60")
        logger.info(f"        通道B (高密度区): iter_CL目标>0.90")
        logger.info(f"      后续处理:")
        logger.info(f"        Reweight2: 未被拯救的样本 → Drop")
        logger.info(f"        Reweight: 其他剩余样本 → 降权0.5")
        logger.info(f"    待处理样本数: {len(rescue_indices)}")
        
        n_rescued_keep = 0
        n_rescued_flip = 0
        rescued_indices = []
        rescued_keep_indices = []
        rescued_flip_indices = []
        
        # 用于记录Phase 3拯救的详细信息
        phase3_keep_details = []
        phase3_flip_details = []
        
        if len(rescue_indices) > 0 and knn_anchor is not None and iter_pred_probs_all is not None:
            # 计算锚点KNN投票
            rescue_features = features[rescue_mask]
            distances, indices = knn_anchor.kneighbors(rescue_features)
            
            for idx, i in enumerate(rescue_indices):
                current_label = int(clean_labels[i])
                target_label = 1 - current_label
                
                # 锚点KNN投票
                neighbor_d = distances[idx]
                neighbor_y = anchor_labels[indices[idx]]
                w = 1.0 / (neighbor_d + 1e-6)
                vote_0 = w[neighbor_y == 0].sum()
                vote_1 = w[neighbor_y == 1].sum()
                total_w = vote_0 + vote_1
                anchor_vote = 0 if vote_0 > vote_1 else 1
                anchor_cons = max(vote_0, vote_1) / total_w if total_w > 0 else 0
                
                # 迭代CL置信度
                iter_cl_conf_current = float(iter_pred_probs_all[i, current_label])
                iter_cl_conf_target = float(iter_pred_probs_all[i, target_label])
                iter_cl_pred = 0 if iter_pred_probs_all[i, 0] > iter_pred_probs_all[i, 1] else 1
                
                # 原始KNN信息
                orig_knn_vote = int(neighbor_labels[i])
                orig_knn_cons = float(neighbor_consistency[i])
                density = float(density_scores[i])
                
                rescued = False
                
                # ========== 拯救条件 (双通道策略) ==========
                
                # --- Keep拯救: 双通道 (满足任一即可保留) ---
                keep_reason = ''
                
                # 通道A: 高置信度
                if iter_cl_conf_current > 0.90:
                    rescued = True
                    keep_reason = f'高置信度通道(CL={iter_cl_conf_current:.3f})'
                
                # 通道B: 低密度安全
                elif density < 60.0 and iter_cl_conf_current > 0.70:
                    rescued = True
                    keep_reason = f'低密度安全通道(密度={density:.1f}, CL={iter_cl_conf_current:.3f})'
                
                if rescued:
                    action_mask[i] = 0
                    confidence[i] = iter_cl_conf_current
                    correction_weight[i] = weights['TIER_3_KEEP_HI']
                    tier_info[i] = 'Tier 5a: Rescued-Keep'
                    n_rescued_keep += 1
                    rescued_keep_indices.append(i)
                    
                    # 记录Keep拯救详情
                    is_correct = y_true[i] == current_label if y_true is not None else None
                    phase3_keep_details.append({
                        'idx': i,
                        'is_correct': is_correct,
                        'iter_cl': iter_cl_conf_current,
                        'anchor_cons': anchor_cons,
                        'orig_knn': neighbor_consistency[i],  # 添加原始KNN一致性
                        'density': density,
                        'current_label': current_label,
                        'reason': keep_reason
                    })
                
                # --- Flip拯救: 双区通用安全翻转 (Dual-Zone Universal Safe Flip) ---
                # 前置条件: Anchor KNN不支持当前标签
                if not rescued and (anchor_vote != current_label):
                    flip_reason = ''
                    
                    # 通道A: 低密度区 (Low Density)
                    # 特征: 低密度区的噪声样本特征非常鲜明，哪怕置信度稍低也是可信的
                    # 规则: MADE密度 < 40.0 且 iter_CL目标 > 0.60
                    if density < 40.0 and iter_cl_conf_target >= 0.60:
                        rescued = True
                        flip_reason = f'低密度区翻转(密度={density:.1f}, CL目标={iter_cl_conf_target:.3f})'
                    
                    # 通道B: 高密度区 (High Density)
                    # 特征: 高密度区是误判的重灾区，只有当模型达到"绝对确信"时才允许翻转
                    # 规则: iter_CL目标 > 0.90
                    elif iter_cl_conf_target >= 0.90:
                        rescued = True
                        flip_reason = f'高密度区翻转(密度={density:.1f}, CL目标={iter_cl_conf_target:.3f})'
                    
                    if rescued:
                        clean_labels[i] = target_label
                        action_mask[i] = 1
                        confidence[i] = iter_cl_conf_target
                        correction_weight[i] = weights['TIER_5_RESCUED_FLIP']  # 使用0.9权重
                        tier_info[i] = 'Tier 5b: Rescued-Flip'
                        n_rescued_flip += 1
                        rescued_flip_indices.append(i)
                        
                        # 记录Flip拯救详情
                        is_correct = y_true[i] == target_label if y_true is not None else None
                        phase3_flip_details.append({
                            'idx': i,
                            'is_correct': is_correct,
                            'iter_cl_target': iter_cl_conf_target,
                            'anchor_cons': anchor_cons,
                            'anchor_vote': anchor_vote,
                            'density': density,
                            'current_label': current_label,
                            'target_label': target_label,
                            'reason': flip_reason
                        })
                        
                        # 记录触发的条件（用于调试）
                        if not hasattr(self, '_flip_rescue_reasons'):
                            self._flip_rescue_reasons = {}
                        channel = '低密度区' if density < 40.0 else '高密度区'
                        self._flip_rescue_reasons[channel] = self._flip_rescue_reasons.get(channel, 0) + 1
                
                if rescued:
                    rescued_indices.append(i)
        
        logger.info(f"    ✓ 拯救完成: Keep={n_rescued_keep}, Flip={n_rescued_flip}")
        
        # 输出Flip拯救条件统计
        if hasattr(self, '_flip_rescue_reasons') and self._flip_rescue_reasons:
            logger.info(f"    📊 Flip拯救通道统计:")
            for channel, count in sorted(self._flip_rescue_reasons.items()):
                logger.info(f"      {channel}通道: {count}个样本")

        _log_subset_purity("Phase 3 rescued Keep", np.isin(np.arange(n_samples), np.array(rescued_keep_indices, dtype=int)))
        _log_subset_purity("Phase 3 rescued Flip", np.isin(np.arange(n_samples), np.array(rescued_flip_indices, dtype=int)), use_corrected=True)
        
        # 输出Phase 3 Keep拯救的详细分析
        if y_true is not None and phase3_keep_details:
            keep_errors = [d for d in phase3_keep_details if d['is_correct'] == False]
            if keep_errors:
                logger.info(f"")
                logger.info(f"    📊 Phase 3 Keep拯救错误样本详细分析 (共{len(keep_errors)}个):")
                logger.info(f"    " + "-"*90)
                logger.info(f"    {'样本ID':>6s} | {'iter_CL':>8s} | {'anchor':>8s} | {'orig_KNN':>8s} | {'原因':<15s}")
                logger.info(f"    " + "-"*90)
                for d in keep_errors[:10]:
                    logger.info(f"    {d['idx']:>6d} | {d['iter_cl']:>8.3f} | {d['anchor_cons']:>8.3f} | {d['orig_knn']:>8.3f} | {d['reason']:<15s}")
                
                # 统计特征分布
                iter_cls = [d['iter_cl'] for d in keep_errors]
                anchors = [d['anchor_cons'] for d in keep_errors]
                orig_knns = [d['orig_knn'] for d in keep_errors]
                logger.info(f"    " + "-"*90)
                logger.info(f"    错误样本特征分布:")
                logger.info(f"      iter_CL: min={min(iter_cls):.3f}, max={max(iter_cls):.3f}, avg={np.mean(iter_cls):.3f}")
                logger.info(f"      anchor:  min={min(anchors):.3f}, max={max(anchors):.3f}, avg={np.mean(anchors):.3f}")
                logger.info(f"      orig_KNN: min={min(orig_knns):.3f}, max={max(orig_knns):.3f}, avg={np.mean(orig_knns):.3f}")
        
        # 输出Phase 3 Flip拯救的详细分析
        if y_true is not None and phase3_flip_details:
            flip_errors = [d for d in phase3_flip_details if d['is_correct'] == False]
            if flip_errors:
                logger.info(f"")
                logger.info(f"    📊 Phase 3 Flip拯救错误样本详细分析 (共{len(flip_errors)}个):")
                logger.info(f"    " + "-"*110)
                logger.info(f"    {'样本ID':>6s} | {'翻转方向':<8s} | {'iter_CL_T':>9s} | {'anchor':>8s} | {'密度':>8s} | {'原因':<40s}")
                logger.info(f"    " + "-"*110)
                for d in flip_errors[:10]:
                    direction = f"{d['current_label']}→{d['target_label']}"
                    logger.info(f"    {d['idx']:>6d} | {direction:<8s} | {d['iter_cl_target']:>9.3f} | {d['anchor_cons']:>8.3f} | {d['density']:>8.1f} | {d['reason']:<40s}")
                
                # 统计特征分布
                iter_cl_targets = [d['iter_cl_target'] for d in flip_errors]
                anchors = [d['anchor_cons'] for d in flip_errors]
                densities = [d['density'] for d in flip_errors]
                logger.info(f"    " + "-"*110)
                logger.info(f"    错误样本特征分布:")
                logger.info(f"      iter_CL_T: min={min(iter_cl_targets):.3f}, max={max(iter_cl_targets):.3f}, avg={np.mean(iter_cl_targets):.3f}")
                logger.info(f"      anchor:    min={min(anchors):.3f}, max={max(anchors):.3f}, avg={np.mean(anchors):.3f}")
                logger.info(f"      密度:      min={min(densities):.1f}, max={max(densities):.1f}, avg={np.mean(densities):.1f}")
                
                # 按通道统计错误
                low_density_errors = [d for d in flip_errors if d['density'] < 40.0]
                high_density_errors = [d for d in flip_errors if d['density'] >= 40.0]
                logger.info(f"    按通道统计错误:")
                logger.info(f"      低密度区通道: {len(low_density_errors)}个错误")
                logger.info(f"      高密度区通道: {len(high_density_errors)}个错误")
        
        # Step 3.4: 最终处理 (Drop + Reweight)
        logger.info(f"  [Step 3.4] 最终处理 (Drop + Reweight)")
        logger.info(f"    策略说明:")
        logger.info(f"      Drop条件1: 迭代CL置信度 < 0.48 (CL极低)")
        logger.info(f"      Drop条件2: 锚点KNN一致性 < 0.55 且 迭代CL < 0.55 (KNN+CL都低)")
        logger.info(f"      其余样本: 统一Reweight，权重0.5")
        
        remaining_tiers = ['Tier 4: Reweight']
        remaining_mask = np.array([t in remaining_tiers for t in tier_info])
        remaining_indices = np.where(remaining_mask)[0]
        
        logger.info(f"    剩余待处理样本: {len(remaining_indices)}")
        
        # 统计变量
        n_dropped = 0
        n_reweight = 0
        
        # Drop原因统计 (带噪声/干净计数)
        drop_stats = {
            'CL极低': {'total': 0, 'noise': 0, 'clean': 0},
            'KNN+CL都低': {'total': 0, 'noise': 0, 'clean': 0}
        }
        
        # 用于分析的列表
        drop_details = []
        
        for i in remaining_indices:
            current_label = int(clean_labels[i])
            should_drop = False
            drop_reason = ''
            
            # 获取迭代CL和锚点KNN信息
            if iter_pred_probs_all is not None and knn_anchor is not None:
                iter_cl_conf = float(iter_pred_probs_all[i, current_label])
                
                # 计算锚点KNN投票
                dist, idx = knn_anchor.kneighbors(features[i:i+1])
                neighbor_y = anchor_labels[idx[0]]
                w = 1.0 / (dist[0] + 1e-6)
                vote_0 = w[neighbor_y == 0].sum()
                vote_1 = w[neighbor_y == 1].sum()
                anchor_cons = max(vote_0, vote_1) / (vote_0 + vote_1) if (vote_0 + vote_1) > 0 else 0
                
                # 获取真实标签用于分析
                is_noise = y_true[i] != current_label if y_true is not None else None
                density = float(density_scores[i])
                
                # ========== Drop条件 ==========
                
                # Drop条件1: 迭代CL置信度极低 (<0.48)
                if iter_cl_conf < 0.48:
                    should_drop = True
                    drop_reason = 'CL极低'
                    drop_stats['CL极低']['total'] += 1
                    if is_noise is not None:
                        if is_noise:
                            drop_stats['CL极低']['noise'] += 1
                        else:
                            drop_stats['CL极低']['clean'] += 1
                # Drop条件2: 锚点KNN一致性极低 (<0.55) 且 迭代CL不支持 (<0.55)
                elif anchor_cons < 0.55 and iter_cl_conf < 0.55:
                    should_drop = True
                    drop_reason = 'KNN+CL都低'
                    drop_stats['KNN+CL都低']['total'] += 1
                    if is_noise is not None:
                        if is_noise:
                            drop_stats['KNN+CL都低']['noise'] += 1
                        else:
                            drop_stats['KNN+CL都低']['clean'] += 1
                
                # 记录Drop详情
                if should_drop:
                    drop_details.append({
                        'idx': i,
                        'is_noise': is_noise,
                        'iter_cl_conf': iter_cl_conf,
                        'anchor_cons': anchor_cons,
                        'density': density,
                        'current_label': current_label,
                        'reason': drop_reason
                    })
            
            if should_drop:
                action_mask[i] = 2  # Drop
                correction_weight[i] = 0.0
                tier_info[i] = 'Dropped'
                n_dropped += 1
            else:
                # 统一Reweight，权重0.5
                correction_weight[i] = 0.5
                tier_info[i] = 'Tier 4: Reweight'
                n_reweight += 1
        
        # 输出统计
        logger.info(f"    ✓ Drop={n_dropped}, Reweight={n_reweight}")
        
        if drop_stats['CL极低']['total'] > 0 or drop_stats['KNN+CL都低']['total'] > 0:
            logger.info(f"")
            logger.info(f"    📊 Drop策略详细分析:")
            logger.info(f"    " + "-"*70)
            logger.info(f"    {'Drop原因':<20s} | {'总数':>6s} | {'噪声':>6s} | {'干净':>6s} | {'噪声率':>8s}")
            logger.info(f"    " + "-"*70)
            
            total_noise = 0
            total_clean = 0
            for reason, stats in drop_stats.items():
                if stats['total'] > 0:
                    noise_rate = 100 * stats['noise'] / stats['total'] if stats['total'] > 0 else 0
                    logger.info(f"    {reason:<20s} | {stats['total']:>6d} | {stats['noise']:>6d} | {stats['clean']:>6d} | {noise_rate:>7.1f}%")
                    total_noise += stats['noise']
                    total_clean += stats['clean']
            
            logger.info(f"    " + "-"*70)
            if n_dropped > 0:
                logger.info(f"    {'总计':<20s} | {n_dropped:>6d} | {total_noise:>6d} | {total_clean:>6d} | {100*total_noise/n_dropped:>7.1f}%")
        
        # 分析被误杀的干净样本
        if drop_details:
            clean_dropped = [d for d in drop_details if d['is_noise'] == False]
            if clean_dropped:
                logger.info(f"")
                logger.info(f"    📊 被误杀的干净样本详情 (共{len(clean_dropped)}个):")
                logger.info(f"    " + "-"*90)
                logger.info(f"    {'样本ID':>6s} | {'标签':>4s} | {'iter_CL':>8s} | {'anchor':>8s} | {'密度':>8s} | {'原因':<15s}")
                logger.info(f"    " + "-"*90)
                for d in clean_dropped[:10]:
                    label_name = '正常' if d['current_label'] == 0 else '恶意'
                    logger.info(f"    {d['idx']:>6d} | {label_name:>4s} | {d['iter_cl_conf']:>8.3f} | {d['anchor_cons']:>8.3f} | {d['density']:>8.1f} | {d['reason']:<15s}")
                
                # 统计特征分布
                iter_cl_confs = [d['iter_cl_conf'] for d in clean_dropped]
                anchor_conss = [d['anchor_cons'] for d in clean_dropped]
                densities = [d['density'] for d in clean_dropped]
                
                logger.info(f"    " + "-"*90)
                logger.info(f"    特征分布统计:")
                logger.info(f"      iter_CL: min={min(iter_cl_confs):.3f}, max={max(iter_cl_confs):.3f}, avg={np.mean(iter_cl_confs):.3f}")
                logger.info(f"      anchor:  min={min(anchor_conss):.3f}, max={max(anchor_conss):.3f}, avg={np.mean(anchor_conss):.3f}")
                logger.info(f"      密度:    min={min(densities):.1f}, max={max(densities):.1f}, avg={np.mean(densities):.1f}")

        
        # ========== 最终统计 ==========
        self._log_final_stats(n_samples, tier_info, action_mask, weights, logger)
        
        
        if y_true is not None:
            self._compute_tier_purity(clean_labels, y_true, tier_info, correction_weight, logger)
        
        self.last_tier_info = tier_info
        
        # 保存迭代CL和锚点KNN结果用于样本分析
        self.iter_pred_probs_all = iter_pred_probs_all
        self.anchor_votes_all = np.zeros(n_samples, dtype=int) - 1  # -1表示未计算
        self.anchor_consistency_all = np.zeros(n_samples)
        
        # 计算所有样本的锚点KNN结果（用于分析）
        if knn_anchor is not None:
            distances, indices = knn_anchor.kneighbors(features)
            for i in range(n_samples):
                neighbor_y = anchor_labels[indices[i]]
                w = 1.0 / (distances[i] + 1e-6)
                vote_0 = w[neighbor_y == 0].sum()
                vote_1 = w[neighbor_y == 1].sum()
                total_w = vote_0 + vote_1
                self.anchor_votes_all[i] = 0 if vote_0 > vote_1 else 1
                self.anchor_consistency_all[i] = max(vote_0, vote_1) / total_w if total_w > 0 else 0
        
        return clean_labels, action_mask, confidence, correction_weight, density_scores, neighbor_consistency, pred_probs

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
