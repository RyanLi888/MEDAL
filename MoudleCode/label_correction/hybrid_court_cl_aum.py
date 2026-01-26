"""
Hybrid Court CL+AUM: 双重校验标签矫正策略
==========================================

核心思想：
1. CL (Contrastive Learning) 提供静态特征度量（空间距离 - 长得像不像）
2. AUM (Area Under Margin) 提供动态训练度量（时间稳定性 - 学得顺不顺）
3. 双重校验机制：与门(AND)剔除噪声，或门(OR)找回干净样本

决策逻辑：
- Tier 1 (Keep): CL高置信度 AND AUM高分数 → 确信干净
- Tier 2 (Flip): (CL低置信度 OR AUM低分数) AND KNN支持翻转 → 矫正标签
- Tier 3 (Drop): 其他情况 → 丢弃（可选）

优势：
- CL 解决 AUM 的边界样本问题（基于特征原型）
- AUM 解决 CL 的相似噪声问题（基于训练动态）
"""

import numpy as np
import torch
import logging
from typing import Tuple, Optional
from .aum_calculator import AUMCalculator

logger = logging.getLogger(__name__)


def correct_labels_cl_aum(
    self,
    features: np.ndarray,
    noisy_labels: np.ndarray,
    device: str = 'cpu',
    y_true: Optional[np.ndarray] = None,
    # CL 参数
    cl_threshold: float = 0.7,
    # AUM 参数
    aum_threshold: float = 0.0,
    aum_epochs: int = 30,
    aum_batch_size: int = 128,
    aum_lr: float = 0.01,
    # KNN 参数
    knn_purity_threshold: float = 0.8,
    # 决策参数
    use_drop: bool = False,  # 是否使用 Drop 动作（默认只 Keep 和 Flip）
    # Phase1: 激进翻转策略（无 Drop）
    phase1_aggressive: bool = False,
    phase1_aggressive_malicious_aum_threshold: float = 0.05,
    phase1_aggressive_malicious_cl_threshold: float = 0.6,
    phase1_aggressive_malicious_knn_cons_threshold: float = 0.6,  # 新增：恶意标签KNN一致性阈值
    phase1_aggressive_benign_aum_threshold: float = -0.05,
    phase1_aggressive_benign_knn_threshold: float = 0.55,
    phase1_malicious_aum_threshold: float = 0.0,
    phase1_malicious_knn_threshold: float = 0.7,
    phase1_malicious_cl_low: float = 0.5,
    phase1_benign_aum_threshold: float = -0.5,
    phase1_benign_knn_threshold: float = 0.7,
    # Phase2: 独立翻转决策（新设计）或保守补刀/救援（旧设计）
    phase2_enable: bool = False,
    phase2_independent: bool = True,  # 是否使用独立翻转策略（不依赖Phase1动作）
    # Phase2独立翻转策略参数
    phase2_malicious_aum_threshold: float = 0.05,
    phase2_malicious_cl_threshold: float = 0.65,
    phase2_malicious_knn_cons_threshold: float = 0.55,
    phase2_benign_aum_threshold: float = -0.2,
    phase2_benign_knn_threshold: float = 0.6,
    # 旧策略参数（兼容）
    phase2_late_flip_aum_threshold: float = -0.5,
    phase2_late_flip_knn_threshold: float = 0.65,
    phase2_late_flip_cl_threshold: float = 0.4,
    phase2_undo_flip_aum_threshold: float = -0.8,
    phase2_undo_flip_cl_threshold: float = 0.25,
    phase2_undo_flip_use_and: bool = False,
    phase2_undo_flip_p1_aum_hesitant: float = -0.2,
    phase2_undo_flip_p1_aum_strong: float = -0.5,
    phase2_undo_flip_p2_aum_weak: float = 1.5,
    recompute_stage2_metrics: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    CL + AUM 双重校验标签矫正
    
    参数:
        self: HybridCourt 实例（包含 cl 和 knn 组件）
        features: (n_samples, feature_dim) 特征矩阵
        noisy_labels: (n_samples,) 噪声标签
        device: 计算设备
        y_true: (n_samples,) 真实标签（用于评估，可选）
        cl_threshold: CL 置信度阈值
        aum_threshold: AUM 分数阈值
        aum_epochs: AUM 训练轮数
        aum_batch_size: AUM 批次大小
        aum_lr: AUM 学习率
        knn_purity_threshold: KNN 纯度阈值
        use_drop: 是否使用 Drop 动作
        
    返回:
        clean_labels: 矫正后的标签
        action_mask: 动作掩码 (0=Keep, 1=Flip, 2=Drop)
        confidence: 置信度
        correction_weight: 矫正权重
        aum_scores: AUM 分数
        neighbor_consistency: KNN 一致性
        pred_probs: CL 预测概率
    """
    n_samples = len(noisy_labels)
    num_classes = len(np.unique(noisy_labels))
    
    def _log_subset_purity(title: str, subset_mask: np.ndarray, use_corrected: bool = False):
        """计算子集纯度"""
        if y_true is None:
            return
        subset_mask = np.asarray(subset_mask, dtype=bool)
        n = int(subset_mask.sum())
        if n == 0:
            logger.info(f"  {title}: 0 samples")
            return
        labels_to_check = clean_labels if use_corrected else noisy_labels
        noise = int(((labels_to_check != y_true) & subset_mask).sum())
        purity = 100.0 * (n - noise) / n
        logger.info(f"  {title}: {n:5d} samples | noise={noise:4d} | purity={purity:.1f}%")
    
    logger.info("="*70)
    logger.info("Hybrid Court Two-Phase Strategy (CL+AUM+KNN, No-Drop)")
    if phase1_aggressive:
        logger.info("  🎯 最终极优化方案 (The Ultimate Design)")
    logger.info("="*70)
    logger.info(f"  样本数: {n_samples}")
    logger.info(f"  类别数: {num_classes}")
    logger.info(f"  CL 阈值: {cl_threshold}")
    logger.info(f"  AUM 阈值: {aum_threshold}")
    logger.info(f"  KNN 纯度阈值: {knn_purity_threshold}")
    if phase1_aggressive:
        logger.info(f"  Phase1 模式: Aggressive (优化版)")
        logger.info(f"     恶意: AUM<{phase1_aggressive_malicious_aum_threshold} 且 ((KNN反对且KNN一致性>{phase1_aggressive_malicious_knn_cons_threshold}) 或 CL<{phase1_aggressive_malicious_cl_threshold})")
        logger.info(f"     正常: AUM<{phase1_aggressive_benign_aum_threshold} 且 KNN反对 且 KNN一致性>{phase1_aggressive_benign_knn_threshold} (或CL<0.4且KNN反对)")
    else:
        logger.info(f"  Phase1 模式: Conservative")
        logger.info(f"    恶意样本: AUM<{phase1_malicious_aum_threshold} 且KNN反对，强KNN>{phase1_malicious_knn_threshold} 或 CL<{phase1_malicious_cl_low}")
        logger.info(f"    正常样本: AUM<{phase1_benign_aum_threshold} 且KNN反对且KNN一致性>{phase1_benign_knn_threshold}")
        if phase2_enable:
            logger.warning("  Phase2 已启用，但 Phase1 仍为 Conservative（未开启 PHASE1_AGGRESSIVE）。若你要启用 Ultimate Strategy 的 Phase1，请设置 PHASE1_AGGRESSIVE=True")
    if phase2_enable:
        logger.info(f"  Phase2: 已启用（策略将在噪声诊断后确定）")
        logger.info(f"    说明: Phase2策略会根据噪声率诊断结果动态选择")
        logger.info(f"    - 低噪声方案: 已禁用（低噪声方案不使用Phase2）")
        logger.info(f"    - 高噪声方案: {'独立翻转决策' if phase2_independent else '保守优化策略'}")
        logger.info(f"    - 高噪声超限方案: 强力修补（救援+补刀）")
    
    # ========== 步骤1: 准备特征 ==========
    logger.info("")
    logger.info("="*70)
    logger.info("步骤1: 准备特征")
    logger.info("="*70)
    
    features_for_analysis = features
    if self.cl.use_projection_head and self.cl.projection_head is not None:
        features_tensor = torch.FloatTensor(features).to(device)
        with torch.no_grad():
            features_projected = self.cl.projection_head(features_tensor).cpu().numpy()
        features_for_analysis = features_projected
        logger.info(f"  ✓ 使用 CL 投影头特征 (原始: {features.shape[1]}D -> 投影: {features_projected.shape[1]}D)")
    else:
        logger.info(f"  ✓ 使用原始特征 ({features.shape[1]}D)")
    
    # ========== 步骤2: 计算 CL 置信度 ==========
    logger.info("")
    logger.info("="*70)
    logger.info("步骤2: 计算 CL 置信度（静态特征度量）")
    logger.info("="*70)
    
    suspected_noise, pred_labels, pred_probs = self.cl.fit_predict(features, noisy_labels)
    
    # 计算 CL 置信度（对当前标签的置信度）
    cl_confidence = np.array([pred_probs[i, int(noisy_labels[i])] for i in range(n_samples)])
    
    logger.info(f"  ✓ CL 完成")
    logger.info(f"    CL 置信度范围: [{cl_confidence.min():.4f}, {cl_confidence.max():.4f}]")
    logger.info(f"    CL 置信度均值: {cl_confidence.mean():.4f}")
    logger.info(f"    CL 识别噪声: {suspected_noise.sum()} 个")
    
    # 分析 CL 与真实噪声的相关性
    if y_true is not None:
        is_noise = (y_true != noisy_labels)
        cl_correlation = np.corrcoef(cl_confidence, is_noise.astype(int))[0, 1]
        logger.info(f"    CL 与噪声相关性: {cl_correlation:.4f} (期望负相关)")
        
        # CL 简单阈值的性能
        cl_pred_noise = (cl_confidence < cl_threshold)
        if cl_pred_noise.sum() > 0:
            cl_precision = (cl_pred_noise & is_noise).sum() / cl_pred_noise.sum()
            cl_recall = (cl_pred_noise & is_noise).sum() / is_noise.sum() if is_noise.sum() > 0 else 0
            logger.info(f"    CL 阈值 {cl_threshold} 性能: Precision={cl_precision:.3f}, Recall={cl_recall:.3f}")
    
    # ========== 步骤3: 计算 AUM 分数 ==========
    logger.info("")
    logger.info("="*70)
    logger.info("步骤3: 计算 AUM 分数（动态训练度量）")
    logger.info("="*70)
    
    aum_calculator = AUMCalculator(
        num_classes=num_classes,
        num_epochs=aum_epochs,
        batch_size=aum_batch_size,
        learning_rate=aum_lr,
        device=device
    )
    
    aum_scores = aum_calculator.fit(features, noisy_labels, verbose=True)
    
    # ========== 噪声率诊断 (Diagnosis) ==========
    # 计算 R_neg: AUM < 0 的样本占比
    r_neg = (aum_scores < 0).sum() / len(aum_scores) * 100.0
    noise_diagnosis_threshold = 35.0  # 判定阈值：35%
    high_aggressive_threshold = 40.0  # 超限方案阈值：40%（对应真实噪声≥40%）
    is_low_noise = r_neg < noise_diagnosis_threshold
    is_high_aggressive = r_neg >= high_aggressive_threshold  # 超限方案
    
    logger.info("")
    logger.info("  📊 噪声率诊断 (Diagnosis):")
    logger.info(f"    R_neg (AUM < 0 占比): {r_neg:.2f}%")
    logger.info(f"    判定阈值: {noise_diagnosis_threshold}%")
    if is_low_noise:
        logger.info(f"    → 方案选择: 低噪声方案 (R_neg < {noise_diagnosis_threshold}%)")
        logger.info(f"    策略: 防守反击 - 高精度优先，稳健清洗")
        logger.info(f"    Phase2: 已禁用（低噪声方案不使用Phase2）")
    elif is_high_aggressive:
        logger.info(f"    → 方案选择: 高噪声超限方案 (R_neg >= {high_aggressive_threshold}%, 真实噪声 ≥ 40%)")
        logger.info(f"    策略: 焦土政策 + 强力修补")
    else:
        logger.info(f"    → 方案选择: 高噪声方案 ({noise_diagnosis_threshold}% <= R_neg < {high_aggressive_threshold}%)")
        logger.info(f"    策略: 保守优化策略")
    logger.info("")
    
    # 分析 AUM 分布
    aum_analysis = aum_calculator.analyze_aum_distribution(aum_scores, y_true, noisy_labels)
    
    if y_true is not None:
        logger.info("  📊 AUM 分布分析:")
        logger.info(f"    干净样本 AUM: {aum_analysis['clean_mean']:.4f} ± {aum_analysis['clean_std']:.4f}")
        logger.info(f"    噪声样本 AUM: {aum_analysis['noise_mean']:.4f} ± {aum_analysis['noise_std']:.4f}")
        logger.info(f"    AUM 与噪声相关性: {aum_analysis['correlation_with_noise']:.4f} (期望负相关)")
    
    # ========== 步骤4: 计算 KNN ==========
    logger.info("")
    logger.info("="*70)
    logger.info("步骤4: 计算 KNN（用于翻转决策）")
    logger.info("="*70)
    
    self.knn.fit(features_for_analysis)
    neighbor_labels, neighbor_consistency = self.knn.predict_semantic_label(features_for_analysis, noisy_labels)
    knn_support_strength = neighbor_consistency
    
    logger.info(f"  ✓ KNN 完成")
    logger.info(f"    KNN 支持强度范围: [{knn_support_strength.min():.4f}, {knn_support_strength.max():.4f}]")
    logger.info(f"    KNN 支持强度均值: {knn_support_strength.mean():.4f}")
    
    # ========== 步骤5: 根据噪声诊断执行完全分离的方案 ==========
    logger.info("")
    logger.info("="*70)
    logger.info("步骤5: 执行标签矫正方案")
    logger.info("="*70)
    
    clean_labels = noisy_labels.copy()
    action_mask = np.zeros(n_samples, dtype=int)  # 0=Keep, 1=Flip, 2=Drop (但只使用Flip)
    confidence = np.ones(n_samples)
    correction_weight = np.ones(n_samples)
    phase1_actions = np.array(['Keep'] * n_samples, dtype=object)
    phase2_actions = np.array([''] * n_samples, dtype=object)
    
    # ========== 完全分离的两套方案 ==========
    if is_low_noise:
        # ========== 低噪声方案：稳健清洗（仅Phase1，不使用Phase2） ==========
        logger.info("")
        logger.info("="*70)
        logger.info("低噪声方案: 稳健清洗 (Conservative Cleaning)")
        logger.info("="*70)
        logger.info("  策略: 防守反击 - 高精度优先，稳健清洗")
        logger.info("  Phase1参数:")
        logger.info("    恶意标签: AUM < -0.1 且 (KNN反对 或 CL < 0.5)")
        logger.info("    正常标签: AUM < -0.15 且 KNN反对 且 KNN一致性 > 0.6")
        logger.info("  Phase2: 已禁用（低噪声方案不使用Phase2）")
        logger.info("")
        
        # Phase1 决策
        flip_count = 0
        keep_count = 0
        flip_correct = 0
        flip_wrong = 0
        
        for i in range(n_samples):
            current_label = int(noisy_labels[i])
            target_label = 1 - current_label
            
            aum_val = float(aum_scores[i])
            cl_cur = float(cl_confidence[i])
            knn_vote = int(neighbor_labels[i])
            knn_cons = float(neighbor_consistency[i])
            knn_opposes = (knn_vote != current_label)
            
            do_flip = False
            
            # 低噪声方案Phase1规则
            if current_label == 1:  # 恶意标签
                # 规则: AUM < -0.1 且 (KNN反对 或 CL < 0.5)
                if aum_val < -0.1:
                    if knn_opposes or cl_cur < 0.5:
                        do_flip = True
            else:  # 正常标签
                # 规则: AUM < -0.15 且 KNN反对 且 KNN一致性 > 0.6
                if aum_val < -0.15 and knn_opposes and knn_cons > 0.6:
                    do_flip = True
            
            if do_flip:
                clean_labels[i] = target_label
                action_mask[i] = 1
                confidence[i] = float(pred_probs[i, target_label])
                correction_weight[i] = 1.0
                phase1_actions[i] = 'Flip'
                flip_count += 1
                if y_true is not None:
                    if int(y_true[i]) == target_label:
                        flip_correct += 1
                    else:
                        flip_wrong += 1
            else:
                action_mask[i] = 0
                confidence[i] = cl_cur
                correction_weight[i] = 1.0
                phase1_actions[i] = 'Keep'
                keep_count += 1
        
        # 低噪声方案：Phase1统计和评估
        logger.info("")
        logger.info("  📊 Phase1 决策统计:")
        logger.info(f"    Flip: {flip_count:5d} ({100*flip_count/n_samples:.1f}%)")
        logger.info(f"    未翻转: {keep_count:5d} ({100*keep_count/n_samples:.1f}%)")
        
        if y_true is not None:
            is_noise = (np.asarray(y_true) != np.asarray(noisy_labels))
            keep_mask = (action_mask == 0)
            flip_mask = (action_mask == 1)
            
            def _log_action_noise_table(action_name: str, mask: np.ndarray):
                n = int(mask.sum())
                if n == 0:
                    logger.info("")
                    logger.info(f"  {action_name}: 0 samples")
                    return
                noise_n = int((is_noise & mask).sum())
                clean_n = n - noise_n
                logger.info("")
                logger.info(f"  {action_name} 统计: total={n} | clean={clean_n} | noise={noise_n} | acc={100.0*(n-noise_n)/n:.1f}%")
            
            _log_action_noise_table("未翻转", keep_mask)
            _log_action_noise_table("Flip", flip_mask)
            
            if flip_count > 0:
                logger.info("")
                logger.info(f"    Flip 准确率: {flip_correct/flip_count:.3f} ({flip_correct}/{flip_count})")
                logger.info(f"    Flip 错误: {flip_wrong} 个")
            
            # Phase1 整体纯度
            correct = (clean_labels == y_true).sum()
            purity = 100.0 * correct / n_samples
            logger.info("")
            logger.info(f"  📊 Phase1 整体纯度: {purity:.2f}% ({correct}/{n_samples})")
            
            _log_subset_purity("未翻转", (action_mask == 0), use_corrected=True)
            _log_subset_purity("Flip", (action_mask == 1), use_corrected=True)
            
            # 对比原始标签
            original_correct = (noisy_labels == y_true).sum()
            original_purity = 100.0 * original_correct / n_samples
            improvement = purity - original_purity
            logger.info("")
            logger.info("  📈 Phase1 改进效果:")
            logger.info(f"    原始纯度: {original_purity:.2f}%")
            logger.info(f"    Phase1矫正纯度: {purity:.2f}%")
            logger.info(f"    提升: {improvement:+.2f}%")
        
        # 低噪声方案：直接返回，不使用Phase2
        logger.info("")
        logger.info("="*70)
        logger.info("✓ 低噪声方案完成（不使用Phase2）")
        logger.info("="*70)
        logger.info(f"  阶段1: Flip={flip_count} | 未翻转={keep_count}")
        logger.info(f"  阶段2: 已跳过（低噪声方案不使用Phase2）")
        
        return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
    
    else:
        # ========== 高噪声方案：完整的两阶段方案（包括普通高噪声和超限方案） ==========
        if is_high_aggressive:
            logger.info("")
            logger.info("="*70)
            logger.info("高噪声超限方案: 焦土政策 + 强力修补")
            logger.info("="*70)
            logger.info("  Phase1参数（统一规则 - 优化版）:")
            logger.info("    AUM < -0.09 或 CL差值 > 0.40 → 翻转")
            logger.info("    (使用'或'逻辑，降低KNN权重，基于数据分析优化)")
            logger.info("  Phase2参数（强力修补）:")
            logger.info("    救援(UndoFlip): Stage2_AUM < -0.8")
            logger.info("    补刀(LateFlip): Stage2_AUM < -0.4")
        else:
            logger.info("")
            logger.info("="*70)
            logger.info("高噪声方案: 保守优化策略")
            logger.info("="*70)
            logger.info("  Phase1参数（统一规则 - 优化版）:")
            logger.info("    AUM < -0.09 或 CL差值 > 0.40 → 翻转")
            logger.info("    (使用'或'逻辑，降低KNN权重，基于数据分析优化)")
            logger.info("  Phase2参数:")
            logger.info(f"    LateFlip: AUM<{phase2_late_flip_aum_threshold} 且 KNN>{phase2_late_flip_knn_threshold} 且 CL<{phase2_late_flip_cl_threshold}")
            logger.info(f"    UndoFlip: AUM<{phase2_undo_flip_aum_threshold} 或 CL<{phase2_undo_flip_cl_threshold} (OR条件)")
        logger.info("")
        
        # Phase1 决策
        flip_count = 0
        keep_count = 0
        flip_correct = 0
        flip_wrong = 0
        
        for i in range(n_samples):
            current_label = int(noisy_labels[i])
            target_label = 1 - current_label
            
            aum_val = float(aum_scores[i])
            cl_cur = float(cl_confidence[i])
            knn_vote = int(neighbor_labels[i])
            knn_cons = float(neighbor_consistency[i])
            knn_opposes = (knn_vote != current_label)
            
            do_flip = False
            
            # 高噪声方案 Phase1 统一规则（优化版：基于分析结果）
            # 优化策略：使用"或"逻辑，AUM < -0.09 或 CL差值 > 0.40 即翻转
            # 降低KNN权重（KNN区分能力弱，仅作为辅助指标）
            cl_target = float(pred_probs[i, target_label])
            cl_gap = cl_target - cl_cur
            # 新规则：AUM < -0.09 或 CL差值 > 0.40
            if (aum_val < -0.09) or (cl_gap > 0.40):
                do_flip = True
            
            if do_flip:
                clean_labels[i] = target_label
                action_mask[i] = 1
                confidence[i] = float(pred_probs[i, target_label])
                correction_weight[i] = 1.0
                phase1_actions[i] = 'Flip'
                flip_count += 1
                if y_true is not None:
                    if int(y_true[i]) == target_label:
                        flip_correct += 1
                    else:
                        flip_wrong += 1
            else:
                action_mask[i] = 0
                confidence[i] = cl_cur
                correction_weight[i] = 1.0
                phase1_actions[i] = 'Keep'
                keep_count += 1
        
        # 高噪声方案：Phase1统计和评估
        logger.info("")
        logger.info("="*70)
        logger.info("步骤5.1: Phase1 决策统计")
        logger.info("="*70)
        logger.info("")
        logger.info("  📊 Phase1 决策统计:")
        logger.info(f"    Flip: {flip_count:5d} ({100*flip_count/n_samples:.1f}%)")
        logger.info(f"    未翻转: {keep_count:5d} ({100*keep_count/n_samples:.1f}%)")

        if y_true is not None:
            is_noise = (np.asarray(y_true) != np.asarray(noisy_labels))
            keep_mask = (action_mask == 0)
            flip_mask = (action_mask == 1)

            def _log_action_noise_table(action_name: str, mask: np.ndarray):
                n = int(mask.sum())
                if n == 0:
                    logger.info("")
                    logger.info(f"  {action_name}: 0 samples")
                    return
                noise_n = int((is_noise & mask).sum())
                clean_n = n - noise_n
                correct_n = int(((clean_labels == y_true) & mask).sum())
                acc = 100.0 * correct_n / n
                logger.info("")
                logger.info(f"  {action_name} 统计: total={n} | clean={clean_n} | noise={noise_n} | acc={acc:.1f}%")

            _log_action_noise_table("未翻转", keep_mask)
            _log_action_noise_table("Flip", flip_mask)

            if flip_count > 0:
                flip_precision = flip_correct / flip_count
                logger.info("")
                logger.info(f"    Flip 准确率: {flip_precision:.3f} ({flip_correct}/{flip_count})")
                logger.info(f"    Flip 错误: {flip_wrong} 个")
            
            # Phase1 整体纯度
            correct = (clean_labels == y_true).sum()
            purity = 100.0 * correct / n_samples
            logger.info("")
            logger.info(f"  📊 Phase1 整体纯度: {purity:.2f}% ({correct}/{n_samples})")
            
            _log_subset_purity("未翻转", (action_mask == 0), use_corrected=True)
            _log_subset_purity("Flip", (action_mask == 1), use_corrected=True)
            
            # 对比原始标签
            original_correct = (noisy_labels == y_true).sum()
            original_purity = 100.0 * original_correct / n_samples
            improvement = purity - original_purity
            logger.info("")
            logger.info("  📈 Phase1 改进效果:")
            logger.info(f"    原始纯度: {original_purity:.2f}%")
            logger.info(f"    Phase1矫正纯度: {purity:.2f}%")
            logger.info(f"    提升: {improvement:+.2f}%")
        
        # ========== Step 7: 重新训练 Phase2 模型（基于 Phase1 矫正后标签） ==========
        stage2_aum_scores = None
        stage2_neighbor_labels = None
        stage2_neighbor_consistency = None
        iter_pred_probs = None
        if phase2_enable:
            logger.info("")
            logger.info("="*70)
            logger.info("步骤7: 基于Phase1矫正后标签重新训练CL/AUM/KNN（用于Phase2指标）")
            logger.info("="*70)
            logger.info("  说明: 将Phase1矫正后的标签视为新的噪声数据集，完全重新训练所有模型")
            logger.info("")

            stage1_suspected_noise = suspected_noise
            stage1_pred_labels = pred_labels
            stage1_pred_probs = pred_probs
            stage1_neighbor_labels = neighbor_labels
            stage1_neighbor_consistency = neighbor_consistency

            # ========== 步骤1: 准备特征 ==========
            logger.info("="*70)
            logger.info("步骤1: 准备特征")
            logger.info("="*70)
            
            # 使用相同的特征准备逻辑
            features_for_stage2 = features
            if self.cl.use_projection_head and self.cl.projection_head is not None:
                features_tensor = torch.FloatTensor(features).to(device)
                with torch.no_grad():
                    features_projected = self.cl.projection_head(features_tensor).cpu().numpy()
                features_for_stage2 = features_projected
                logger.info(f"  ✓ 使用 CL 投影头特征 (原始: {features.shape[1]}D -> 投影: {features_projected.shape[1]}D)")
            else:
                logger.info(f"  ✓ 使用原始特征 ({features.shape[1]}D)")
            logger.info("")

            # ========== 步骤2: 计算 CL 置信度（基于Phase1矫正后标签） ==========
            logger.info("="*70)
            logger.info("步骤2: 计算 CL 置信度（静态特征度量）")
            logger.info("="*70)
            
            suspected_noise_2, pred_labels_2, pred_probs_2 = self.cl.fit_predict(features, clean_labels)
            
            # 计算 CL 置信度（对Phase1矫正后标签的置信度）
            cl_confidence_2 = np.array([pred_probs_2[i, int(clean_labels[i])] for i in range(n_samples)])
            
            logger.info(f"  ✓ CL 完成")
            logger.info(f"    CL 置信度范围: [{cl_confidence_2.min():.4f}, {cl_confidence_2.max():.4f}]")
            logger.info(f"    CL 置信度均值: {cl_confidence_2.mean():.4f}")
            logger.info(f"    CL 识别噪声: {suspected_noise_2.sum()} 个")
            
            # 分析 CL 与真实噪声的相关性（基于Phase1矫正后的标签）
            if y_true is not None:
                is_noise_2 = (y_true != clean_labels)  # 基于Phase1矫正后的标签判断噪声
                cl_correlation_2 = np.corrcoef(cl_confidence_2, is_noise_2.astype(int))[0, 1]
                logger.info(f"    CL 与噪声相关性: {cl_correlation_2:.4f} (期望负相关)")
                
                # CL 简单阈值的性能
                cl_pred_noise_2 = (cl_confidence_2 < cl_threshold)
                if cl_pred_noise_2.sum() > 0:
                    cl_precision_2 = (cl_pred_noise_2 & is_noise_2).sum() / cl_pred_noise_2.sum()
                    cl_recall_2 = (cl_pred_noise_2 & is_noise_2).sum() / is_noise_2.sum() if is_noise_2.sum() > 0 else 0
                    logger.info(f"    CL 阈值 {cl_threshold} 性能: Precision={cl_precision_2:.3f}, Recall={cl_recall_2:.3f}")
            logger.info("")

            # ========== 步骤3: 计算 AUM 分数（基于Phase1矫正后标签） ==========
            logger.info("="*70)
            logger.info("步骤3: 计算 AUM 分数（动态训练度量）")
            logger.info("="*70)
            
            aum_calculator_2 = AUMCalculator(
                num_classes=num_classes,
                num_epochs=aum_epochs,
                batch_size=aum_batch_size,
                learning_rate=aum_lr,
                device=device
            )
            
            stage2_aum_scores = aum_calculator_2.fit(features, clean_labels, verbose=True)
            
            # 分析 AUM 分布（基于Phase1矫正后的标签）
            aum_analysis_2 = aum_calculator_2.analyze_aum_distribution(stage2_aum_scores, y_true, clean_labels)
            
            if y_true is not None:
                logger.info("")
                logger.info("  📊 AUM 分布分析:")
                logger.info(f"    干净样本 AUM: {aum_analysis_2['clean_mean']:.4f} ± {aum_analysis_2['clean_std']:.4f}")
                logger.info(f"    噪声样本 AUM: {aum_analysis_2['noise_mean']:.4f} ± {aum_analysis_2['noise_std']:.4f}")
                logger.info(f"    AUM 与噪声相关性: {aum_analysis_2['correlation_with_noise']:.4f} (期望负相关)")

            # ========== 步骤4: 计算 KNN（基于Phase1矫正后标签） ==========
            logger.info("")
            logger.info("="*70)
            logger.info("步骤4: 计算 KNN（用于翻转决策）")
            logger.info("="*70)
            
            self.knn.fit(features_for_stage2)
            stage2_neighbor_labels, stage2_neighbor_consistency = self.knn.predict_semantic_label(features_for_stage2, clean_labels)
            knn_support_strength_2 = stage2_neighbor_consistency
            
            logger.info(f"  ✓ KNN 完成")
            logger.info(f"    KNN 支持强度范围: [{knn_support_strength_2.min():.4f}, {knn_support_strength_2.max():.4f}]")
            logger.info(f"    KNN 支持强度均值: {knn_support_strength_2.mean():.4f}")

            iter_pred_probs = pred_probs_2
            self.stage2_cl_suspected_noise_all = suspected_noise_2
            self.stage2_cl_pred_labels_all = pred_labels_2
            self.stage2_cl_pred_probs_all = pred_probs_2
            self.stage2_aum_scores_all = stage2_aum_scores
            self.stage2_knn_neighbor_labels_all = stage2_neighbor_labels
            self.stage2_knn_neighbor_consistency_all = stage2_neighbor_consistency

            self.iter_pred_probs_all = pred_probs_2

            self.cl.last_suspected_noise = stage1_suspected_noise
            self.cl.last_pred_labels = stage1_pred_labels
            self.cl.last_pred_probs = stage1_pred_probs
            self.knn.last_neighbor_labels = stage1_neighbor_labels
            self.knn.last_neighbor_consistency = stage1_neighbor_consistency

            # ========== Step 8: Phase2 决策（高噪声方案专用） ==========
            phase2_actions = np.array([''] * n_samples, dtype=object)
            logger.info("")
            logger.info("="*70)
            
            # 高噪声方案的Phase2策略
            if is_high_aggressive:
                # 高噪声超限方案：强力修补 (Strong Repair)
                logger.info("阶段2: Phase2 强力修补 (高噪声超限方案)")
                logger.info("="*70)
                logger.info("  策略说明:")
                logger.info("    - 救援逻辑: 因为P1杀红眼了，P2必须放宽救援门槛，把误杀的好人救回来")
                logger.info("      * UndoFlip: Stage2_AUM < -0.8")
                logger.info("    - 补刀逻辑: 继续清理漏网之鱼")
                logger.info("      * LateFlip: Stage2_AUM < -0.4")
            elif phase2_independent:
                # 高噪声方案：独立翻转决策（新设计）
                logger.info("阶段2: Phase2 独立翻转决策 (高噪声方案)")
                logger.info("="*70)
                logger.info("  策略说明:")
                logger.info("    - 完全独立于Phase1动作，基于Phase1矫正后标签重新训练的模型指标")
                logger.info("    - 对所有样本独立决定是否翻转，不区分UndoFlip/LateFlip")
                logger.info(f"    【恶意标签】: stage2_AUM<{phase2_malicious_aum_threshold} 且 ((stage2_KNN反对 且 stage2_KNN一致性<{phase2_malicious_knn_cons_threshold}) 或 stage2_CL<{phase2_malicious_cl_threshold})")
                logger.info(f"    【正常标签】: stage2_AUM<{phase2_benign_aum_threshold} 且 stage2_KNN反对 且 stage2_KNN一致性>{phase2_benign_knn_threshold}")
                logger.info("      使用更严格的阈值，因为Phase1已经做了初步矫正")
            else:
                # 高噪声方案：保守优化策略（旧设计）
                logger.info("阶段2: Phase2 保守优化策略 (高噪声方案)")
                logger.info("="*70)
                logger.info("  策略说明:")
                logger.info(f"    - LateFlip: Phase1保持但Stage2指标显示应翻转（AUM<{phase2_late_flip_aum_threshold} 且 KNN>{phase2_late_flip_knn_threshold} 且 CL<{phase2_late_flip_cl_threshold}）")
                logger.info(f"    - UndoFlip: Phase1翻转但Stage2指标显示应撤销（严苛：AUM<{phase2_undo_flip_aum_threshold} 或 CL<{phase2_undo_flip_cl_threshold}，OR条件）")
                logger.info("      更严格的条件，减少误判")

            final_labels = clean_labels.copy()

            if is_high_aggressive:
                # ========== 高噪声超限方案: 强力修补 (Strong Repair) ==========
                # 因为P1杀红眼了，P2必须放宽救援门槛，把误杀的好人救回来
                for i in range(n_samples):
                    cur_label = int(clean_labels[i])
                    orig_label = int(noisy_labels[i])
                    phase2_actions[i] = 'NoChange'

                    # 获取Stage2指标
                    s2_aum = float(stage2_aum_scores[i]) if stage2_aum_scores is not None else None

                    # 救援逻辑：UndoFlip - 因为P1杀红眼了，P2必须放宽救援门槛
                    if phase1_actions[i] == 'Flip':
                        # 救援条件：Stage2_AUM < -0.8（放宽门槛，救回误杀的好人）
                        if (s2_aum is not None) and (s2_aum < -0.8):
                            final_labels[i] = orig_label
                            phase2_actions[i] = 'UndoFlip'

                    # 补刀逻辑：LateFlip - 继续清理漏网之鱼
                    elif phase1_actions[i] == 'Keep':
                        # 补刀条件：Stage2_AUM < -0.4（更积极的补刀）
                        if (s2_aum is not None) and (s2_aum < -0.4):
                            final_labels[i] = 1 - cur_label
                            phase2_actions[i] = 'LateFlip'
            elif phase2_independent:
                # 新设计：独立翻转决策（只做Flip，不Keep）
                # 优化阈值：根据样本数据分析，阶段2需要更严格的阈值以提升准确度
                for i in range(n_samples):
                    cur_label = int(clean_labels[i])  # Phase1矫正后的标签
                    phase2_actions[i] = 'Flip'  # 默认标记为Flip，但实际只翻转满足条件的

                    # 获取Stage2指标（基于Phase1矫正后标签重新训练的模型）
                    iter_cl_current = None
                    if iter_pred_probs is not None:
                        iter_cl_current = float(iter_pred_probs[i, cur_label])

                    s2_aum = float(stage2_aum_scores[i]) if stage2_aum_scores is not None else None
                    s2_knn_vote = int(stage2_neighbor_labels[i]) if stage2_neighbor_labels is not None else None
                    s2_knn_cons = float(stage2_neighbor_consistency[i]) if stage2_neighbor_consistency is not None else None
                    s2_knn_opposes = (s2_knn_vote is not None) and (s2_knn_vote != cur_label)

                    do_flip = False
                    if cur_label == 1:  # 恶意标签
                        # 优化：更严格的阈值，要求AUM更低且(KNN反对且一致性更低 或 CL更低)
                        # 根据样本数据分析，降低阈值以提升准确度
                        if (s2_aum is not None) and (s2_aum < phase2_malicious_aum_threshold):
                            condition1 = s2_knn_opposes and (s2_knn_cons is not None) and (s2_knn_cons < phase2_malicious_knn_cons_threshold)
                            condition2 = (iter_cl_current is not None) and (iter_cl_current < phase2_malicious_cl_threshold)
                            if condition1 or condition2:
                                do_flip = True
                    else:  # 正常标签
                        # 优化：更严格的阈值，要求AUM更低且KNN反对且一致性更高
                        if s2_knn_opposes and (s2_aum is not None) and (s2_aum < phase2_benign_aum_threshold) and (s2_knn_cons is not None) and (s2_knn_cons > phase2_benign_knn_threshold):
                            do_flip = True

                    if do_flip:
                        final_labels[i] = 1 - cur_label
                        phase2_actions[i] = 'Flip'
                    else:
                        # 不满足Flip条件，保持Phase1的标签，但不标记为Keep
                        phase2_actions[i] = 'Flip'  # 统一标记，但实际不翻转
            else:
                # 旧设计：保守补刀/救援
                for i in range(n_samples):
                    cur_label = int(clean_labels[i])
                    orig_label = int(noisy_labels[i])
                    phase2_actions[i] = 'NoChange'

                    # iter_CL_current 代表当前标签(Phase1矫正后标签)的 CL 置信度
                    iter_cl_current = None
                    if iter_pred_probs is not None:
                        iter_cl_current = float(iter_pred_probs[i, cur_label])

                    s2_aum = float(stage2_aum_scores[i]) if stage2_aum_scores is not None else None
                    s2_knn_vote = int(stage2_neighbor_labels[i]) if stage2_neighbor_labels is not None else None
                    s2_knn_cons = float(stage2_neighbor_consistency[i]) if stage2_neighbor_consistency is not None else None
                    s2_knn_opposes = (s2_knn_vote is not None) and (s2_knn_vote != cur_label)

                    # 1) Rescue / Undo Flip - 严苛OR条件（简化策略）
                    if phase1_actions[i] == 'Flip':
                        undo = False
                        # 严苛条件：AUM<-0.8 或 CL<0.25（OR条件）
                        aum_condition = (s2_aum is not None) and (s2_aum < phase2_undo_flip_aum_threshold)
                        cl_condition = (iter_cl_current is not None) and (iter_cl_current < phase2_undo_flip_cl_threshold)
                        undo = aum_condition or cl_condition

                        if undo:
                            final_labels[i] = orig_label
                            phase2_actions[i] = 'UndoFlip'

                    # 2) Late Flip - AND条件（AUM<-0.5 且 KNN>0.65 且 CL<0.4）
                    elif phase1_actions[i] == 'Keep':
                        late = False
                        # 严苛条件：必须同时满足AUM、KNN、CL三个条件
                        aum_ok = (s2_aum is not None) and (s2_aum < phase2_late_flip_aum_threshold)
                        knn_ok = s2_knn_opposes and (s2_knn_cons is not None) and (s2_knn_cons > phase2_late_flip_knn_threshold)
                        cl_ok = (iter_cl_current is not None) and (iter_cl_current < phase2_late_flip_cl_threshold)
                        if aum_ok and knn_ok and cl_ok:
                            late = True

                        if late:
                            final_labels[i] = 1 - cur_label
                            phase2_actions[i] = 'LateFlip'

            # Phase2 统计 - 增强输出
            if is_high_aggressive:
                # 高噪声超限方案：UndoFlip、LateFlip、NoChange
                n_undo = int((phase2_actions == 'UndoFlip').sum())
                n_late = int((phase2_actions == 'LateFlip').sum())
                n_nochange = int((phase2_actions == 'NoChange').sum())

                logger.info("")
                logger.info("  📊 Phase2 动作统计:")
                logger.info(f"    UndoFlip (救援): {n_undo} 个")
                logger.info(f"    LateFlip (补刀): {n_late} 个")
                logger.info(f"    NoChange (无变化): {n_nochange} 个")

                if y_true is not None:
                    is_noise = (np.asarray(y_true) != np.asarray(noisy_labels))
                    undo_mask = (phase2_actions == 'UndoFlip')
                    late_mask = (phase2_actions == 'LateFlip')

                    def _log_mask(title: str, mask: np.ndarray):
                        n = int(mask.sum())
                        if n == 0:
                            logger.info(f"    {title}: 0 个")
                            return
                        noise_n = int((is_noise & mask).sum())
                        clean_n = n - noise_n
                        correct_n = int(((final_labels == y_true) & mask).sum())
                        acc = 100.0 * correct_n / n
                        logger.info(f"    {title}: {n} 个 | 正确={correct_n} | 错误={n-correct_n} | 准确率={acc:.1f}%")

                    logger.info("")
                    logger.info("  📈 Phase2 效果评估:")
                    _log_mask('UndoFlip', undo_mask)
                    _log_mask('LateFlip', late_mask)
            elif phase2_independent:
                # 新设计：只有Flip（不Keep）
                # 计算实际翻转的样本数（final_labels != clean_labels）
                actual_flip_mask = (final_labels != clean_labels)
                n_flip = int(actual_flip_mask.sum())
                n_no_flip = n_samples - n_flip

                logger.info("")
                logger.info("  📊 Phase2 动作统计:")
                logger.info(f"    Flip (翻转): {n_flip} 个")
                logger.info(f"    未翻转: {n_no_flip} 个")

                if y_true is not None:
                    flip_mask = actual_flip_mask
                    no_flip_mask = ~actual_flip_mask

                    def _log_mask(title: str, mask: np.ndarray):
                        n = int(mask.sum())
                        if n == 0:
                            logger.info(f"    {title}: 0 个")
                            return
                        correct_n = int(((final_labels == y_true) & mask).sum())
                        acc = 100.0 * correct_n / n
                        logger.info(f"    {title}: {n} 个 | 正确={correct_n} | 错误={n-correct_n} | 准确率={acc:.1f}%")

                    logger.info("")
                    logger.info("  📈 Phase2 效果评估:")
                    _log_mask('Flip', flip_mask)
                    _log_mask('未翻转', no_flip_mask)
            else:
                # 旧设计：UndoFlip、LateFlip、NoChange
                n_undo = int((phase2_actions == 'UndoFlip').sum())
                n_late = int((phase2_actions == 'LateFlip').sum())
                n_nochange = int((phase2_actions == 'NoChange').sum())

                logger.info("")
                logger.info("  📊 Phase2 动作统计:")
                logger.info(f"    UndoFlip (撤销翻转): {n_undo} 个")
                logger.info(f"    LateFlip (延迟翻转): {n_late} 个")
                logger.info(f"    NoChange (无变化):   {n_nochange} 个")

                if y_true is not None:
                    is_noise = (np.asarray(y_true) != np.asarray(noisy_labels))
                    undo_mask = (phase2_actions == 'UndoFlip')
                    late_mask = (phase2_actions == 'LateFlip')

                    def _log_mask(title: str, mask: np.ndarray):
                        n = int(mask.sum())
                        if n == 0:
                            logger.info(f"    {title}: 0 个")
                            return
                        noise_n = int((is_noise & mask).sum())
                        clean_n = n - noise_n
                        correct_n = int(((final_labels == y_true) & mask).sum())
                        acc = 100.0 * correct_n / n
                        logger.info(f"    {title}: {n} 个 | 正确={correct_n} | 错误={n-correct_n} | 准确率={acc:.1f}%")

                    logger.info("")
                    logger.info("  📈 Phase2 效果评估:")
                    _log_mask('UndoFlip', undo_mask)
                    _log_mask('LateFlip', late_mask)

            # 暴露 Phase2 action 供分析输出
            self.phase2_action_all = phase2_actions

            # ========== 步骤9: 最终评估（Phase2之后） ==========
            logger.info("")
            logger.info("="*70)
            logger.info("步骤9: 最终评估（阶段2独立总结）")
            logger.info("="*70)
            
            if y_true is not None:
                # 高噪声方案：阶段2的最终结果就是最后的结果
                # 计算Phase1的纯度（需要从phase1_actions反推）
                # 在独立翻转策略中，clean_labels是Phase1矫正后的标签
                # 如果Phase2翻转了，final_labels是Phase2矫正后的标签
                # 所以phase1_labels就是clean_labels（Phase1矫正后的标签）
                phase1_labels = clean_labels.copy()
                if not phase2_independent and not is_high_aggressive:
                    # 旧策略需要特殊处理（不包括超限方案，因为超限方案已经在前面处理了）
                    for i in range(n_samples):
                        if phase2_actions[i] == 'UndoFlip':
                            # UndoFlip撤销了Flip，所以Phase1是Flip后的标签
                            phase1_labels[i] = 1 - noisy_labels[i]
                        elif phase2_actions[i] == 'LateFlip':
                            # LateFlip是从Keep翻转的，所以Phase1是Keep，即noisy_label
                            phase1_labels[i] = noisy_labels[i]
                        # NoChange的情况，Phase1的标签就是clean_labels（因为Phase2没改）
                elif is_high_aggressive:
                    # 超限方案：需要根据Phase2动作反推Phase1标签
                    for i in range(n_samples):
                        if phase2_actions[i] == 'UndoFlip':
                            # UndoFlip撤销了Flip，所以Phase1是Flip后的标签
                            phase1_labels[i] = 1 - noisy_labels[i]
                        elif phase2_actions[i] == 'LateFlip':
                            # LateFlip是从Keep翻转的，所以Phase1是Keep，即noisy_label
                            phase1_labels[i] = noisy_labels[i]
                        else:
                            # NoChange的情况，Phase1的标签就是clean_labels（因为Phase2没改）
                            phase1_labels[i] = clean_labels[i]
                
                # 阶段1独立总结
                phase1_correct = (phase1_labels == y_true).sum()
                phase1_purity = 100.0 * phase1_correct / n_samples
                original_correct = (noisy_labels == y_true).sum()
                original_purity = 100.0 * original_correct / n_samples
                phase1_improvement = phase1_purity - original_purity
                
                logger.info("")
                logger.info("  📊 阶段1独立总结:")
                logger.info(f"    原始纯度: {original_purity:.2f}%")
                logger.info(f"    阶段1矫正纯度: {phase1_purity:.2f}%")
                logger.info(f"    阶段1提升: {phase1_improvement:+.2f}%")
                
                # 阶段2独立总结（阶段2的最终结果就是最后的结果）
                final_correct = (final_labels == y_true).sum()
                final_purity = 100.0 * final_correct / n_samples
                phase2_improvement = final_purity - phase1_purity
                
                logger.info("")
                logger.info("  📊 阶段2独立总结（最终结果）:")
                logger.info(f"    阶段1矫正后纯度: {phase1_purity:.2f}%")
                logger.info(f"    阶段2最终纯度: {final_purity:.2f}%")
                logger.info(f"    阶段2提升: {phase2_improvement:+.2f}%")
                logger.info(f"    总提升: {final_purity - original_purity:+.2f}%")
                
                # 统计各动作的最终纯度
                final_action_mask = (final_labels != noisy_labels).astype(int)
                _log_subset_purity("未翻转", (final_action_mask == 0), use_corrected=True)
                _log_subset_purity("Flip", (final_action_mask == 1), use_corrected=True)
        
        logger.info("")
        logger.info("="*70)
        logger.info("✓ 高噪声方案两阶段标签矫正完成")
        logger.info("="*70)
        phase1_flip_count = int((action_mask == 1).sum())
        phase1_no_flip_count = n_samples - phase1_flip_count
        logger.info(f"  阶段1: Flip={phase1_flip_count} | 未翻转={phase1_no_flip_count}")
        if phase2_enable:
            if phase2_independent:
                phase2_flip_count = int((final_labels != clean_labels).sum())
                phase2_no_flip_count = n_samples - phase2_flip_count
                logger.info(f"  阶段2: Flip={phase2_flip_count} | 未翻转={phase2_no_flip_count}")
            else:
                logger.info(f"  阶段2: UndoFlip={int((phase2_actions == 'UndoFlip').sum())} | LateFlip={int((phase2_actions == 'LateFlip').sum())}")
        else:
            logger.info(f"  阶段2: 未启用")

        # 返回最终标签：高噪声方案如果有Phase2，返回final_labels；否则返回clean_labels
        if phase2_enable:
            return final_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        else:
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
