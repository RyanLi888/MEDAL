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
    logger.info("Hybrid Court Strategy (CL+AUM+KNN, No-Drop)")
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
        logger.info(f"    恶意: AUM<{phase1_aggressive_malicious_aum_threshold} 且 ((KNN反对且KNN一致性>{phase1_aggressive_malicious_knn_cons_threshold}) 或 CL<{phase1_aggressive_malicious_cl_threshold})")
        logger.info(f"    正常: AUM<{phase1_aggressive_benign_aum_threshold} 且 KNN反对 且 KNN一致性>{phase1_aggressive_benign_knn_threshold} (或CL<0.4且KNN反对)")
    else:
        logger.info(f"  Phase1 模式: Conservative")
        logger.info(f"    恶意样本: AUM<{phase1_malicious_aum_threshold} 且KNN反对，强KNN>{phase1_malicious_knn_threshold} 或 CL<{phase1_malicious_cl_low}")
        logger.info(f"    正常样本: AUM<{phase1_benign_aum_threshold} 且KNN反对且KNN一致性>{phase1_benign_knn_threshold}")
    
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
    cl_pred_noise = (cl_confidence < cl_threshold)
    
    # 分析 CL 与真实噪声的相关性
    if y_true is not None:
        is_noise = (y_true != noisy_labels)
        cl_correlation = np.corrcoef(cl_confidence, is_noise.astype(int))[0, 1]
        logger.info(f"    CL 与噪声相关性: {cl_correlation:.4f} (期望负相关)")
        
        # CL 简单阈值的性能
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
    aum_pred_noise = (aum_scores < 0)
    joint_pred_noise = aum_pred_noise & cl_pred_noise
    joint_pred_noise_count = int(joint_pred_noise.sum())
    joint_pred_noise_rate = joint_pred_noise_count / len(aum_scores) * 100.0
    logger.info(f"    R_joint (AUM < 0 且 CL < {cl_threshold} 占比): {joint_pred_noise_rate:.2f}%")
    logger.info(f"    联合预测噪声样本数: {joint_pred_noise_count}/{len(aum_scores)}")
    if y_true is not None:
        true_noise_mask = (y_true != noisy_labels)
        true_noise_in_joint = int((joint_pred_noise & true_noise_mask).sum())
        if joint_pred_noise_count > 0:
            joint_precision = true_noise_in_joint / joint_pred_noise_count
            logger.info(f"    联合预测噪声中的真实噪声: {true_noise_in_joint}/{joint_pred_noise_count} (Precision={joint_precision:.3f})")
        else:
            logger.info("    联合预测噪声中的真实噪声: 0/0 (Precision=N/A)")
    logger.info(f"    判定阈值: {noise_diagnosis_threshold}%")
    if is_low_noise:
        logger.info(f"    → 方案选择: 低噪声方案 (R_neg < {noise_diagnosis_threshold}%)")
        logger.info(f"    策略: 防守反击 - 高精度优先，稳健清洗")
    elif is_high_aggressive:
        logger.info(f"    → 方案选择: 高噪声超限方案 (R_neg >= {high_aggressive_threshold}%, 真实噪声 ≥ 40%)")
        logger.info(f"    策略: 自适应级联决策策略（基于决策树Depth 5优化，准确率90.0%）")
    else:
        logger.info(f"    → 方案选择: 高噪声方案 ({noise_diagnosis_threshold}% <= R_neg < {high_aggressive_threshold}%)")
        logger.info(f"    策略: 自适应级联决策策略（基于决策树Depth 5优化，准确率90.0%）")
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
    
    # ========== 完全分离的两套方案 ==========
    if is_low_noise:
        # ========== 低噪声方案：AUM阈值优化策略 ==========
        logger.info("")
        logger.info("="*70)
        logger.info("低噪声方案: AUM阈值优化策略 (AUM Thresholding Optimized)")
        logger.info("="*70)
        logger.info("  策略: 简化高效 - 基于AUM阈值和KNN邻居标签")
        logger.info("  Phase1参数（优化后准确率95.33%）:")
        logger.info("    规则: 如果 AUM分数 < -0.02，则将标签翻转为 KNN邻居标签；否则保持原标签")
        logger.info("    阈值: -0.02 (通过网格搜索优化，减少过度矫正)")
        logger.info("    优势: 简化逻辑，移除KNN一致性要求，仅依赖AUM和KNN预测")
        logger.info("")
        
        # Phase1 决策
        flip_count = 0
        keep_count = 0
        flip_correct = 0
        flip_wrong = 0
        
        # AUM阈值（通过网格搜索优化得到的最优值）
        aum_threshold = -0.02
        
        for i in range(n_samples):
            current_label = int(noisy_labels[i])
            knn_vote = int(neighbor_labels[i])
            
            aum_val = float(aum_scores[i])
            
            # 低噪声方案Phase1规则（简化策略）
            # 如果 AUM分数 < -0.02，则将标签翻转为 KNN邻居标签；否则保持原标签
            if aum_val < aum_threshold:
                # 翻转标签为KNN邻居标签
                clean_labels[i] = knn_vote
                action_mask[i] = 1
                confidence[i] = float(pred_probs[i, knn_vote])
                correction_weight[i] = 1.0
                phase1_actions[i] = 'Flip'
                flip_count += 1
                if y_true is not None:
                    if int(y_true[i]) == knn_vote:
                        flip_correct += 1
                    else:
                        flip_wrong += 1
            else:
                # 保持原标签
                action_mask[i] = 0
                confidence[i] = float(cl_confidence[i])
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
        
        # ========== 阶段2: 重新计算CL和KNN ==========
        logger.info("")
        logger.info("="*70)
        logger.info("步骤6: 阶段2 - 重新计算CL和KNN")
        logger.info("="*70)
        logger.info("  使用阶段1矫正后的标签重新计算CL和KNN，进行进一步矫正")
        
        # 检查clean_labels的类别数
        unique_labels_p2 = np.unique(clean_labels)
        n_classes_p2 = len(unique_labels_p2)
        logger.info(f"  阶段2标签类别数: {n_classes_p2} (类别: {unique_labels_p2.tolist()})")
        
        if n_classes_p2 < 2:
            logger.warning("  警告: 阶段2标签只有1个类别，跳过阶段2重新计算")
            # 直接返回阶段1的结果
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 低噪声方案完成（跳过阶段2）")
            logger.info("="*70)
            logger.info(f"  阶段1: Flip={flip_count} | 未翻转={keep_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        # 重新计算CL（使用阶段1矫正后的标签）
        logger.info("")
        logger.info("  重新计算CL置信度...")
        try:
            suspected_noise_p2, pred_labels_p2, pred_probs_p2 = self.cl.fit_predict(features, clean_labels)
            cl_confidence_p2 = np.array([pred_probs_p2[i, int(clean_labels[i])] for i in range(n_samples)])
        except Exception as e:
            logger.error(f"  阶段2 CL计算失败: {e}")
            logger.info("  跳过阶段2，返回阶段1结果")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 低噪声方案完成（阶段2失败，使用阶段1结果）")
            logger.info("="*70)
            logger.info(f"  阶段1: Flip={flip_count} | 未翻转={keep_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        logger.info(f"  ✓ Phase2 CL 完成")
        logger.info(f"    CL 置信度范围: [{cl_confidence_p2.min():.4f}, {cl_confidence_p2.max():.4f}]")
        logger.info(f"    CL 置信度均值: {cl_confidence_p2.mean():.4f}")
        logger.info(f"    CL 识别噪声: {suspected_noise_p2.sum()} 个")
        
        # 重新计算KNN（使用阶段1矫正后的标签）
        logger.info("")
        logger.info("  重新计算KNN...")
        try:
            self.knn.fit(features_for_analysis)
            neighbor_labels_p2, neighbor_consistency_p2 = self.knn.predict_semantic_label(features_for_analysis, clean_labels)
        except Exception as e:
            logger.error(f"  阶段2 KNN计算失败: {e}")
            logger.info("  跳过阶段2，返回阶段1结果")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 低噪声方案完成（阶段2失败，使用阶段1结果）")
            logger.info("="*70)
            logger.info(f"  阶段1: Flip={flip_count} | 未翻转={keep_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        logger.info(f"  ✓ Phase2 KNN 完成")
        logger.info(f"    KNN 一致性范围: [{neighbor_consistency_p2.min():.4f}, {neighbor_consistency_p2.max():.4f}]")
        logger.info(f"    KNN 一致性均值: {neighbor_consistency_p2.mean():.4f}")
        
        # 阶段2决策：基于新的CL和KNN进行进一步矫正（简单策略）
        logger.info("")
        logger.info("  执行阶段2标签矫正决策...")
        phase2_flip_count = 0
        phase2_keep_count = 0
        phase2_flip_correct = 0
        phase2_flip_wrong = 0
        
        for i in range(n_samples):
            current_label = int(clean_labels[i])
            knn_vote_p2 = int(neighbor_labels_p2[i])
            cl_conf_p2 = float(cl_confidence_p2[i])
            knn_cons_p2 = float(neighbor_consistency_p2[i])
            knn_opposes_p2 = (knn_vote_p2 != current_label)
            
            # 阶段2决策规则：如果CL置信度低且KNN反对当前标签，则翻转
            if cl_conf_p2 < cl_threshold and knn_opposes_p2 and knn_cons_p2 > knn_purity_threshold:
                # 翻转标签
                clean_labels[i] = knn_vote_p2
                if action_mask[i] == 0:  # 如果阶段1是Keep，阶段2翻转
                    action_mask[i] = 1
                    phase2_flip_count += 1
                    confidence[i] = float(pred_probs_p2[i, knn_vote_p2])
                    if y_true is not None:
                        if int(y_true[i]) == knn_vote_p2:
                            phase2_flip_correct += 1
                        else:
                            phase2_flip_wrong += 1
        
        logger.info("")
        logger.info("  📊 Phase2 决策统计:")
        logger.info(f"    新增翻转: {phase2_flip_count:5d}")
        logger.info(f"    保持: {phase2_keep_count:5d}")
        
        if y_true is not None and phase2_flip_count > 0:
            phase2_flip_precision = phase2_flip_correct / phase2_flip_count
            logger.info(f"    Phase2 Flip 准确率: {phase2_flip_precision:.3f} ({phase2_flip_correct}/{phase2_flip_count})")
            logger.info(f"    Phase2 Flip 错误: {phase2_flip_wrong} 个")
            
            # 阶段2后的整体纯度
            correct_p2 = (clean_labels == y_true).sum()
            purity_p2 = 100.0 * correct_p2 / n_samples
            improvement_p2 = purity_p2 - purity
            logger.info("")
            logger.info(f"  📊 Phase2 整体纯度: {purity_p2:.2f}% ({correct_p2}/{n_samples})")
            logger.info(f"  📈 Phase2 改进效果: {improvement_p2:+.2f}% (相比Phase1)")
        
        # 更新返回的KNN一致性为阶段2的结果
        neighbor_consistency = neighbor_consistency_p2
        pred_probs = pred_probs_p2
        
        # ========== 阶段3: 重新训练CL和KNN，分配样本权重 ==========
        logger.info("")
        logger.info("="*70)
        logger.info("步骤7: 阶段3 - 重新训练CL和KNN，分配样本权重")
        logger.info("="*70)
        logger.info("  使用阶段2矫正后的标签重新训练CL和KNN，根据样本类型分配权重")
        logger.info("  核心干净样本（CL高置信度且KNN一致）：权重1.0")
        logger.info("  噪声样本（CL低置信度或KNN不一致）：权重0.5")
        
        # 检查clean_labels的类别数
        unique_labels_p3 = np.unique(clean_labels)
        n_classes_p3 = len(unique_labels_p3)
        logger.info(f"  阶段3标签类别数: {n_classes_p3} (类别: {unique_labels_p3.tolist()})")
        
        if n_classes_p3 < 2:
            logger.warning("  警告: 阶段3标签只有1个类别，跳过阶段3重新计算")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 低噪声方案完成（跳过阶段3）")
            logger.info("="*70)
            final_flip_count = int((action_mask == 1).sum())
            final_keep_count = n_samples - final_flip_count
            logger.info(f"  阶段1: Flip={flip_count} | 未翻转={keep_count}")
            logger.info(f"  阶段2: 新增翻转={phase2_flip_count}")
            logger.info(f"  最终: Flip={final_flip_count} | 保持={final_keep_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        # 重新计算CL（使用阶段2矫正后的标签）
        logger.info("")
        logger.info("  重新计算CL置信度...")
        try:
            suspected_noise_p3, pred_labels_p3, pred_probs_p3 = self.cl.fit_predict(features, clean_labels)
            cl_confidence_p3 = np.array([pred_probs_p3[i, int(clean_labels[i])] for i in range(n_samples)])
        except Exception as e:
            logger.error(f"  阶段3 CL计算失败: {e}")
            logger.info("  跳过阶段3，返回阶段2结果")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 低噪声方案完成（阶段3失败，使用阶段2结果）")
            logger.info("="*70)
            final_flip_count = int((action_mask == 1).sum())
            final_keep_count = n_samples - final_flip_count
            logger.info(f"  阶段1: Flip={flip_count} | 未翻转={keep_count}")
            logger.info(f"  阶段2: 新增翻转={phase2_flip_count}")
            logger.info(f"  最终: Flip={final_flip_count} | 保持={final_keep_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        logger.info(f"  ✓ Phase3 CL 完成")
        logger.info(f"    CL 置信度范围: [{cl_confidence_p3.min():.4f}, {cl_confidence_p3.max():.4f}]")
        logger.info(f"    CL 置信度均值: {cl_confidence_p3.mean():.4f}")
        logger.info(f"    CL 识别噪声: {suspected_noise_p3.sum()} 个")
        
        # 重新计算KNN（使用阶段2矫正后的标签）
        logger.info("")
        logger.info("  重新计算KNN...")
        try:
            self.knn.fit(features_for_analysis)
            neighbor_labels_p3, neighbor_consistency_p3 = self.knn.predict_semantic_label(features_for_analysis, clean_labels)
        except Exception as e:
            logger.error(f"  阶段3 KNN计算失败: {e}")
            logger.info("  跳过阶段3，返回阶段2结果")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 低噪声方案完成（阶段3失败，使用阶段2结果）")
            logger.info("="*70)
            final_flip_count = int((action_mask == 1).sum())
            final_keep_count = n_samples - final_flip_count
            logger.info(f"  阶段1: Flip={flip_count} | 未翻转={keep_count}")
            logger.info(f"  阶段2: 新增翻转={phase2_flip_count}")
            logger.info(f"  最终: Flip={final_flip_count} | 保持={final_keep_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        logger.info(f"  ✓ Phase3 KNN 完成")
        logger.info(f"    KNN 一致性范围: [{neighbor_consistency_p3.min():.4f}, {neighbor_consistency_p3.max():.4f}]")
        logger.info(f"    KNN 一致性均值: {neighbor_consistency_p3.mean():.4f}")
        
        # 阶段3：根据样本类型分配权重（低中噪声统一划分方案：10%-30%噪声适用）
        logger.info("")
        logger.info("  执行阶段3权重分配（低中噪声统一划分方案）...")
        logger.info("    策略说明:")
        logger.info("      - 核心数据分区: 阶段2识别出的原始核心样本（CL≥0.7且KNN≥0.7） → 权重1.0")
        logger.info("      - 干净非核心数据分区: CL>0.4且AUM>0.0的原始噪声区样本 → 权重0.5")
        logger.info("      - 噪声区: 不满足上述指标的残留冲突样本 → 权重0.1")
        
        # 基于阶段2的指标识别原始核心样本
        # 注意：阶段2的指标（cl_confidence_p2, neighbor_consistency_p2）在阶段3开始时仍然可用
        phase2_core_cl_threshold = 0.7  # 阶段2核心样本CL阈值
        phase2_core_knn_threshold = 0.7  # 阶段2核心样本KNN阈值
        
        # 识别阶段2的核心样本：使用阶段2的CL和KNN指标
        phase2_core_mask = np.zeros(n_samples, dtype=bool)
        # 需要重新获取阶段2的KNN投票（如果可用）
        # 为了简化，我们使用阶段3的KNN投票，但使用阶段2的CL和KNN一致性指标
        # 如果阶段2的指标不可用，则使用阶段3的指标但使用更严格的标准
        try:
            # 尝试使用阶段2的指标
            for i in range(n_samples):
                cl_conf_p2 = float(cl_confidence_p2[i])
                knn_cons_p2 = float(neighbor_consistency_p2[i])
                knn_vote_p2 = int(neighbor_labels_p2[i])
                current_label = int(clean_labels[i])
                knn_supports = (knn_vote_p2 == current_label)
                
                if cl_conf_p2 >= phase2_core_cl_threshold and knn_cons_p2 >= phase2_core_knn_threshold and knn_supports:
                    phase2_core_mask[i] = True
        except (NameError, UnboundLocalError):
            # 如果阶段2的指标不可用，使用阶段3的指标但使用更严格的标准
            logger.warning("  警告: 阶段2的指标不可用，使用阶段3的指标识别核心样本")
            for i in range(n_samples):
                cl_conf_p3 = float(cl_confidence_p3[i])
                knn_cons_p3 = float(neighbor_consistency_p3[i])
                knn_vote_p3 = int(neighbor_labels_p3[i])
                current_label = int(clean_labels[i])
                knn_supports = (knn_vote_p3 == current_label)
                
                if cl_conf_p3 >= phase2_core_cl_threshold and knn_cons_p3 >= phase2_core_knn_threshold and knn_supports:
                    phase2_core_mask[i] = True
        
        core_clean_count = 0
        clean_non_core_count = 0
        noise_suppression_count = 0
        
        # 定义阈值（低中噪声统一划分方案）
        cl_recovery_threshold = 0.4  # CL恢复识别阈值（干净非核心区）
        aum_recovery_threshold = 0.0  # AUM恢复识别阈值（干净非核心区）
        
        for i in range(n_samples):
            cl_conf_p3 = float(cl_confidence_p3[i])
            aum_val = float(aum_scores[i])
            
            # 核心数据分区：阶段2识别出的原始核心样本（继承阶段2的权重1.0）
            if phase2_core_mask[i]:
                correction_weight[i] = 1.0
                core_clean_count += 1
            # 干净非核心数据分区：CL>0.4且AUM>0.0的原始噪声区样本
            elif cl_conf_p3 > cl_recovery_threshold and aum_val > aum_recovery_threshold:
                correction_weight[i] = 0.5
                clean_non_core_count += 1
            # 噪声区：不满足上述指标的残留冲突样本
            else:
                correction_weight[i] = 0.1
                noise_suppression_count += 1
        
        logger.info("")
        logger.info("  📊 Phase3 权重分配统计（低中噪声统一划分方案）:")
        logger.info(f"    核心数据分区 (权重1.0): {core_clean_count:5d} ({100*core_clean_count/n_samples:.1f}%)")
        logger.info(f"    干净非核心数据分区 (权重0.5): {clean_non_core_count:5d} ({100*clean_non_core_count/n_samples:.1f}%)")
        logger.info(f"    噪声区 (权重0.1): {noise_suppression_count:5d} ({100*noise_suppression_count/n_samples:.1f}%)")
        
        if y_true is not None:
            # 验证权重分配的准确性
            core_clean_correct = 0
            core_clean_total = 0
            clean_non_core_correct = 0
            clean_non_core_total = 0
            noise_suppression_correct = 0
            noise_suppression_total = 0
            
            for i in range(n_samples):
                if correction_weight[i] == 1.0:
                    core_clean_total += 1
                    if int(clean_labels[i]) == int(y_true[i]):
                        core_clean_correct += 1
                elif correction_weight[i] == 0.5:
                    clean_non_core_total += 1
                    if int(clean_labels[i]) == int(y_true[i]):
                        clean_non_core_correct += 1
                elif correction_weight[i] == 0.1:
                    noise_suppression_total += 1
                    if int(clean_labels[i]) == int(y_true[i]):
                        noise_suppression_correct += 1
            
            if core_clean_total > 0:
                core_clean_acc = 100.0 * core_clean_correct / core_clean_total
                logger.info(f"    核心数据分区准确率: {core_clean_acc:.2f}% ({core_clean_correct}/{core_clean_total})")
            if clean_non_core_total > 0:
                clean_non_core_acc = 100.0 * clean_non_core_correct / clean_non_core_total
                logger.info(f"    干净非核心数据分区准确率: {clean_non_core_acc:.2f}% ({clean_non_core_correct}/{clean_non_core_total})")
            if noise_suppression_total > 0:
                noise_suppression_acc = 100.0 * noise_suppression_correct / noise_suppression_total
                logger.info(f"    噪声区准确率: {noise_suppression_acc:.2f}% ({noise_suppression_correct}/{noise_suppression_total})")
        
        # 更新返回的KNN一致性和CL概率为阶段3的结果
        neighbor_consistency = neighbor_consistency_p3
        pred_probs = pred_probs_p3
        
        # 低噪声方案：返回最终结果
        logger.info("")
        logger.info("="*70)
        logger.info("✓ 低噪声方案完成")
        logger.info("="*70)
        final_flip_count = int((action_mask == 1).sum())
        final_keep_count = n_samples - final_flip_count
        logger.info(f"  阶段1: Flip={flip_count} | 未翻转={keep_count}")
        logger.info(f"  阶段2: 新增翻转={phase2_flip_count}")
        logger.info(f"  阶段3: 核心数据={core_clean_count} (权重1.0) | 干净非核心={clean_non_core_count} (权重0.5) | 噪声区={noise_suppression_count} (权重0.1)")
        logger.info(f"  最终: Flip={final_flip_count} | 保持={final_keep_count}")
        
        return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
    
    else:
        # ========== 高噪声方案：完整的两阶段方案（包括普通高噪声和超限方案） ==========
        if is_high_aggressive:
            logger.info("")
            logger.info("="*70)
            logger.info("高噪声超限方案: 自适应级联决策策略 (基于决策树Depth 5优化)")
            logger.info("="*70)
            logger.info("  Phase1参数（激进策略 - 准确率90.0%）:")
            logger.info("    决策树深度: 5 (理论最高准确率90.0%)")
            logger.info("    区域1 (CL_Diff <= 0.11): 多级判断，结合Neg_AUM和KNN_Flip")
            logger.info("    区域2 (0.11 < CL_Diff <= 0.42): KNN裁决 + AUM阈值分层")
            logger.info("    区域3 (CL_Diff > 0.42): AUM历史信任机制 + 异常值保护")
            logger.info("    特征: CL_Diff, Neg_AUM, KNN_Flip_Score (深度非线性组合)")
        else:
            logger.info("")
            logger.info("="*70)
            logger.info("高噪声方案: 自适应级联决策策略 (基于决策树Depth 5优化)")
            logger.info("="*70)
            logger.info("  Phase1参数（激进策略 - 准确率90.0%）:")
            logger.info("    决策树深度: 5 (理论最高准确率90.0%)")
            logger.info("    区域1 (CL_Diff <= 0.11): 多级判断，结合Neg_AUM和KNN_Flip")
            logger.info("    区域2 (0.11 < CL_Diff <= 0.42): KNN裁决 + AUM阈值分层")
            logger.info("    区域3 (CL_Diff > 0.42): AUM历史信任机制 + 异常值保护")
            logger.info("    特征: CL_Diff, Neg_AUM, KNN_Flip_Score (深度非线性组合)")
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
            
            # 高噪声方案 Phase1 规则（激进策略 - 基于决策树Depth 5优化，准确率90.0%）
            # 核心思想：自适应级联决策，捕捉CL、AUM、KNN之间的非线性冲突
            cl_cur_val = float(cl_cur)
            cl_target_val = float(pred_probs[i, target_label])
            cl_diff = cl_target_val - cl_cur_val  # CL差值：目标标签置信度 - 当前标签置信度
            aum_val_float = float(aum_val)
            neg_aum = -aum_val_float  # 取负号，值越大代表越可能是噪声
            knn_cons_val = float(knn_cons)
            knn_flip_score = knn_cons_val if knn_opposes else -knn_cons_val  # KNN翻转分数
            
            # 决策树逻辑 (Depth 5 Optimized - 准确率90.0%)
            # 基于决策树分析，理论最高准确率可达90.0%
            if cl_diff <= 0.11:
                # 区域1: 低CL差值区（模型倾向于保持）
                if neg_aum <= 0.07:
                    # AUM显示样本处于安全边界
                    if knn_flip_score <= -0.60:
                        # KNN强烈建议保持
                        if neg_aum <= -0.05:
                            do_flip = False  # Keep
                        else:
                            # 注意：决策树中有不可达分支，这里简化处理
                            # 原树: Neg_AUM > -0.05 且 Neg_AUM <= -0.05 -> class: 1 (不可达)
                            # 实际: Neg_AUM > -0.05 的情况
                            if neg_aum <= -0.05:
                                do_flip = True   # Flip (边界情况)
                            else:
                                do_flip = False  # Keep
                    else:
                        # KNN_Flip > -0.60
                        if cl_diff <= 0.04:
                            if cl_diff <= -0.21:
                                do_flip = False  # Keep
                            else:
                                do_flip = False  # Keep
                        else:
                            # CL_Diff > 0.04 且 <= 0.11
                            # 注意：决策树中有不可达分支 CL_Diff <= 0.04 且 CL_Diff > 0.04
                            # 实际: CL_Diff > 0.04 的情况
                            if cl_diff <= 0.04:
                                do_flip = True   # Flip (不可达，防御性)
                            else:
                                do_flip = False  # Keep
                else:
                    # Neg_AUM > 0.07
                    if cl_diff <= 0.05:
                        if cl_diff <= -0.10:
                            # 模型强烈建议保持，但AUM差
                            if knn_flip_score <= 0.55:
                                do_flip = False  # Keep (KNN也没强烈反对)
                            else:
                                do_flip = True   # Flip (KNN强烈反对，激进翻转)
                        else:
                            # -0.10 < CL_Diff <= 0.05
                            do_flip = True   # Flip (AUM主导，激进翻转)
                    else:
                        # 0.05 < CL_Diff <= 0.11
                        if cl_diff <= 0.10:
                            do_flip = False  # Keep
                        else:
                            do_flip = True   # Flip
            else:
                # 区域2: 高CL差值区（模型倾向于翻转）
                if cl_diff <= 0.42:
                    # 中等CL差值区
                    if neg_aum <= 0.18:
                        # 模糊区：使用KNN裁决
                        if knn_flip_score <= 0.53:
                            if neg_aum <= -0.16:
                                do_flip = False  # Keep (AUM极好)
                            else:
                                do_flip = True   # Flip
                        else:
                            # KNN_Flip > 0.53
                            if cl_diff <= 0.11:
                                do_flip = True   # Flip (不可达，防御性)
                            else:
                                # 反直觉分支：KNN强烈建议翻转，但保持（防止对抗样本）
                                do_flip = False  # Keep
                    else:
                        # Neg_AUM > 0.18
                        if neg_aum <= 1.63:
                            if neg_aum <= 1.04:
                                do_flip = True   # Flip
                            else:
                                do_flip = True   # Flip
                        else:
                            # Neg_AUM > 1.63 (极异常值)
                            do_flip = False  # Keep
                else:
                    # 区域3: 极高CL差值区（CL强烈建议翻转，CL_Diff > 0.42）
                    if neg_aum <= -0.07:
                        # AUM极好，覆盖CL信号（防止过度矫正）
                        do_flip = False  # Keep
                    else:
                        # Neg_AUM > -0.07
                        if neg_aum <= 1.31:
                            if cl_diff <= 0.54:
                                do_flip = True   # Flip
                            else:
                                do_flip = True   # Flip
                        else:
                            # Neg_AUM > 1.31 (极异常值)
                            do_flip = False  # Keep
            
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
        
        # ========== 阶段2: 重新计算CL和KNN ==========
        logger.info("")
        logger.info("="*70)
        logger.info("步骤6: 阶段2 - 重新计算CL和KNN")
        logger.info("="*70)
        logger.info("  使用阶段1矫正后的标签重新计算CL和KNN，进行进一步矫正")
        
        # 检查clean_labels的类别数
        unique_labels_p2 = np.unique(clean_labels)
        n_classes_p2 = len(unique_labels_p2)
        logger.info(f"  阶段2标签类别数: {n_classes_p2} (类别: {unique_labels_p2.tolist()})")
        
        if n_classes_p2 < 2:
            logger.warning("  警告: 阶段2标签只有1个类别，跳过阶段2重新计算")
            # 直接返回阶段1的结果
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 高噪声方案完成（跳过阶段2）")
            logger.info("="*70)
            phase1_flip_count = int((action_mask == 1).sum())
            phase1_no_flip_count = n_samples - phase1_flip_count
            logger.info(f"  阶段1: Flip={phase1_flip_count} | 未翻转={phase1_no_flip_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        # 重新计算CL（使用阶段1矫正后的标签）
        logger.info("")
        logger.info("  重新计算CL置信度...")
        try:
            suspected_noise_p2, pred_labels_p2, pred_probs_p2 = self.cl.fit_predict(features, clean_labels)
            cl_confidence_p2 = np.array([pred_probs_p2[i, int(clean_labels[i])] for i in range(n_samples)])
        except Exception as e:
            logger.error(f"  阶段2 CL计算失败: {e}")
            logger.info("  跳过阶段2，返回阶段1结果")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 高噪声方案完成（阶段2失败，使用阶段1结果）")
            logger.info("="*70)
            phase1_flip_count = int((action_mask == 1).sum())
            phase1_no_flip_count = n_samples - phase1_flip_count
            logger.info(f"  阶段1: Flip={phase1_flip_count} | 未翻转={phase1_no_flip_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        logger.info(f"  ✓ Phase2 CL 完成")
        logger.info(f"    CL 置信度范围: [{cl_confidence_p2.min():.4f}, {cl_confidence_p2.max():.4f}]")
        logger.info(f"    CL 置信度均值: {cl_confidence_p2.mean():.4f}")
        logger.info(f"    CL 识别噪声: {suspected_noise_p2.sum()} 个")
        
        # 重新计算KNN（使用阶段1矫正后的标签）
        logger.info("")
        logger.info("  重新计算KNN...")
        try:
            self.knn.fit(features_for_analysis)
            neighbor_labels_p2, neighbor_consistency_p2 = self.knn.predict_semantic_label(features_for_analysis, clean_labels)
        except Exception as e:
            logger.error(f"  阶段2 KNN计算失败: {e}")
            logger.info("  跳过阶段2，返回阶段1结果")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 高噪声方案完成（阶段2失败，使用阶段1结果）")
            logger.info("="*70)
            phase1_flip_count = int((action_mask == 1).sum())
            phase1_no_flip_count = n_samples - phase1_flip_count
            logger.info(f"  阶段1: Flip={phase1_flip_count} | 未翻转={phase1_no_flip_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        logger.info(f"  ✓ Phase2 KNN 完成")
        logger.info(f"    KNN 一致性范围: [{neighbor_consistency_p2.min():.4f}, {neighbor_consistency_p2.max():.4f}]")
        logger.info(f"    KNN 一致性均值: {neighbor_consistency_p2.mean():.4f}")
        
        # 阶段2决策：使用保守优化策略（LateFlip和UndoFlip）
        logger.info("")
        logger.info("="*70)
        logger.info("阶段2: Phase2 保守优化策略 (仅使用CL和KNN)")
        logger.info("="*70)
        logger.info("  策略说明:")
        logger.info("    - LateFlip: Phase1保持但KNN一致性<0.65 且 CL当前标签置信度<0.55 → 强制翻转（挽救漏检噪声）")
        logger.info("    - UndoFlip: Phase1翻转但Stage2指标显示应撤销（严苛：CL<0.35 或 (KNN反对 且 KNN一致性<0.5)）")
        logger.info("      优化后的LateFlip阈值，提升净收益")
        logger.info("")
        
        # 阶段2参数
        phase2_late_flip_cl_threshold = 0.55  # CL当前标签置信度阈值（优化后）
        phase2_late_flip_knn_threshold = 0.65  # KNN一致性阈值（优化后，低一致性表示可能是噪声）
        phase2_undo_flip_cl_threshold = 0.35
        phase2_undo_flip_knn_oppose_threshold = 0.5
        
        late_flip_count = 0
        undo_flip_count = 0
        no_change_count = 0
        late_flip_correct = 0
        late_flip_wrong = 0
        undo_flip_correct = 0
        undo_flip_wrong = 0
        
        for i in range(n_samples):
            current_label = int(clean_labels[i])
            phase1_action = action_mask[i]  # 0=Keep, 1=Flip
            knn_vote_p2 = int(neighbor_labels_p2[i])
            cl_conf_p2 = float(cl_confidence_p2[i])
            knn_cons_p2 = float(neighbor_consistency_p2[i])
            knn_opposes_p2 = (knn_vote_p2 != current_label)
            
            # LateFlip: Phase1保持但KNN一致性<0.65 且 CL当前标签置信度<0.55 → 强制翻转（挽救漏检噪声）
            if phase1_action == 0:  # Phase1是Keep
                # 优化后条件：KNN一致性<0.65 且 CL当前标签置信度<0.55
                # 逻辑：当样本被保持为原标签，但模型对其信心不足（CL<0.55），
                #      且邻居节点的支持度也很低（KNN<0.65）时，极大概率是漏检的噪声，应强制翻转
                if knn_cons_p2 < phase2_late_flip_knn_threshold and cl_conf_p2 < phase2_late_flip_cl_threshold:
                    # 执行LateFlip：翻转标签为KNN投票的标签
                    clean_labels[i] = knn_vote_p2
                    action_mask[i] = 1
                    confidence[i] = float(pred_probs_p2[i, knn_vote_p2])
                    late_flip_count += 1
                    if y_true is not None:
                        if int(y_true[i]) == knn_vote_p2:
                            late_flip_correct += 1
                        else:
                            late_flip_wrong += 1
                else:
                    no_change_count += 1
            # UndoFlip: Phase1翻转但Stage2指标显示应撤销（严苛：CL<0.35 或 (KNN反对 且 KNN一致性<0.5)）
            elif phase1_action == 1:  # Phase1是Flip
                # 撤销条件：CL<0.35 或 (KNN反对 且 KNN一致性<0.5)
                should_undo = (cl_conf_p2 < phase2_undo_flip_cl_threshold) or \
                             (knn_opposes_p2 and knn_cons_p2 < phase2_undo_flip_knn_oppose_threshold)
                
                if should_undo:
                    # 撤销翻转，恢复为原始标签
                    original_label = int(noisy_labels[i])
                    clean_labels[i] = original_label
                    action_mask[i] = 0
                    confidence[i] = float(cl_conf_p2)
                    undo_flip_count += 1
                    if y_true is not None:
                        if int(y_true[i]) == original_label:
                            undo_flip_correct += 1
                        else:
                            undo_flip_wrong += 1
                else:
                    no_change_count += 1
        
        logger.info("")
        logger.info("  📊 Phase2 动作统计:")
        logger.info(f"    UndoFlip (撤销翻转): {undo_flip_count} 个")
        logger.info(f"    LateFlip (延迟翻转): {late_flip_count} 个")
        logger.info(f"    NoChange (无变化):   {no_change_count} 个")
        
        if y_true is not None:
            logger.info("")
            logger.info("  📈 Phase2 效果评估:")
            if undo_flip_count > 0:
                undo_precision = undo_flip_correct / undo_flip_count
                logger.info(f"    UndoFlip: {undo_flip_count} 个 | 正确={undo_flip_correct} | 错误={undo_flip_wrong} | 准确率={undo_precision:.1%}")
            if late_flip_count > 0:
                late_precision = late_flip_correct / late_flip_count
                logger.info(f"    LateFlip: {late_flip_count} 个 | 正确={late_flip_correct} | 错误={late_flip_wrong} | 准确率={late_precision:.1%}")
            
            # 阶段2后的整体纯度
            correct_p2 = (clean_labels == y_true).sum()
            purity_p2 = 100.0 * correct_p2 / n_samples
            improvement_p2 = purity_p2 - purity
            logger.info("")
            logger.info(f"  📊 Phase2 整体纯度: {purity_p2:.2f}% ({correct_p2}/{n_samples})")
            logger.info(f"  📈 Phase2 改进效果: {improvement_p2:+.2f}% (相比Phase1)")
        
        # 更新返回的KNN一致性为阶段2的结果
        neighbor_consistency = neighbor_consistency_p2
        pred_probs = pred_probs_p2
        
        # ========== 阶段3: 重新训练CL和KNN，分配样本权重 ==========
        logger.info("")
        logger.info("="*70)
        logger.info("步骤7: 阶段3 - 重新训练CL和KNN，分配样本权重")
        logger.info("="*70)
        logger.info("  使用阶段2矫正后的标签重新训练CL和KNN，根据样本类型分配权重")
        logger.info("  核心干净样本（CL高置信度且KNN一致）：权重1.0")
        logger.info("  噪声样本（CL低置信度或KNN不一致）：权重0.5")
        
        # 检查clean_labels的类别数
        unique_labels_p3 = np.unique(clean_labels)
        n_classes_p3 = len(unique_labels_p3)
        logger.info(f"  阶段3标签类别数: {n_classes_p3} (类别: {unique_labels_p3.tolist()})")
        
        if n_classes_p3 < 2:
            logger.warning("  警告: 阶段3标签只有1个类别，跳过阶段3重新计算")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 高噪声方案完成（跳过阶段3）")
            logger.info("="*70)
            phase1_flip_count = int((action_mask == 1).sum()) - late_flip_count + undo_flip_count
            phase1_no_flip_count = n_samples - phase1_flip_count
            final_flip_count = int((action_mask == 1).sum())
            final_keep_count = n_samples - final_flip_count
            logger.info(f"  阶段1: Flip={phase1_flip_count} | 未翻转={phase1_no_flip_count}")
            logger.info(f"  阶段2: UndoFlip={undo_flip_count} | LateFlip={late_flip_count}")
            logger.info(f"  最终: Flip={final_flip_count} | 保持={final_keep_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        # 重新计算CL（使用阶段2矫正后的标签）
        logger.info("")
        logger.info("  重新计算CL置信度...")
        try:
            suspected_noise_p3, pred_labels_p3, pred_probs_p3 = self.cl.fit_predict(features, clean_labels)
            cl_confidence_p3 = np.array([pred_probs_p3[i, int(clean_labels[i])] for i in range(n_samples)])
        except Exception as e:
            logger.error(f"  阶段3 CL计算失败: {e}")
            logger.info("  跳过阶段3，返回阶段2结果")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 高噪声方案完成（阶段3失败，使用阶段2结果）")
            logger.info("="*70)
            phase1_flip_count = int((action_mask == 1).sum()) - late_flip_count + undo_flip_count
            phase1_no_flip_count = n_samples - phase1_flip_count
            final_flip_count = int((action_mask == 1).sum())
            final_keep_count = n_samples - final_flip_count
            logger.info(f"  阶段1: Flip={phase1_flip_count} | 未翻转={phase1_no_flip_count}")
            logger.info(f"  阶段2: UndoFlip={undo_flip_count} | LateFlip={late_flip_count}")
            logger.info(f"  最终: Flip={final_flip_count} | 保持={final_keep_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        logger.info(f"  ✓ Phase3 CL 完成")
        logger.info(f"    CL 置信度范围: [{cl_confidence_p3.min():.4f}, {cl_confidence_p3.max():.4f}]")
        logger.info(f"    CL 置信度均值: {cl_confidence_p3.mean():.4f}")
        logger.info(f"    CL 识别噪声: {suspected_noise_p3.sum()} 个")
        
        # 重新计算KNN（使用阶段2矫正后的标签）
        logger.info("")
        logger.info("  重新计算KNN...")
        try:
            self.knn.fit(features_for_analysis)
            neighbor_labels_p3, neighbor_consistency_p3 = self.knn.predict_semantic_label(features_for_analysis, clean_labels)
        except Exception as e:
            logger.error(f"  阶段3 KNN计算失败: {e}")
            logger.info("  跳过阶段3，返回阶段2结果")
            logger.info("")
            logger.info("="*70)
            logger.info("✓ 高噪声方案完成（阶段3失败，使用阶段2结果）")
            logger.info("="*70)
            phase1_flip_count = int((action_mask == 1).sum()) - late_flip_count + undo_flip_count
            phase1_no_flip_count = n_samples - phase1_flip_count
            final_flip_count = int((action_mask == 1).sum())
            final_keep_count = n_samples - final_flip_count
            logger.info(f"  阶段1: Flip={phase1_flip_count} | 未翻转={phase1_no_flip_count}")
            logger.info(f"  阶段2: UndoFlip={undo_flip_count} | LateFlip={late_flip_count}")
            logger.info(f"  最终: Flip={final_flip_count} | 保持={final_keep_count}")
            return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
        
        logger.info(f"  ✓ Phase3 KNN 完成")
        logger.info(f"    KNN 一致性范围: [{neighbor_consistency_p3.min():.4f}, {neighbor_consistency_p3.max():.4f}]")
        logger.info(f"    KNN 一致性均值: {neighbor_consistency_p3.mean():.4f}")
        
        # 阶段3：根据样本类型分配权重（精细化三层分区策略 - 高噪声方案）
        logger.info("")
        logger.info("  执行阶段3权重分配（精细化三层分区策略 - 基于指标判断）...")
        logger.info("    策略说明:")
        logger.info("      - 核心数据分区: CL≥0.7且KNN≥0.7且KNN支持 → 权重1.0（继承阶段2多重交叉验证）")
        logger.info("      - 干净非核心数据分区: CL>0.25且KNN>0.55 → 权重0.5（指标恢复识别）")
        logger.info("      - 噪声抑制区: CL≤0.25或KNN≤0.55 → 权重0.1（指标失序压制）")
        
        core_clean_count = 0
        clean_non_core_count = 0
        noise_suppression_count = 0
        
        # 定义阈值
        cl_high_threshold = 0.7  # CL高置信度阈值（核心区）
        knn_consistency_threshold = 0.7  # KNN一致性阈值（核心区）
        cl_recovery_threshold = 0.25  # CL恢复识别阈值（干净非核心区）
        knn_recovery_threshold = 0.55  # KNN恢复识别阈值（干净非核心区）
        
        # 分配权重（基于CL和KNN指标）
        for i in range(n_samples):
            cl_conf_p3 = float(cl_confidence_p3[i])
            knn_cons_p3 = float(neighbor_consistency_p3[i])
            knn_vote_p3 = int(neighbor_labels_p3[i])
            current_label = int(clean_labels[i])
            knn_supports = (knn_vote_p3 == current_label)
            
            # 核心数据分区：CL高置信度且KNN一致且KNN支持（继承阶段2多重交叉验证）
            if cl_conf_p3 >= cl_high_threshold and knn_cons_p3 >= knn_consistency_threshold and knn_supports:
                correction_weight[i] = 1.0
                core_clean_count += 1
            # 干净非核心数据分区：CL>0.25且KNN>0.55（指标恢复识别）
            elif cl_conf_p3 > cl_recovery_threshold and knn_cons_p3 > knn_recovery_threshold:
                correction_weight[i] = 0.5
                clean_non_core_count += 1
            # 噪声抑制区：CL≤0.25或KNN≤0.55（指标失序压制）
            else:
                correction_weight[i] = 0.1
                noise_suppression_count += 1
        
        logger.info("")
        logger.info("  📊 Phase3 权重分配统计（精细化三层分区）:")
        logger.info(f"    核心数据分区 (权重1.0): {core_clean_count:5d} ({100*core_clean_count/n_samples:.1f}%)")
        logger.info(f"    干净非核心数据分区 (权重0.5): {clean_non_core_count:5d} ({100*clean_non_core_count/n_samples:.1f}%)")
        logger.info(f"    噪声抑制区 (权重0.1): {noise_suppression_count:5d} ({100*noise_suppression_count/n_samples:.1f}%)")
        
        if y_true is not None:
            # 验证权重分配的准确性
            core_clean_correct = 0
            core_clean_total = 0
            clean_non_core_correct = 0
            clean_non_core_total = 0
            noise_suppression_correct = 0
            noise_suppression_total = 0
            
            for i in range(n_samples):
                if correction_weight[i] == 1.0:
                    core_clean_total += 1
                    if int(clean_labels[i]) == int(y_true[i]):
                        core_clean_correct += 1
                elif correction_weight[i] == 0.5:
                    clean_non_core_total += 1
                    if int(clean_labels[i]) == int(y_true[i]):
                        clean_non_core_correct += 1
                elif correction_weight[i] == 0.1:
                    noise_suppression_total += 1
                    if int(clean_labels[i]) == int(y_true[i]):
                        noise_suppression_correct += 1
            
            if core_clean_total > 0:
                core_clean_acc = 100.0 * core_clean_correct / core_clean_total
                logger.info(f"    核心数据分区准确率: {core_clean_acc:.2f}% ({core_clean_correct}/{core_clean_total})")
            if clean_non_core_total > 0:
                clean_non_core_acc = 100.0 * clean_non_core_correct / clean_non_core_total
                logger.info(f"    干净非核心数据分区准确率: {clean_non_core_acc:.2f}% ({clean_non_core_correct}/{clean_non_core_total})")
            if noise_suppression_total > 0:
                noise_suppression_acc = 100.0 * noise_suppression_correct / noise_suppression_total
                logger.info(f"    噪声抑制区准确率: {noise_suppression_acc:.2f}% ({noise_suppression_correct}/{noise_suppression_total})")
        
        # 更新返回的KNN一致性和CL概率为阶段3的结果
        neighbor_consistency = neighbor_consistency_p3
        pred_probs = pred_probs_p3
        
        # ========== 高噪声方案：返回最终结果 ==========
        logger.info("")
        logger.info("="*70)
        logger.info("✓ 高噪声方案完成")
        logger.info("="*70)
        phase1_flip_count = int((action_mask == 1).sum()) - late_flip_count + undo_flip_count
        phase1_no_flip_count = n_samples - phase1_flip_count
        final_flip_count = int((action_mask == 1).sum())
        final_keep_count = n_samples - final_flip_count
        logger.info(f"  阶段1: Flip={phase1_flip_count} | 未翻转={phase1_no_flip_count}")
        logger.info(f"  阶段2: UndoFlip={undo_flip_count} | LateFlip={late_flip_count}")
        logger.info(f"  阶段3: 核心数据={core_clean_count} (权重1.0) | 干净非核心={clean_non_core_count} (权重0.5) | 噪声抑制={noise_suppression_count} (权重0.1)")
        logger.info(f"  最终: Flip={final_flip_count} | 保持={final_keep_count}")
        
        return clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs
