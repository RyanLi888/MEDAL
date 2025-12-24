"""
Label Correction Analysis Script
Only runs label correction part and generates detailed analysis documents and feature distribution plots.

This script:
1. Loads dataset and injects label noise
2. Extracts features using pre-trained backbone
3. Runs Hybrid Court label correction (CL + MADE + KNN)
4. Generates detailed per-sample analysis document
5. Creates feature distribution plots for clean, noisy, and corrected data
"""
import sys
import os
from pathlib import Path
# Ensure project root is on sys.path so `MoudleCode.*` imports work when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import (
    set_seed, setup_logger, inject_label_noise
)
from MoudleCode.preprocessing.pcap_parser import load_dataset
from MoudleCode.feature_extraction.backbone import MicroBiMambaBackbone
from MoudleCode.label_correction.hybrid_court import HybridCourt

import logging

# 尝试导入Excel写入库
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("警告: openpyxl未安装，将只生成CSV文件。安装命令: pip install openpyxl")

# 简化的预处理文件检测和加载
def check_preprocessed_exists(split='train'):
    """检查预处理文件是否存在"""
    preprocessed_dir = PROJECT_ROOT / 'output' / 'preprocessed'
    if not preprocessed_dir.exists():
        return False
    
    X_file = preprocessed_dir / f'{split}_X.npy'
    y_file = preprocessed_dir / f'{split}_y.npy'
    files_file = preprocessed_dir / f'{split}_files.npy'
    
    return X_file.exists() and y_file.exists() and files_file.exists()

def load_preprocessed(split='train'):
    """加载预处理文件"""
    preprocessed_dir = PROJECT_ROOT / 'output' / 'preprocessed'
    
    X = np.load(preprocessed_dir / f'{split}_X.npy')
    y = np.load(preprocessed_dir / f'{split}_y.npy')
    files = np.load(preprocessed_dir / f'{split}_files.npy', allow_pickle=True)
    
    return X, y, files

logger = None  # Will be initialized in main()


def _configure_matplotlib_chinese_fonts():
    try:
        preferred = [
            'Noto Sans CJK SC',
            'Noto Sans CJK TC',
            'Noto Sans CJK JP',
            'Noto Sans SC',
            'Noto Sans TC',
            'Source Han Sans SC',
            'Source Han Sans CN',
            'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei',
            'Microsoft YaHei',
            'SimHei'
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = None
        for name in preferred:
            if name in available:
                chosen = name
                break

        if chosen is not None:
            current = mpl.rcParams.get('font.sans-serif', [])
            if not isinstance(current, list):
                current = [current]
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['font.sans-serif'] = [chosen] + [f for f in current if f != chosen]

        mpl.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


_configure_matplotlib_chinese_fonts()


def extract_features_with_backbone(backbone, X_data, config, logger):
    """
    Extract features using pre-trained backbone
    
    Args:
        backbone: Pre-trained MicroBiMambaBackbone
        X_data: (N, L, 5) input sequences
        config: configuration object
        logger: logger
        
    Returns:
        features: (N, d) extracted features
    """
    logger.info("Extracting features using backbone...")
    
    backbone.freeze()
    backbone.eval()
    backbone.to(config.DEVICE)
    
    features_list = []
    batch_size = 64
    X_tensor = torch.FloatTensor(X_data).to(config.DEVICE)
    
    total_batches = (len(X_tensor) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_idx = i // batch_size + 1
            X_batch = X_tensor[i:i+batch_size]
            z_batch = backbone(X_batch, return_sequence=False)
            features_list.append(z_batch.cpu().numpy())
            
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                progress = batch_idx / total_batches * 100
                logger.info(f"  Feature extraction progress: {batch_idx}/{total_batches} batches ({progress:.1f}%)")
    
    features = np.concatenate(features_list, axis=0)
    logger.info(f"✓ Feature extraction complete: {features.shape}")
    
    return features


def run_hybrid_court_with_detailed_tracking(features, noisy_labels, config, logger, y_true=None):
    """
    Run Hybrid Court v9 三阶段标签矫正 with detailed tracking
    
    Args:
        features: 特征矩阵
        noisy_labels: 噪声标签
        config: 配置对象
        logger: 日志器
        y_true: 真实标签 (可选，用于详细统计)
    
    Returns:
        results: dict containing all intermediate and final results
    """
    logger.info("")
    logger.info("="*70)
    logger.info("Running Hybrid Court v9 三阶段标签矫正")
    logger.info("="*70)
    
    hybrid_court = HybridCourt(config)
    n_samples = len(noisy_labels)
    
    # 调用三阶段标签矫正
    clean_labels, action_mask, confidence, correction_weight, density_scores, neighbor_consistency, pred_probs = hybrid_court.correct_labels(
        features, noisy_labels, device=config.DEVICE, y_true=y_true
    )
    
    # 从各子模块中读取缓存的中间结果
    suspected_noise = getattr(hybrid_court.cl, "last_suspected_noise", None)
    pred_labels = getattr(hybrid_court.cl, "last_pred_labels", None)
    is_dense = getattr(hybrid_court.made, "last_is_dense", None)
    neighbor_labels = getattr(hybrid_court.knn, "last_neighbor_labels", None)
    tier_info = getattr(hybrid_court, "last_tier_info", [''] * n_samples)
    
    # 获取迭代CL和锚点KNN结果
    iter_pred_probs = getattr(hybrid_court, "iter_pred_probs_all", None)
    anchor_votes = getattr(hybrid_court, "anchor_votes_all", None)
    anchor_consistency = getattr(hybrid_court, "anchor_consistency_all", None)
    
    # 生成决策原因
    decision_reasons = []
    for i in range(n_samples):
        tier = tier_info[i] if i < len(tier_info) else ''
        
        # 获取迭代CL和锚点KNN信息
        iter_cl_current = float(iter_pred_probs[i, int(noisy_labels[i])]) if iter_pred_probs is not None else None
        iter_cl_target = float(iter_pred_probs[i, 1 - int(noisy_labels[i])]) if iter_pred_probs is not None else None
        anchor_vote = int(anchor_votes[i]) if anchor_votes is not None else None
        anchor_cons = float(anchor_consistency[i]) if anchor_consistency is not None else None
        
        reason = _generate_decision_reason(
            tier, 
            int(noisy_labels[i]), 
            int(clean_labels[i]),
            int(action_mask[i]),
            bool(suspected_noise[i]) if suspected_noise is not None else False,
            bool(is_dense[i]) if is_dense is not None else True,
            int(neighbor_labels[i]) if neighbor_labels is not None else -1,
            float(neighbor_consistency[i]),
            float(pred_probs[i, 0]),
            float(pred_probs[i, 1]),
            float(density_scores[i]),
            iter_cl_current,
            iter_cl_target,
            anchor_vote,
            anchor_cons
        )
        decision_reasons.append(reason)
    
    # Compile results
    results = {
        'features': features,
        'noisy_labels': noisy_labels,
        'clean_labels': clean_labels,
        'action_mask': action_mask,
        'confidence': confidence,
        'correction_weight': correction_weight,
        # CL results
        'cl_suspected_noise': suspected_noise,
        'cl_pred_labels': pred_labels,
        'cl_pred_probs': pred_probs,
        # MADE results
        'made_is_dense': is_dense,
        'made_density_scores': density_scores,
        # KNN results
        'knn_neighbor_labels': neighbor_labels,
        'knn_neighbor_consistency': neighbor_consistency,
        # Iterative CL results
        'iter_cl_probs': iter_pred_probs,
        # Anchor KNN results
        'anchor_votes': anchor_votes,
        'anchor_consistency': anchor_consistency,
        # Decision tracking
        'decision_reasons': decision_reasons,
        'tier_info': tier_info
    }
    
    # 详细的Tier统计
    _log_tier_statistics(results, y_true, logger)
    
    return results


def _generate_decision_reason(tier, noisy_label, clean_label, action, is_suspected, is_dense, 
                               knn_label, knn_cons, cl_prob_0, cl_prob_1, density,
                               iter_cl_current=None, iter_cl_target=None, 
                               anchor_vote=None, anchor_cons=None):
    """生成详细的决策原因说明"""
    current_label_name = '正常' if noisy_label == 0 else '恶意'
    target_label_name = '恶意' if noisy_label == 0 else '正常'
    new_label_name = '正常' if clean_label == 0 else '恶意'
    
    # 原始CL置信度
    orig_cl_current = cl_prob_0 if noisy_label == 0 else cl_prob_1
    orig_cl_target = cl_prob_1 if noisy_label == 0 else cl_prob_0
    
    # KNN投票结果
    knn_vote_name = '正常' if knn_label == 0 else '恶意'
    knn_support = '支持' if knn_label == noisy_label else '反对'
    
    # Tier 1: Core - 定海神针
    if 'Tier 1: Core' in tier:
        reasons = []
        if orig_cl_current >= 0.70:
            reasons.append(f"原CL高({orig_cl_current:.3f}≥0.70)")
        if knn_cons >= 0.70 and knn_label == noisy_label:
            reasons.append(f"KNN强支持({knn_cons:.3f}≥0.70)")
        if not reasons:
            reasons.append(f"MADE+KNN补充(KNN={knn_cons:.3f})")
        return f"[Phase1-Core] {' 或 '.join(reasons)} → 核心样本(w=1.0)"
    
    # Tier 2: Flip - 智能翻转
    elif 'Tier 2: Flip' in tier:
        cl_diff = orig_cl_target - orig_cl_current
        if noisy_label == 0:  # 正常→恶意
            return (f"[Phase2-Flip] CL差值={cl_diff:.3f}≥0.15 + 正常样本 + "
                   f"原CL={orig_cl_current:.3f}≤0.95 + MADE={density:.1f}≤35 → "
                   f"翻转{current_label_name}→{new_label_name}(w=1.0)")
        else:  # 恶意→正常
            return (f"[Phase2-Flip] CL差值={cl_diff:.3f}≥0.15 + 恶意样本 + "
                   f"MADE={density:.1f}≥15 + 原CL={orig_cl_current:.3f}≥0.5 → "
                   f"翻转{current_label_name}→{new_label_name}(w=1.0)")
    
    # Tier 3: Keep - 高质量保持
    elif 'Tier 3' in tier or 'Keep' in tier:
        if 'High' in tier or '3a' in tier:
            return (f"[Phase2-Keep-High] 原CL={orig_cl_current:.3f}≥0.55 + KNN支持({knn_cons:.3f}≥0.55) + "
                   f"MADE正常({density:.1f}) → 保持{current_label_name}(w=0.8-1.0)")
        else:
            return (f"[Phase2-Keep-Low] 恶意样本 + MADE异常高({density:.1f}) → "
                   f"可能误标正常,保持{current_label_name}(w=0.4)")
    
    # Tier 4: Reweight - 不确定样本
    elif 'Tier 4' in tier or 'Reweight' in tier:
        reasons = []
        if orig_cl_current < 0.55:
            reasons.append(f"原CL低({orig_cl_current:.3f}<0.55)")
        if knn_cons < 0.55:
            reasons.append(f"KNN弱({knn_cons:.3f}<0.55)")
        if not is_suspected and knn_label == noisy_label:
            reasons.append("CL不怀疑+KNN支持")
        
        if not reasons:
            reasons.append("未达Core/Flip/Keep标准")
        
        return f"[Phase2-Reweight] {' + '.join(reasons)} → 降权保持{current_label_name}(w=0.5)"
    
    # Tier 5: Rescued - Phase 3拯救
    elif 'Tier 5' in tier or 'Rescued' in tier:
        if anchor_cons is not None and anchor_vote is not None:
            anchor_vote_name = '正常' if anchor_vote == 0 else '恶意'
            if 'Keep' in tier or '5a' in tier:
                return (f"[Phase3-Rescued-Keep] iter_T={iter_cl_target:.3f}≥0.60 + "
                       f"anchor强支持当前({anchor_vote_name},{anchor_cons:.3f}≥0.60) → "
                       f"拯救保持{current_label_name}(w=0.85)")
            else:
                return (f"[Phase3-Rescued-Flip] iter_T={iter_cl_target:.3f}≥0.60 + "
                       f"anchor强反对当前({anchor_vote_name},{anchor_cons:.3f}≥0.60) → "
                       f"拯救翻转{current_label_name}→{new_label_name}(w=0.75)")
        else:
            return f"[Phase3-Rescued] 锚点拯救 → {new_label_name}"
    
    # Dropped - 丢弃
    elif 'Dropped' in tier:
        drop_reasons = []
        if iter_cl_current is not None:
            if iter_cl_current < 0.48:
                drop_reasons.append(f"iter_CL极低({iter_cl_current:.3f}<0.48)")
            elif iter_cl_current < 0.55 and anchor_cons is not None and anchor_cons < 0.55:
                drop_reasons.append(f"iter_CL低({iter_cl_current:.3f}<0.55) + anchor低({anchor_cons:.3f}<0.55)")
        
        if not drop_reasons:
            drop_reasons.append("质量过低")
        
        return f"[Dropped] {' 或 '.join(drop_reasons)} → 丢弃(w=0.0)"
    
    # 未知Tier
    else:
        action_names = ['保持', '翻转', '丢弃', '重加权']
        action_name = action_names[action] if action < len(action_names) else '未知'
        return f"[未分类] tier={tier}, action={action_name}, 原CL={orig_cl_current:.3f}, KNN={knn_cons:.3f}"


def _log_tier_statistics(results, y_true, logger):
    """输出详细的Tier统计"""
    tier_info = results.get('tier_info', [])
    clean_labels = results['clean_labels']
    correction_weight = results['correction_weight']
    n_samples = len(clean_labels)
    
    # 统计各Tier
    tier_counts = {}
    tier_correct = {}
    tier_weights = {}
    
    for i, tier in enumerate(tier_info):
        if tier not in tier_counts:
            tier_counts[tier] = 0
            tier_correct[tier] = 0
            tier_weights[tier] = correction_weight[i]
        tier_counts[tier] += 1
        if y_true is not None and clean_labels[i] == y_true[i]:
            tier_correct[tier] += 1
    
    logger.info("")
    logger.info("="*90)
    logger.info("📊 三阶段标签矫正 - 详细Tier统计")
    logger.info("="*90)
    
    tier_order = [
        'Tier 1: Core',
        'Tier 2: Flip',
        'Tier 3a: Keep-High',
        'Tier 3b: Keep-Low',
        'Tier 4a: Reweight-High',
        'Tier 4b: Reweight-Low',
        'Tier 5a: Rescued-Keep',
        'Tier 5b: Rescued-Flip',
        'Dropped (Low CL)',
        'Dropped'
    ]
    
    role_map = {
        'Tier 1: Core': '[定海神针] 绝对纯净的基石数据',
        'Tier 2: Flip': '[强力纠错] 成功挽回的样本',
        'Tier 3a: Keep-High': '[难例精华] 优质保持样本',
        'Tier 3b: Keep-Low': '[风险隔离] 边缘样本低权重',
        'Tier 4a: Reweight-High': '[泛化主力] 清洗后的长尾数据',
        'Tier 4b: Reweight-Low': '[噪声监狱] 关押残留噪声',
        'Tier 5a: Rescued-Keep': '[二次拯救] 锚点KNN拯救Keep',
        'Tier 5b: Rescued-Flip': '[二次拯救] 锚点KNN拯救Flip',
        'Dropped (Low CL)': '[已丢弃] CL信心过低',
        'Dropped': '[已丢弃]'
    }
    
    logger.info(f"{'Tier':<30s} | {'权重':>6s} | {'样本数':>8s} | {'纯度':>8s} | {'含噪数':>8s} | 角色定位")
    logger.info("-" * 90)
    
    total_weighted_correct = 0
    total_weighted_count = 0
    
    for tier in tier_order:
        if tier in tier_counts:
            count = tier_counts[tier]
            correct = tier_correct.get(tier, 0)
            weight = tier_weights.get(tier, 0)
            purity = 100 * correct / count if count > 0 else 0
            noise_count = count - correct
            role = role_map.get(tier, '')
            
            logger.info(f"{tier:<30s} | {weight:>6.2f} | {count:>8d} | {purity:>7.1f}% | {noise_count:>8d} | {role}")
            
            if 'Dropped' not in tier:
                total_weighted_correct += correct * weight
                total_weighted_count += count * weight
    
    logger.info("-" * 90)
    
    if total_weighted_count > 0:
        weighted_purity = 100 * total_weighted_correct / total_weighted_count
        logger.info(f"📈 加权纯度: {weighted_purity:.2f}%")
    
    logger.info("="*90)


def generate_sample_analysis_document(results, y_true, noise_mask, save_path, logger):
    """
    Generate detailed per-sample analysis document in strategy-grouped format
    
    Args:
        results: dict from run_hybrid_court_with_detailed_tracking
        y_true: (N,) ground truth labels
        noise_mask: (N,) boolean array indicating which labels were flipped to create noise
        save_path: path to save the document
        logger: logger
    """
    logger.info("")
    logger.info("="*70)
    logger.info("生成样本级分析文档（策略分组格式）")
    logger.info("="*70)
    
    def _classify_error_type(is_noise, action, true_label, noisy_label, corrected_label):
        """分类错误类型"""
        if corrected_label == true_label:
            return '正确'
        
        if is_noise:
            # 噪声样本
            if action == 1:  # 翻转
                return '翻转错误-噪声未矫正'
            else:
                return '保持错误-噪声未检测'
        else:
            # 干净样本
            if action == 1:  # 翻转
                return '翻转错误-误杀干净样本'
            else:
                return '保持错误-不应发生'
    
    n_samples = len(y_true)
    tier_info = results.get('tier_info', [''] * n_samples)
    
    # 获取迭代CL和锚点KNN的结果（如果存在）
    iter_cl_probs = results.get('iter_cl_probs', None)
    anchor_votes = results.get('anchor_votes', None)
    anchor_consistency = results.get('anchor_consistency', None)
    
    # Prepare data for DataFrame
    data = []
    
    for i in range(n_samples):
        # Basic info
        sample_id = i
        true_label = int(y_true[i])
        true_label_name = "正常" if true_label == 0 else "恶意"
        is_noise = bool(noise_mask[i])
        noisy_label = int(results['noisy_labels'][i])
        noisy_label_name = "正常" if noisy_label == 0 else "恶意"
        
        # CL (Confident Learning) results
        cl_suspected_noise = bool(results['cl_suspected_noise'][i])
        cl_pred_label = int(results['cl_pred_labels'][i])
        cl_pred_label_name = "正常" if cl_pred_label == 0 else "恶意"
        cl_pred_prob_benign = float(results['cl_pred_probs'][i, 0])
        cl_pred_prob_malicious = float(results['cl_pred_probs'][i, 1])
        
        # MADE (Density Estimation) results
        made_is_dense = bool(results['made_is_dense'][i])
        made_density_score = float(results['made_density_scores'][i])
        
        # KNN (Semantic Voting) results
        knn_neighbor_label = int(results['knn_neighbor_labels'][i])
        knn_neighbor_label_name = "正常" if knn_neighbor_label == 0 else "恶意"
        knn_consistency = float(results['knn_neighbor_consistency'][i])
        
        # Iterative CL results (if available)
        if iter_cl_probs is not None:
            iter_cl_current = float(iter_cl_probs[i, noisy_label])
            iter_cl_target = float(iter_cl_probs[i, 1 - noisy_label])
        else:
            iter_cl_current = None
            iter_cl_target = None
        
        # Anchor KNN results (if available)
        if anchor_votes is not None:
            anchor_vote = int(anchor_votes[i])
            anchor_vote_name = "正常" if anchor_vote == 0 else "恶意"
            anchor_cons = float(anchor_consistency[i])
        else:
            anchor_vote = None
            anchor_vote_name = None
            anchor_cons = None
        
        # Final correction decision
        action = int(results['action_mask'][i])
        action_names = ['保持', '翻转', '丢弃', '重加权']
        action_name = action_names[action]
        corrected_label = int(results['clean_labels'][i])
        corrected_label_name = "正常" if corrected_label == 0 else "恶意"
        confidence = float(results['confidence'][i])
        correction_weight = float(results['correction_weight'][i])
        decision_reason = results['decision_reasons'][i]
        tier = tier_info[i] if i < len(tier_info) else ''
        
        # Correction correctness (compared to ground truth)
        if action == 2:  # Dropped
            correction_correct = "不适用(已丢弃)"
        else:
            correction_correct = "正确" if corrected_label == true_label else "错误"
        
        # Compile row with ALL detailed metrics
        row = {
            # 基本信息
            '样本ID': sample_id,
            '真实标签': true_label_name,
            '真实标签值': true_label,
            '是否噪声': '是' if is_noise else '否',
            '噪声标签': noisy_label_name,
            '噪声标签值': noisy_label,
            
            # CL (Confident Learning) 详细结果
            'CL疑似噪声': '是' if cl_suspected_noise else '否',
            'CL预测标签': cl_pred_label_name,
            'CL预测标签值': cl_pred_label,
            'CL正常概率': cl_pred_prob_benign,
            'CL恶意概率': cl_pred_prob_malicious,
            'CL当前标签置信度': cl_pred_prob_benign if noisy_label == 0 else cl_pred_prob_malicious,
            'CL目标标签置信度': cl_pred_prob_malicious if noisy_label == 0 else cl_pred_prob_benign,
            
            # MADE (Density Estimation) 详细结果
            'MADE高密度': '是' if made_is_dense else '否',
            'MADE密度分数': made_density_score,
            'MADE密度等级': 'High' if made_density_score > 60 else ('Medium' if made_density_score > 0 else 'Low'),
            
            # KNN (Semantic Voting) 详细结果
            'KNN邻居标签': knn_neighbor_label_name,
            'KNN邻居标签值': knn_neighbor_label,
            'KNN一致性': knn_consistency,
            'KNN是否支持当前标签': '是' if knn_neighbor_label == noisy_label else '否',
            'KNN一致性等级': 'High' if knn_consistency >= 0.7 else ('Medium' if knn_consistency >= 0.5 else 'Low'),
        }
        
        # Add iterative CL columns if available
        if iter_cl_current is not None:
            row['iter_CL当前标签置信度'] = iter_cl_current
            row['iter_CL目标标签置信度'] = iter_cl_target
            row['iter_CL置信度差值'] = abs(iter_cl_current - iter_cl_target)
        
        # Add anchor KNN columns if available
        if anchor_vote is not None:
            row['anchor_KNN投票'] = anchor_vote_name
            row['anchor_KNN投票值'] = anchor_vote
            row['anchor_KNN一致性'] = anchor_cons
            row['anchor_KNN是否支持当前'] = '是' if anchor_vote == noisy_label else '否'
        
        # 最终决策详细信息
        row.update({
            '矫正动作': action_name,
            '矫正动作值': action,
            '矫正后标签': corrected_label_name,
            '矫正后标签值': corrected_label,
            '系统置信度': confidence,
            '样本权值': correction_weight,
            'Tier分级': tier,
            'Tier阶段': 'Phase 1' if 'Tier 1' in tier else ('Phase 2' if any(t in tier for t in ['Tier 2', 'Tier 3', 'Tier 4']) else ('Phase 3' if 'Tier 5' in tier else 'Dropped')),
            '决策理由': decision_reason,
            
            # 矫正结果评估
            '矫正是否正确': correction_correct,
            '是否翻转': '是' if action == 1 else '否',
            '是否丢弃': '是' if action == 2 else '否',
            '标签是否改变': '是' if corrected_label != noisy_label else '否',
            
            # 错误类型分析
            '错误类型': _classify_error_type(
                is_noise, action, true_label, noisy_label, corrected_label
            ) if action != 2 else '已丢弃'
        })
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV (基础版本，所有数据在一个文件)
    csv_path = save_path.replace('.txt', '.csv').replace('.log', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"✓ CSV文件已保存: {csv_path}")
    
    # Save to Excel with multiple sheets (如果openpyxl可用)
    if EXCEL_AVAILABLE:
        excel_path = csv_path.replace('.csv', '.xlsx')
        try:
            from openpyxl.styles import PatternFill, Font
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # 重新排列列顺序，把关键信息放前面
                key_columns = [
                    '样本ID', '矫正是否正确', '错误类型', 'Tier分级', 'Tier阶段',
                    '是否噪声', '真实标签', '真实标签值', '噪声标签', '噪声标签值',
                    '矫正后标签', '矫正后标签值', '矫正动作', '决策理由',
                    '系统置信度', '样本权值', '是否翻转', '是否丢弃', '标签是否改变'
                ]
                # 只选择存在的列
                existing_key_columns = [col for col in key_columns if col in df.columns]
                other_columns = [col for col in df.columns if col not in existing_key_columns]
                df_reordered = df[existing_key_columns + other_columns]
                
                # Sheet 1: 总览统计
                tier_summary = []
                for tier in sorted(df['Tier分级'].unique()):
                    tier_df = df[df['Tier分级'] == tier]
                    valid_df = tier_df[tier_df['矫正是否正确'] != '不适用(已丢弃)']
                    correct_count = len(tier_df[tier_df['矫正是否正确'] == '正确'])
                    error_count = len(tier_df[tier_df['矫正是否正确'] == '错误'])
                    dropped_count = len(tier_df[tier_df['矫正是否正确'] == '不适用(已丢弃)'])
                    
                    tier_summary.append({
                        'Tier': tier,
                        '总样本数': len(tier_df),
                        '✓正确处理': correct_count,
                        '✗错误处理': error_count,
                        '⊗已丢弃': dropped_count,
                        '准确率': f"{correct_count / len(valid_df) * 100:.2f}%" if len(valid_df) > 0 else 'N/A',
                        '平均权值': f"{tier_df['样本权值'].mean():.4f}",
                        '噪声样本数': len(tier_df[tier_df['是否噪声'] == '是']),
                        '干净样本数': len(tier_df[tier_df['是否噪声'] == '否'])
                    })
                
                summary_df = pd.DataFrame(tier_summary)
                summary_df.to_excel(writer, sheet_name='📊总览统计', index=False)
                
                # 为总览统计添加格式
                ws = writer.sheets['📊总览统计']
                for row in range(2, len(summary_df) + 2):
                    # 正确处理 - 绿色
                    ws.cell(row=row, column=3).fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                    # 错误处理 - 红色
                    ws.cell(row=row, column=4).fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                    # 已丢弃 - 灰色
                    ws.cell(row=row, column=5).fill = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
                
                # Sheet 2: 所有样本（完整数据）
                df_reordered.to_excel(writer, sheet_name='所有样本', index=False)
                ws_all = writer.sheets['所有样本']
                
                # 为所有样本添加条件格式
                for row in range(2, len(df_reordered) + 2):
                    correction_status = df_reordered.iloc[row-2]['矫正是否正确']
                    if correction_status == '正确':
                        # 整行浅绿色
                        for col in range(1, len(df_reordered.columns) + 1):
                            ws_all.cell(row=row, column=col).fill = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
                    elif correction_status == '错误':
                        # 整行浅红色
                        for col in range(1, len(df_reordered.columns) + 1):
                            ws_all.cell(row=row, column=col).fill = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")
                
                # Sheet 3-N: 每个Tier的详细数据（按策略分组）
                tier_order = [
                    'Tier 1: Core',
                    'Tier 2: Flip',
                    'Tier 3: Keep',
                    'Tier 4: Reweight',
                    'Tier 5a: Rescued-Keep',
                    'Tier 5b: Rescued-Flip',
                    'Tier 5b: Rescued-Flip (Drop前拯救)',
                    'Dropped'
                ]
                
                for tier in tier_order:
                    tier_data = df_reordered[df_reordered['Tier分级'] == tier]
                    
                    if len(tier_data) == 0:
                        # 尝试模糊匹配
                        tier_base = tier.split('(')[0].strip()
                        tier_data = df_reordered[df_reordered['Tier分级'].str.contains(tier_base, na=False, regex=False)]
                    
                    if len(tier_data) > 0:
                        # 分离正确和错误处理的样本
                        correct_samples = tier_data[tier_data['矫正是否正确'] == '正确']
                        error_samples = tier_data[tier_data['矫正是否正确'] == '错误']
                        dropped_samples = tier_data[tier_data['矫正是否正确'] == '不适用(已丢弃)']
                        
                        # 创建sheet名称（Excel限制31字符）
                        sheet_name = tier.replace(':', '').replace('(', '').replace(')', '').replace(' ', '_')[:28]
                        
                        # 合并数据：正确的在前，错误的在后
                        combined_data = pd.concat([correct_samples, error_samples, dropped_samples])
                        combined_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # 添加颜色标记
                        ws = writer.sheets[sheet_name]
                        current_row = 2
                        
                        # 正确样本 - 绿色背景
                        for _ in range(len(correct_samples)):
                            for col in range(1, len(combined_data.columns) + 1):
                                ws.cell(row=current_row, column=col).fill = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
                            current_row += 1
                        
                        # 错误样本 - 红色背景
                        for _ in range(len(error_samples)):
                            for col in range(1, len(combined_data.columns) + 1):
                                ws.cell(row=current_row, column=col).fill = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")
                            current_row += 1
                        
                        # 丢弃样本 - 灰色背景
                        for _ in range(len(dropped_samples)):
                            for col in range(1, len(combined_data.columns) + 1):
                                ws.cell(row=current_row, column=col).fill = PatternFill(start_color="F5F5F5", end_color="F5F5F5", fill_type="solid")
                            current_row += 1
                        
                        logger.info(f"  ✓ {sheet_name}: {len(correct_samples)}正确 / {len(error_samples)}错误 / {len(dropped_samples)}丢弃")
                
                # 额外的分析sheet
                # Sheet: 所有错误样本汇总
                error_samples = df_reordered[df_reordered['矫正是否正确'] == '错误']
                if len(error_samples) > 0:
                    error_samples.to_excel(writer, sheet_name='❌所有错误样本', index=False)
                    logger.info(f"  ✓ 错误样本汇总: {len(error_samples)}个")
                
                # Sheet: 误杀分析（被错误处理的干净样本）
                false_positive = df_reordered[(df_reordered['是否噪声'] == '否') & (df_reordered['矫正是否正确'] == '错误')]
                if len(false_positive) > 0:
                    false_positive.to_excel(writer, sheet_name='⚠️误杀干净样本', index=False)
                    logger.info(f"  ✓ 误杀样本: {len(false_positive)}个")
                
                # Sheet: 漏网之鱼（未被矫正的噪声）
                false_negative = df_reordered[(df_reordered['是否噪声'] == '是') & (df_reordered['矫正是否正确'] == '错误')]
                if len(false_negative) > 0:
                    false_negative.to_excel(writer, sheet_name='🐟漏网噪声', index=False)
                    logger.info(f"  ✓ 漏网噪声: {len(false_negative)}个")
            
            logger.info(f"✓ Excel文件已保存（多sheet，带颜色标记）: {excel_path}")
            logger.info(f"  包含 {len(pd.ExcelFile(excel_path).sheet_names)} 个工作表")
            logger.info(f"  颜色说明: 绿色=正确处理, 红色=错误处理, 灰色=已丢弃")
        except Exception as e:
            logger.warning(f"⚠ Excel文件保存失败: {e}")
            import traceback
            logger.warning(traceback.format_exc())
    else:
        logger.info("  提示: 安装openpyxl可生成多sheet Excel文件 (pip install openpyxl)")
    
    # ========================================
    # 新格式：按策略分组的日志总结
    # ========================================
    
    # 按Tier分组样本
    tier_groups = {}
    for i, row in df.iterrows():
        tier = row['Tier分级']
        if tier not in tier_groups:
            tier_groups[tier] = {'correct': [], 'incorrect': [], 'dropped': []}
        
        if row['矫正是否正确'] == '正确':
            tier_groups[tier]['correct'].append(row)
        elif row['矫正是否正确'] == '错误':
            tier_groups[tier]['incorrect'].append(row)
        else:  # 不适用(已丢弃)
            tier_groups[tier]['dropped'].append(row)
    
    # 定义Tier顺序和角色说明（包含详细路径）
    tier_order = [
        'Tier 1: Core',
        'Tier 2: Flip',
        'Tier 3: Keep',
        'Tier 4: Reweight',
        'Tier 5a: Rescued-Keep',
        'Tier 5b: Rescued-Flip',
        'Tier 5b: Rescued-Flip (Drop前拯救)',
        'Dropped'
    ]
    
    role_map = {
        'Tier 1: Core': 'Phase 1 - 定海神针：绝对纯净的基石数据',
        'Tier 2: Flip': 'Phase 2 - 智能翻转：基于CL差值(≥0.15)和MADE密度的零误杀翻转',
        'Tier 3: Keep': 'Phase 2 - 保持样本：高质量保持',
        'Tier 4: Reweight': 'Phase 2 - 重加权样本：不确定样本降权',
        'Tier 5a: Rescued-Keep': 'Phase 3 - 锚点拯救：拯救的保持样本',
        'Tier 5b: Rescued-Flip': 'Phase 3 - 锚点拯救：拯救的翻转样本',
        'Tier 5b: Rescued-Flip (Drop前拯救)': 'Phase 3 - 锚点拯救：Drop前拯救的翻转样本',
        'Dropped': '已丢弃：质量过低'
    }
    
    # 策略详细说明
    strategy_details = {
        'Tier 1: Core': {
            'phase': 'Phase 1: 核心严选',
            'condition': 'CL置信度≥阈值 或 KNN一致性≥阈值',
            'action': '保持原标签',
            'weight': '1.0'
        },
        'Tier 2: Flip': {
            'phase': 'Phase 2: 智能翻转',
            'condition': 'CL差值≥0.15 + 正常样本(CL≤0.95且密度≤35翻转) / 恶意样本(密度≥15且CL≥0.5翻转)',
            'action': '翻转标签',
            'weight': '1.0'
        },
        'Tier 3: Keep': {
            'phase': 'Phase 2: 分级挽救 - Keep策略',
            'condition': 'KNN支持 + 原CL≥0.55 + KNN≥0.55 + MADE正常',
            'action': '保持原标签',
            'weight': '0.8-1.0'
        },
        'Tier 4: Reweight': {
            'phase': 'Phase 2: 分级挽救 - Reweight策略',
            'condition': '未达到Core/Flip/Keep标准',
            'action': '保持原标签但降权',
            'weight': '0.5'
        },
        'Tier 5a: Rescued-Keep': {
            'phase': 'Phase 3: 锚点拯救',
            'condition': 'iter_T≥0.6 + anchor强支持当前(≥0.6)',
            'action': '保持原标签',
            'weight': '0.85'
        },
        'Tier 5b: Rescued-Flip': {
            'phase': 'Phase 3: 锚点拯救',
            'condition': 'iter_T≥0.6 + anchor强反对当前(≥0.6)',
            'action': '翻转标签',
            'weight': '0.75'
        },
        'Tier 5b: Rescued-Flip (Drop前拯救)': {
            'phase': 'Phase 3: 锚点拯救（Drop前）',
            'condition': 'iter_T≥0.6 + anchor支持目标≥0.6 + orig_KNN支持目标≥0.55',
            'action': '翻转标签（避免Drop）',
            'weight': '1.0'
        },
        'Dropped': {
            'phase': 'Phase 3: 最终清理',
            'condition': 'iter_CL<0.48 或 (iter_CL<0.55 + anchor<0.55)',
            'action': '丢弃样本',
            'weight': '0.0'
        }
    }
    
    # 将文件后缀改为.log以支持更好的表格格式
    log_path = save_path.replace('.txt', '.log')
    
    # Save to formatted log file (新格式 - 表格形式)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("MEDAL-Lite 标签矫正分析 - 策略分组日志报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*120 + "\n\n")
        
        # Summary statistics
        f.write("📊 总体统计\n")
        f.write("-"*120 + "\n")
        f.write(f"总样本数: {n_samples}\n")
        f.write(f"  正常流量:  {(y_true == 0).sum()} ({100*(y_true == 0).sum()/n_samples:.1f}%)\n")
        f.write(f"  恶意流量:  {(y_true == 1).sum()} ({100*(y_true == 1).sum()/n_samples:.1f}%)\n\n")
        
        f.write(f"噪声注入率: {100*noise_mask.sum()/n_samples:.1f}%\n")
        f.write(f"  噪声样本数: {noise_mask.sum()}\n\n")
        
        action_mask = results['action_mask']
        f.write(f"矫正动作统计:\n")
        f.write(f"  保持:     {(action_mask == 0).sum()} ({100*(action_mask == 0).sum()/n_samples:.1f}%)\n")
        f.write(f"  翻转:     {(action_mask == 1).sum()} ({100*(action_mask == 1).sum()/n_samples:.1f}%)\n")
        f.write(f"  丢弃:     {(action_mask == 2).sum()} ({100*(action_mask == 2).sum()/n_samples:.1f}%)\n")
        f.write(f"  重加权:   {(action_mask == 3).sum()} ({100*(action_mask == 3).sum()/n_samples:.1f}%)\n\n")
        
        # Correction accuracy (excluding dropped samples)
        keep_mask = action_mask != 2
        corrected_labels = results['clean_labels'][keep_mask]
        true_labels_kept = y_true[keep_mask]
        correction_accuracy = (corrected_labels == true_labels_kept).mean()
        f.write(f"矫正准确率 (不含丢弃样本): {correction_accuracy*100:.2f}%\n")
        f.write(f"  正确: {(corrected_labels == true_labels_kept).sum()}\n")
        f.write(f"  错误: {(corrected_labels != true_labels_kept).sum()}\n\n")
        
        f.write("="*120 + "\n\n")
        
        # 按策略分组的详细分析
        f.write("📋 策略分组详细分析（按阶段和路径）\n")
        f.write("="*120 + "\n\n")
        
        # 按Phase分组
        phase_groups = {
            'Phase 1: 核心严选': [],
            'Phase 2: 分级挽救': [],
            'Phase 3: 锚点拯救': [],
            '已丢弃': []
        }
        
        for tier in tier_order:
            if tier not in tier_groups or len(tier_groups[tier]['correct']) + len(tier_groups[tier]['incorrect']) + len(tier_groups[tier]['dropped']) == 0:
                continue
            
            if 'Tier 1' in tier:
                phase_groups['Phase 1: 核心严选'].append(tier)
            elif 'Tier 2' in tier or 'Tier 3' in tier or 'Tier 4' in tier:
                phase_groups['Phase 2: 分级挽救'].append(tier)
            elif 'Tier 5' in tier:
                phase_groups['Phase 3: 锚点拯救'].append(tier)
            elif 'Dropped' in tier:
                phase_groups['已丢弃'].append(tier)
        
        # 按Phase输出
        for phase_name, tiers in phase_groups.items():
            if not tiers:
                continue
            
            f.write("\n" + "="*120 + "\n")
            f.write(f"【{phase_name}】\n")
            f.write("="*120 + "\n\n")
            
            for tier in tiers:
                if tier not in tier_groups:
                    continue
                
                correct_samples = tier_groups[tier]['correct']
                incorrect_samples = tier_groups[tier]['incorrect']
                dropped_samples = tier_groups[tier]['dropped']
                total_samples = len(correct_samples) + len(incorrect_samples) + len(dropped_samples)
                
                if total_samples == 0:
                    continue
                
                role = role_map.get(tier, '')
                
                # 计算准确率（不包括丢弃的样本）
                valid_samples = len(correct_samples) + len(incorrect_samples)
                accuracy = len(correct_samples) / valid_samples * 100 if valid_samples > 0 else 0
                
                # 获取该Tier的权重
                if correct_samples:
                    weight = correct_samples[0]['样本权值']
                elif incorrect_samples:
                    weight = incorrect_samples[0]['样本权值']
                elif dropped_samples:
                    weight = dropped_samples[0]['样本权值']
                else:
                    weight = 'N/A'
                
                f.write(f"┌─ 【策略】{tier}\n")
                f.write(f"│  角色定位: {role}\n")
                
                # 输出策略详细说明
                if tier in strategy_details:
                    details = strategy_details[tier]
                    f.write(f"│  决策条件: {details['condition']}\n")
                    f.write(f"│  执行动作: {details['action']}\n")
                    f.write(f"│  样本权重: {details['weight']}\n")
                
                f.write(f"│  样本统计: 总数={total_samples} | 实际权重={weight} | 准确率={accuracy:.1f}%\n")
                f.write(f"│             正确={len(correct_samples)} | 错误={len(incorrect_samples)} | 丢弃={len(dropped_samples)}\n")
                f.write(f"└─" + "-"*116 + "\n\n")
                
                # 正确处理的样本 - 表格形式
                if correct_samples:
                    f.write(f"  ✓ 正确处理的样本 ({len(correct_samples)}个):\n")
                    f.write("  " + "-"*150 + "\n")
                    # 表头 - 添加更多关键指标
                    f.write(f"  {'样本ID':>8s} | {'真实':>4s} | {'噪声':>4s} | {'矫正后':>6s} | "
                           f"{'原CL':>6s} | {'iter_CL':>7s} | {'iter_T':>7s} | "
                           f"{'anchor':>7s} | {'orig_KNN':>8s} | 决策理由\n")
                    f.write("  " + "-"*150 + "\n")
                    
                    # 显示前10个样本
                    for row in correct_samples[:10]:
                        # 原始CL置信度
                        orig_cl = row['CL正常概率'] if row['噪声标签']=='正常' else row['CL恶意概率']
                        # 迭代CL
                        iter_cl_current = row.get('iter_CL当前标签置信度', 'N/A')
                        iter_cl_target = row.get('iter_CL目标标签置信度', 'N/A')
                        # 锚点KNN
                        anchor_cons = row.get('anchor_KNN一致性', 'N/A')
                        # 原始KNN
                        orig_knn = row.get('KNN一致性', 'N/A')
                        
                        reason_short = row['决策理由'][:40] + '...' if len(row['决策理由']) > 40 else row['决策理由']
                        f.write(f"  {row['样本ID']:>8d} | {str(row['真实标签']):>4} | {str(row['噪声标签']):>4} | "
                               f"{str(row['矫正后标签']):>6} | "
                               f"{str(orig_cl)[:6]:>6} | {str(iter_cl_current)[:7]:>7} | {str(iter_cl_target)[:7]:>7} | "
                               f"{str(anchor_cons)[:7]:>7} | {str(orig_knn)[:8]:>8} | {reason_short}\n")
                    
                    if len(correct_samples) > 10:
                        f.write(f"  ... 还有 {len(correct_samples) - 10} 个正确样本（详见Excel文件）\n")
                    f.write("\n")
                
                # 错误处理的样本 - 表格形式（显示所有）
                if incorrect_samples:
                    f.write(f"  ✗ 错误处理的样本 ({len(incorrect_samples)}个):\n")
                    f.write("  " + "-"*150 + "\n")
                    # 表头
                    f.write(f"  {'样本ID':>8s} | {'真实':>4s} | {'噪声':>4s} | {'矫正后':>6s} | "
                           f"{'原CL':>6s} | {'iter_CL':>7s} | {'iter_T':>7s} | "
                           f"{'anchor':>7s} | {'orig_KNN':>8s} | 错误原因\n")
                    f.write("  " + "-"*150 + "\n")
                    
                    for row in incorrect_samples:
                        # 原始CL置信度
                        orig_cl = row['CL正常概率'] if row['噪声标签']=='正常' else row['CL恶意概率']
                        # 迭代CL
                        iter_cl_current = row.get('iter_CL当前标签置信度', 'N/A')
                        iter_cl_target = row.get('iter_CL目标标签置信度', 'N/A')
                        # 锚点KNN
                        anchor_cons = row.get('anchor_KNN一致性', 'N/A')
                        # 原始KNN
                        orig_knn = row.get('KNN一致性', 'N/A')
                        
                        error_reason = f"真实={row['真实标签']}, 矫正为{row['矫正后标签']}"
                        f.write(f"  {row['样本ID']:>8d} | {str(row['真实标签']):>4} | {str(row['噪声标签']):>4} | "
                               f"{str(row['矫正后标签']):>6} | "
                               f"{str(orig_cl)[:6]:>6} | {str(iter_cl_current)[:7]:>7} | {str(iter_cl_target)[:7]:>7} | "
                               f"{str(anchor_cons)[:7]:>7} | {str(orig_knn)[:8]:>8} | {error_reason}\n")
                    f.write("\n")
                
                # 丢弃的样本 - 表格形式
                if dropped_samples:
                    f.write(f"  🗑 丢弃的样本 ({len(dropped_samples)}个):\n")
                    f.write("  " + "-"*150 + "\n")
                    # 表头
                    f.write(f"  {'样本ID':>8s} | {'真实':>4s} | {'噪声':>4s} | {'是否噪声':>8s} | "
                           f"{'原CL':>6s} | {'iter_CL':>7s} | {'iter_T':>7s} | "
                           f"{'anchor':>7s} | {'orig_KNN':>8s} | 丢弃原因\n")
                    f.write("  " + "-"*150 + "\n")
                    
                    # 显示前10个丢弃样本
                    for row in dropped_samples[:10]:
                        # 原始CL置信度
                        orig_cl = row['CL正常概率'] if row['噪声标签']=='正常' else row['CL恶意概率']
                        # 迭代CL
                        iter_cl_current = row.get('iter_CL当前标签置信度', 'N/A')
                        iter_cl_target = row.get('iter_CL目标标签置信度', 'N/A')
                        # 锚点KNN
                        anchor_cons = row.get('anchor_KNN一致性', 'N/A')
                        # 原始KNN
                        orig_knn = row.get('KNN一致性', 'N/A')
                        
                        drop_reason = "CL极低" if 'CL' in row['决策理由'] else "无法挽救"
                        f.write(f"  {row['样本ID']:>8d} | {str(row['真实标签']):>4} | {str(row['噪声标签']):>4} | "
                               f"{str(row['是否噪声']):>8} | "
                               f"{str(orig_cl)[:6]:>6} | {str(iter_cl_current)[:7]:>7} | {str(iter_cl_target)[:7]:>7} | "
                               f"{str(anchor_cons)[:7]:>7} | {str(orig_knn)[:8]:>8} | {drop_reason}\n")
                    
                    if len(dropped_samples) > 10:
                        f.write(f"  ... 还有 {len(dropped_samples) - 10} 个丢弃样本（详见Excel文件）\n")
                    f.write("\n")
                
                f.write("\n")
        
        # 添加误杀分析（被错误丢弃的干净样本）
        f.write("="*120 + "\n")
        f.write("⚠️  误杀分析 - 被错误丢弃的干净样本\n")
        f.write("="*120 + "\n\n")
        
        # 找出所有被丢弃但实际是干净的样本
        false_drops = []
        for i, row in df.iterrows():
            if row['矫正动作'] == '丢弃' and row['是否噪声'] == '否':
                false_drops.append(row)
        
        if false_drops:
            f.write(f"📊 被误杀的干净样本详情 (共{len(false_drops)}个):\n")
            f.write("-"*120 + "\n")
            f.write(f"  {'样本ID':>8s} | {'标签':>4s} | {'iter_CL':>7s} | {'iter_CL_T':>9s} | {'anchor':>7s} | {'orig_KNN':>8s} | 原因\n")
            f.write("-"*120 + "\n")
            
            for row in false_drops:
                # 使用iter_CL当前标签置信度（这才是Drop判断的依据）
                iter_cl_current = row.get('iter_CL当前标签置信度', 'N/A')
                iter_cl_target = row.get('iter_CL目标标签置信度', 'N/A')
                anchor_cons = row.get('anchor_KNN一致性', 'N/A')
                orig_knn_cons = row.get('KNN一致性', 'N/A')
                
                f.write(f"  {row['样本ID']:>8d} | {str(row['真实标签']):>4} | {str(iter_cl_current):>7} | "
                       f"{str(iter_cl_target):>9} | {str(anchor_cons):>7} | {str(orig_knn_cons):>8} | CL极低\n")
        else:
            f.write("✓ 没有误杀干净样本\n")
        
        f.write("\n")
        
        # 添加漏网之鱼分析（未被检测到的噪声样本）
        f.write("="*120 + "\n")
        f.write("🐟 漏网之鱼分析 - 未被矫正的噪声样本\n")
        f.write("="*120 + "\n\n")
        
        # 找出所有是噪声但未被翻转的样本
        missed_noise = []
        for i, row in df.iterrows():
            if row['是否噪声'] == '是' and row['矫正动作'] != '翻转':
                missed_noise.append(row)
        
        if missed_noise:
            f.write(f"📊 漏网的噪声样本详情 (共{len(missed_noise)}个):\n")
            f.write("-"*120 + "\n")
            f.write(f"  {'样本ID':>8s} | {'真实':>4s} | {'噪声':>4s} | {'动作':>4s} | "
                   f"{'CL置信':>7s} | {'KNN一致':>7s} | {'MADE密度':>9s} | 原因\n")
            f.write("-"*120 + "\n")
            
            for row in missed_noise[:20]:  # 显示前20个
                cl_conf = row['CL正常概率'] if row['噪声标签']=='正常' else row['CL恶意概率']
                reason = "未被CL检测" if row['CL疑似噪声'] == '否' else "KNN支持错误标签"
                f.write(f"  {row['样本ID']:>8d} | {str(row['真实标签']):>4} | {str(row['噪声标签']):>4} | "
                       f"{str(row['矫正动作']):>4} | {str(cl_conf):>7} | "
                       f"{str(row['KNN一致性']):>7} | {str(row['MADE密度分数']):>9} | {reason}\n")
            
            if len(missed_noise) > 20:
                f.write(f"  ... 还有 {len(missed_noise) - 20} 个漏网样本（详见CSV文件）\n")
        else:
            f.write("✓ 所有噪声样本都被成功检测\n")
        
        f.write("\n")
        f.write("="*120 + "\n")
        f.write("报告结束\n")
        f.write("="*120 + "\n")
    
    logger.info(f"✓ 策略分组日志已保存: {log_path}")
    logger.info(f"  已分析样本总数: {n_samples}")
    logger.info(f"  策略分组数: {len([t for t in tier_order if t in tier_groups])}")
    
    return df


def plot_feature_distributions(results, y_true, noise_mask, save_dir, logger):
    """
    Generate feature distribution plots for clean, noisy, and corrected data
    
    Args:
        results: dict from run_hybrid_court_with_detailed_tracking
        y_true: (N,) ground truth labels
        noise_mask: (N,) boolean array indicating noise
        save_dir: directory to save plots
        logger: logger
    """
    logger.info("")
    logger.info("="*70)
    logger.info("Generating Feature Distribution Plots")
    logger.info("="*70)
    
    features = results['features']
    noisy_labels = results['noisy_labels']
    clean_labels = results['clean_labels']
    action_mask = results['action_mask']
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Use t-SNE for dimensionality reduction
    logger.info("Running t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    logger.info("✓ t-SNE complete")
    
    # Define colors
    color_benign = '#2E86AB'  # Blue
    color_malicious = '#A23B72'  # Red
    
    # 1. Ground Truth Labels (Clean Data)
    logger.info("绘制 1/4: 真实标签分布...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    benign_mask = y_true == 0
    malicious_mask = y_true == 1
    
    ax.scatter(features_2d[benign_mask, 0], features_2d[benign_mask, 1],
               c=color_benign, label=f'正常流量 (n={benign_mask.sum()})', 
               alpha=0.6, s=30, edgecolors='k', linewidths=0.3)
    ax.scatter(features_2d[malicious_mask, 0], features_2d[malicious_mask, 1],
               c=color_malicious, label=f'恶意流量 (n={malicious_mask.sum()})', 
               alpha=0.6, s=30, edgecolors='k', linewidths=0.3)
    
    ax.set_xlabel('t-SNE 第一主成分', fontsize=12)
    ax.set_ylabel('t-SNE 第二主成分', fontsize=12)
    ax.set_title('特征分布: 真实标签 (干净数据)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, '1_真实标签分布.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已保存: {save_path}")
    
    # 2. Noisy Labels (with noise highlighted)
    logger.info("绘制 2/4: 噪声标签分布...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot clean samples first
    clean_samples = ~noise_mask
    benign_clean = clean_samples & (noisy_labels == 0)
    malicious_clean = clean_samples & (noisy_labels == 1)
    
    ax.scatter(features_2d[benign_clean, 0], features_2d[benign_clean, 1],
               c=color_benign, label=f'正常流量 (干净, n={benign_clean.sum()})', 
               alpha=0.4, s=25, edgecolors='k', linewidths=0.3)
    ax.scatter(features_2d[malicious_clean, 0], features_2d[malicious_clean, 1],
               c=color_malicious, label=f'恶意流量 (干净, n={malicious_clean.sum()})', 
               alpha=0.4, s=25, edgecolors='k', linewidths=0.3)
    
    # Plot noisy samples with special marker
    noisy_samples = noise_mask
    benign_noisy = noisy_samples & (noisy_labels == 0)
    malicious_noisy = noisy_samples & (noisy_labels == 1)
    
    ax.scatter(features_2d[benign_noisy, 0], features_2d[benign_noisy, 1],
               c=color_benign, label=f'正常流量 (噪声, n={benign_noisy.sum()})', 
               alpha=0.9, s=80, edgecolors='yellow', linewidths=2, marker='s')
    ax.scatter(features_2d[malicious_noisy, 0], features_2d[malicious_noisy, 1],
               c=color_malicious, label=f'恶意流量 (噪声, n={malicious_noisy.sum()})', 
               alpha=0.9, s=80, edgecolors='yellow', linewidths=2, marker='s')
    
    ax.set_xlabel('t-SNE 第一主成分', fontsize=12)
    ax.set_ylabel('t-SNE 第二主成分', fontsize=12)
    ax.set_title(f'特征分布: 噪声标签 (噪声率: {100*noise_mask.sum()/len(noise_mask):.1f}%)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, '2_噪声标签分布.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已保存: {save_path}")
    
    # 3. Corrected Labels (excluding dropped samples)
    logger.info("绘制 3/4: 矫正后标签分布...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    kept_samples = action_mask != 2  # Exclude dropped samples
    
    benign_corrected = kept_samples & (clean_labels == 0)
    malicious_corrected = kept_samples & (clean_labels == 1)
    dropped_samples = action_mask == 2
    
    ax.scatter(features_2d[benign_corrected, 0], features_2d[benign_corrected, 1],
               c=color_benign, label=f'正常流量 (n={benign_corrected.sum()})', 
               alpha=0.6, s=30, edgecolors='k', linewidths=0.3)
    ax.scatter(features_2d[malicious_corrected, 0], features_2d[malicious_corrected, 1],
               c=color_malicious, label=f'恶意流量 (n={malicious_corrected.sum()})', 
               alpha=0.6, s=30, edgecolors='k', linewidths=0.3)
    
    # Show dropped samples
    if dropped_samples.sum() > 0:
        ax.scatter(features_2d[dropped_samples, 0], features_2d[dropped_samples, 1],
                   c='gray', label=f'已丢弃 (n={dropped_samples.sum()})', 
                   alpha=0.5, s=50, marker='x', linewidths=1.5)
    
    ax.set_xlabel('t-SNE 第一主成分', fontsize=12)
    ax.set_ylabel('t-SNE 第二主成分', fontsize=12)
    ax.set_title('特征分布: 矫正后标签 (Hybrid Court矫正后)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, '3_矫正后标签分布.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已保存: {save_path}")
    
    # 4. Action Visualization (Keep, Flip, Drop, Reweight)
    logger.info("绘制 4/4: 矫正动作分布...")
    fig, ax = plt.subplots(figsize=(12, 9))
    
    keep_mask = action_mask == 0
    flip_mask = action_mask == 1
    drop_mask = action_mask == 2
    reweight_mask = action_mask == 3
    
    # Plot in reverse order so important actions are on top
    if reweight_mask.sum() > 0:
        ax.scatter(features_2d[reweight_mask, 0], features_2d[reweight_mask, 1],
                   c='#F77F00', label=f'重加权 (长尾样本, n={reweight_mask.sum()})', 
                   alpha=0.7, s=60, edgecolors='k', linewidths=0.5, marker='v')
    
    if keep_mask.sum() > 0:
        ax.scatter(features_2d[keep_mask, 0], features_2d[keep_mask, 1],
                   c='#06A77D', label=f'保持 (核心样本, n={keep_mask.sum()})', 
                   alpha=0.5, s=25, edgecolors='k', linewidths=0.3)
    
    if flip_mask.sum() > 0:
        ax.scatter(features_2d[flip_mask, 0], features_2d[flip_mask, 1],
                   c='#D62828', label=f'翻转 (已矫正, n={flip_mask.sum()})', 
                   alpha=0.8, s=70, edgecolors='k', linewidths=0.8, marker='*')
    
    if drop_mask.sum() > 0:
        ax.scatter(features_2d[drop_mask, 0], features_2d[drop_mask, 1],
                   c='#6A4C93', label=f'丢弃 (系统噪声, n={drop_mask.sum()})', 
                   alpha=0.8, s=80, edgecolors='k', linewidths=1, marker='X')
    
    ax.set_xlabel('t-SNE 第一主成分', fontsize=12)
    ax.set_ylabel('t-SNE 第二主成分', fontsize=12)
    ax.set_title('特征分布: Hybrid Court矫正动作', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, '4_矫正动作分布.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已保存: {save_path}")
    
    # 5. Comparison Plot (Before vs After)
    logger.info("绘制 5/5: 矫正前后对比...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Before: Noisy Labels
    ax = axes[0]
    benign_noisy_all = noisy_labels == 0
    malicious_noisy_all = noisy_labels == 1
    
    ax.scatter(features_2d[benign_noisy_all, 0], features_2d[benign_noisy_all, 1],
               c=color_benign, label=f'正常流量 (n={benign_noisy_all.sum()})', 
               alpha=0.5, s=30, edgecolors='k', linewidths=0.3)
    ax.scatter(features_2d[malicious_noisy_all, 0], features_2d[malicious_noisy_all, 1],
               c=color_malicious, label=f'恶意流量 (n={malicious_noisy_all.sum()})', 
               alpha=0.5, s=30, edgecolors='k', linewidths=0.3)
    
    # Highlight noise
    ax.scatter(features_2d[noise_mask, 0], features_2d[noise_mask, 1],
               facecolors='none', edgecolors='yellow', 
               label=f'噪声样本 (n={noise_mask.sum()})', 
               s=100, linewidths=2, marker='o')
    
    ax.set_xlabel('t-SNE 第一主成分', fontsize=12)
    ax.set_ylabel('t-SNE 第二主成分', fontsize=12)
    ax.set_title(f'矫正前: 噪声标签\n(噪声率: {100*noise_mask.sum()/len(noise_mask):.1f}%)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # After: Corrected Labels
    ax = axes[1]
    benign_corrected_all = kept_samples & (clean_labels == 0)
    malicious_corrected_all = kept_samples & (clean_labels == 1)
    
    ax.scatter(features_2d[benign_corrected_all, 0], features_2d[benign_corrected_all, 1],
               c=color_benign, label=f'正常流量 (n={benign_corrected_all.sum()})', 
               alpha=0.5, s=30, edgecolors='k', linewidths=0.3)
    ax.scatter(features_2d[malicious_corrected_all, 0], features_2d[malicious_corrected_all, 1],
               c=color_malicious, label=f'恶意流量 (n={malicious_corrected_all.sum()})', 
               alpha=0.5, s=30, edgecolors='k', linewidths=0.3)
    
    # Show corrections
    ax.scatter(features_2d[flip_mask, 0], features_2d[flip_mask, 1],
               facecolors='none', edgecolors='lime', 
               label=f'已翻转 (n={flip_mask.sum()})', 
               s=100, linewidths=2, marker='o')
    
    if drop_mask.sum() > 0:
        ax.scatter(features_2d[drop_mask, 0], features_2d[drop_mask, 1],
                   c='gray', label=f'已丢弃 (n={drop_mask.sum()})', 
                   alpha=0.6, s=50, marker='x', linewidths=1.5)
    
    # Calculate correction accuracy
    correction_accuracy = (clean_labels[kept_samples] == y_true[kept_samples]).mean()
    
    ax.set_xlabel('t-SNE 第一主成分', fontsize=12)
    ax.set_ylabel('t-SNE 第二主成分', fontsize=12)
    ax.set_title(f'矫正后: Hybrid Court\n(准确率: {correction_accuracy*100:.2f}%)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, '5_矫正前后对比.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ 已保存: {save_path}")
    
    logger.info("="*70)
    logger.info("✓ 所有特征分布图已生成")


def main(args):
    """Main function"""
    global logger
    
    # Setup
    set_seed(config.SEED)
    config.create_dirs()
    
    # 噪声率百分比
    noise_pct = int(args.noise_rate * 100)
    
    # Create analysis output directory (包含噪声率)
    analysis_dir = os.path.join(config.LABEL_CORRECTION_DIR, "analysis", f"noise_{noise_pct}pct")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "documents"), exist_ok=True)
    
    # 创建带噪声率的日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"noise_{noise_pct}pct_analysis_{timestamp}.log"
    log_dir = os.path.join(config.OUTPUT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    # 创建共享的handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    # 配置root logger (捕获所有模块的日志，包括HybridCourt)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # 清除已有handlers
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)
    
    # 获取本模块的logger (使用root logger，不需要单独配置)
    logger = logging.getLogger('label_correction_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除handlers，使用root的
    logger.propagate = True  # 传播到root logger
    
    logger.info("="*70)
    logger.info(f"MEDAL-Lite 标签矫正分析 - 噪声率 {noise_pct}%")
    logger.info("="*70)
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"设备: {config.DEVICE}")
    logger.info(f"噪声率: {noise_pct}%")
    logger.info("")
    
    # Override config with arguments
    if args.noise_rate is not None:
        config.LABEL_NOISE_RATE = args.noise_rate
    
    # ========================
    # 1. Load Dataset
    # ========================
    logger.info("="*70)
    logger.info("Step 1: 加载数据集")
    logger.info("="*70)
    logger.info(f"正常训练数据:    {config.BENIGN_TRAIN}")
    logger.info(f"恶意训练数据:    {config.MALICIOUS_TRAIN}")
    logger.info(f"序列长度:        {config.SEQUENCE_LENGTH}")
    logger.info("")
    
    # 优先使用预处理好的数据
    if check_preprocessed_exists('train'):
        logger.info("✓ 发现预处理文件，直接加载...")
        X_train, y_train_clean, train_files = load_preprocessed('train')
        logger.info(f"  从预处理文件加载: {X_train.shape[0]} 个样本")
    else:
        # 从PCAP文件加载
        logger.info("开始加载训练数据集（从PCAP文件）...")
        logger.info("💡 提示: 运行 'python scripts/utils/preprocess.py --train_only' 可预处理训练集，加速后续分析")
        X_train, y_train_clean, train_files = load_dataset(
            benign_dir=config.BENIGN_TRAIN,
            malicious_dir=config.MALICIOUS_TRAIN,
            sequence_length=config.SEQUENCE_LENGTH
        )
    
    if X_train is None:
        logger.error("❌ 数据集加载失败!")
        return
    
    logger.info("✓ 数据集加载完成")
    logger.info(f"  数据形状:     {X_train.shape}")
    logger.info(f"  正常样本:     {(y_train_clean==0).sum()}")
    logger.info(f"  恶意样本:     {(y_train_clean==1).sum()}")
    logger.info("")
    
    # ========================
    # 2. Inject Label Noise
    # ========================
    logger.info("="*70)
    logger.info("Step 2: 注入标签噪声")
    logger.info("="*70)
    logger.info(f"噪声率: {config.LABEL_NOISE_RATE*100:.0f}%")
    
    # 固定随机种子，确保相同噪声率的结果可复现
    set_seed(config.SEED)
    y_train_noisy, noise_mask = inject_label_noise(y_train_clean, config.LABEL_NOISE_RATE)
    
    logger.info(f"✓ 噪声注入完成: {noise_mask.sum()} 个标签被翻转")
    logger.info(f"  原始: 正常={(y_train_clean==0).sum()}, 恶意={(y_train_clean==1).sum()}")
    logger.info(f"  噪声后: 正常={(y_train_noisy==0).sum()}, 恶意={(y_train_noisy==1).sum()}")
    logger.info("")
    
    # ========================
    # 3. Extract Features
    # ========================
    logger.info("="*70)
    logger.info("Step 3: 提取特征")
    logger.info("="*70)
    
    backbone = MicroBiMambaBackbone(config)
    
    # 确定backbone路径：优先使用命令行参数，否则使用默认路径
    if args.backbone_path:
        backbone_path = args.backbone_path
    else:
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
    
    # Try to load pre-trained backbone
    if os.path.exists(backbone_path) and not args.retrain_backbone:
        logger.info(f"加载预训练backbone: {backbone_path}")
        state = torch.load(backbone_path, map_location=config.DEVICE)
        try:
            backbone.load_state_dict(state)
        except RuntimeError as e:
            logger.warning(f"⚠ 骨干网络检查点与当前结构不完全匹配，将使用 strict=False 加载: {e}")
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"  missing_keys: {missing}")
            if unexpected:
                logger.warning(f"  unexpected_keys: {unexpected}")
        logger.info("✓ Backbone加载完成")
    else:
        if args.retrain_backbone:
            logger.warning("⚠ 指定了--retrain_backbone，将使用随机初始化的backbone")
        else:
            logger.warning(f"⚠ 未找到预训练backbone: {backbone_path}")
            logger.warning("  将使用随机初始化的backbone")
    
    features = extract_features_with_backbone(backbone, X_train, config, logger)
    
    # Save features
    features_path = os.path.join(analysis_dir, "extracted_features.npy")
    np.save(features_path, features)
    logger.info(f"✓ 特征已保存: {features_path}")
    logger.info("")
    
    # ========================
    # 4. Run Hybrid Court Label Correction
    # ========================
    logger.info("="*70)
    logger.info("Step 4: 运行 Hybrid Court 标签矫正")
    logger.info("="*70)
    
    results = run_hybrid_court_with_detailed_tracking(features, y_train_noisy, config, logger, y_true=y_train_clean)
    
    # Save results
    results_path = os.path.join(analysis_dir, "correction_results.npz")
    np.savez(results_path,
             y_clean=y_train_clean,
             y_noisy=y_train_noisy,
             y_corrected=results['clean_labels'],
             action_mask=results['action_mask'],
             confidence=results['confidence'],
             correction_weight=results['correction_weight'],
             noise_mask=noise_mask,
             features=features,
             cl_suspected_noise=results['cl_suspected_noise'],
             cl_pred_labels=results['cl_pred_labels'],
             cl_pred_probs=results['cl_pred_probs'],
             made_is_dense=results['made_is_dense'],
             made_density_scores=results['made_density_scores'],
             knn_neighbor_labels=results['knn_neighbor_labels'],
             knn_neighbor_consistency=results['knn_neighbor_consistency'],
             tier_info=np.array(results.get('tier_info', []), dtype=object))
    logger.info(f"✓ 矫正结果已保存: {results_path}")
    logger.info("")
    
    # ========================
    # 5. Generate Analysis Document
    # ========================
    logger.info("="*70)
    logger.info("Step 5: 生成样本分析文档")
    logger.info("="*70)
    
    doc_path = os.path.join(analysis_dir, "documents", "sample_analysis.log")
    df_analysis = generate_sample_analysis_document(
        results, y_train_clean, noise_mask, doc_path, logger
    )
    logger.info("")
    
    # ========================
    # 6. Generate Feature Distribution Plots
    # ========================
    logger.info("="*70)
    logger.info("Step 6: 生成特征分布图")
    logger.info("="*70)
    
    plot_dir = os.path.join(analysis_dir, "figures")
    plot_feature_distributions(results, y_train_clean, noise_mask, plot_dir, logger)
    logger.info("")
    
    # ========================
    # Summary
    # ========================
    logger.info("="*70)
    logger.info(f"🎉 噪声率 {noise_pct}% 标签矫正分析完成!")
    logger.info("="*70)
    logger.info("")
    logger.info("📁 输出文件:")
    logger.info(f"  分析目录:       {analysis_dir}")
    logger.info(f"  矫正结果:       {results_path}")
    logger.info(f"  样本分析CSV:    {doc_path.replace('.log', '.csv')}")
    logger.info(f"  样本分析LOG:    {doc_path}")
    logger.info(f"  特征分布图:     {plot_dir}")
    logger.info(f"  日志文件:       {os.path.join(log_dir, log_filename)}")
    logger.info("")
    logger.info("📊 统计摘要:")
    n_samples = len(y_train_clean)
    action_mask = results['action_mask']
    logger.info(f"  总样本数:           {n_samples}")
    logger.info(f"  噪声注入:           {noise_mask.sum()} ({100*noise_mask.sum()/n_samples:.1f}%)")
    logger.info(f"  Keep (core):        {(action_mask==0).sum()} ({100*(action_mask==0).sum()/n_samples:.1f}%)")
    logger.info(f"  Flip (corrected):   {(action_mask==1).sum()} ({100*(action_mask==1).sum()/n_samples:.1f}%)")
    logger.info(f"  Drop (noise):       {(action_mask==2).sum()} ({100*(action_mask==2).sum()/n_samples:.1f}%)")
    logger.info(f"  Reweight (tail):    {(action_mask==3).sum()} ({100*(action_mask==3).sum()/n_samples:.1f}%)")
    
    # Correction accuracy
    keep_mask = action_mask != 2
    correction_accuracy = (results['clean_labels'][keep_mask] == y_train_clean[keep_mask]).mean()
    logger.info(f"  矫正准确率:         {correction_accuracy*100:.2f}%")
    logger.info("")
    logger.info("="*70)
    
    # 清理root logger的handlers
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label Correction Analysis for MEDAL-Lite"
    )
    parser.add_argument(
        "--noise_rate", 
        type=float, 
        default=0.30, 
        help="Label noise rate (default: 0.30)"
    )
    parser.add_argument(
        "--retrain_backbone",
        action='store_true',
        help="Use randomly initialized backbone instead of pre-trained"
    )
    parser.add_argument(
        "--backbone_path",
        type=str,
        default='',
        help="Path to backbone model (optional, default: backbone_pretrained.pth)"
    )
    
    args = parser.parse_args()
    
    main(args)

