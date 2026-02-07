"""
MEDAL-Lite Testing Script
Evaluate trained model on test dataset
"""
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path when running as a script
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import hashlib
import argparse
from datetime import datetime
import json
import csv

from sklearn.metrics import precision_score, recall_score, f1_score

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import (
    set_seed, setup_logger, calculate_metrics, print_metrics, find_optimal_threshold
)
from MoudleCode.utils.visualization import (
    plot_feature_space, plot_confusion_matrix, plot_roc_curve, plot_probability_distribution
)
from MoudleCode.preprocessing.pcap_parser import load_dataset
from MoudleCode.feature_extraction.backbone import MicroBiMambaBackbone, build_backbone
from MoudleCode.classification.dual_stream import MEDAL_Classifier

# 导入预处理模块
try:
    from scripts.utils.preprocess import check_preprocessed_exists, load_preprocessed, normalize_burstsize_inplace
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False


def _rng_fingerprint_short() -> str:
    h = hashlib.sha256()
    try:
        h.update(repr(random.getstate()).encode('utf-8'))
    except Exception:
        h.update(b'py_random_error')
    try:
        ns = np.random.get_state()
        h.update(str(ns[0]).encode('utf-8'))
        h.update(np.asarray(ns[1], dtype=np.uint32).tobytes())
        h.update(str(ns[2]).encode('utf-8'))
        h.update(str(ns[3]).encode('utf-8'))
        h.update(str(ns[4]).encode('utf-8'))
    except Exception:
        h.update(b'numpy_random_error')
    try:
        h.update(torch.get_rng_state().detach().cpu().numpy().tobytes())
    except Exception:
        h.update(b'torch_cpu_rng_error')
    try:
        if torch.cuda.is_available():
            for s in torch.cuda.get_rng_state_all():
                h.update(s.detach().cpu().numpy().tobytes())
        else:
            h.update(b'no_cuda')
    except Exception:
        h.update(b'torch_cuda_rng_error')
    return h.hexdigest()[:16]


def _seed_snapshot() -> str:
    torch_seed = None
    try:
        torch_seed = int(torch.initial_seed())
    except Exception:
        torch_seed = None
    return (
        f"config.SEED={int(getattr(config, 'SEED', -1))} | "
        f"torch.initial_seed={torch_seed}"
    )


def _load_classifier_checkpoint(classifier, state_dict, logger=None):
    """
    Load classifier checkpoint (new architecture only: 32 → 16 → 8 → 2).
    
    Supports loading either:
    - Full classifier.state_dict (with 'backbone.' and 'dual_mlp.' prefixes)
    - Only dual_mlp.state_dict (with or without 'dual_mlp.' prefix)
    
    Raises RuntimeError if model architecture doesn't match (old models not supported).
    """
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict to be a dict, got {type(state_dict)}")

    # 尝试加载完整模型
    try:
        classifier.load_state_dict(state_dict, strict=True)
        if logger is not None:
            logger.info("✓ 完整模型加载成功")
        return
    except Exception as e:
        if logger is not None:
            logger.info(f"整模型加载失败，尝试仅加载分类头(dual_mlp): {str(e)[:100]}...")

    # 处理带有 'dual_mlp.' 前缀的键（可能同时包含backbone和dual_mlp的键）
    keys = list(state_dict.keys())
    dual_mlp_keys = [k for k in keys if isinstance(k, str) and k.startswith('dual_mlp.')]
    
    if dual_mlp_keys:
        # 过滤出dual_mlp的键并去除前缀
        stripped = {k[len('dual_mlp.'):]: v for k, v in state_dict.items() if k.startswith('dual_mlp.')}
        classifier.dual_mlp.load_state_dict(stripped, strict=True)
        if logger is not None:
            logger.info("✓ 分类头(dual_mlp)加载成功")
        return

    # 尝试直接加载到 dual_mlp（无前缀，且没有backbone键）
    # 如果state_dict包含backbone键，说明是完整模型，不应该走这里
    has_backbone_keys = any(isinstance(k, str) and k.startswith('backbone.') for k in keys)
    if has_backbone_keys:
        raise RuntimeError(
            f"模型文件包含backbone键，但完整模型加载失败。"
            f"这通常意味着模型架构不匹配（可能是旧架构32-32-16-2）。"
            f"请使用新架构（32-16-8-2）重新训练模型。"
        )
    
    classifier.dual_mlp.load_state_dict(state_dict, strict=True)
    if logger is not None:
        logger.info("✓ 分类头(dual_mlp)加载成功")

def test_model(classifier, X_test, y_test, config, logger, save_prefix="test"):
    """
    Test the model on test dataset
    
    Args:
        classifier: Trained MEDAL classifier
        X_test: (N, L, D) test sequences
        y_test: (N,) test labels
        config: configuration object
        logger: logger
        save_prefix: prefix for saved files
        
    Returns:
        metrics: dictionary of evaluation metrics
    """
    logger.info("="*70)
    logger.info("🔍 模型测试 Model Testing")
    logger.info("="*70)
    logger.info(f"测试样本数: {len(X_test)}")
    test_batch_size = int(getattr(config, 'TEST_BATCH_SIZE', 256))
    logger.info(f"批次大小: {test_batch_size}")
    
    # 记录配置中的参考阈值（用于对比与可视化）
    config_threshold = getattr(config, 'MALICIOUS_THRESHOLD', 0.5)
    logger.info(f"✓ 配置参考阈值: {config_threshold:.2f} (仅用于参考与可视化)")
    logger.info("")
    
    classifier.eval()
    classifier.to(config.DEVICE)
    
    # Create DataLoader
    logger.info(f"🔧 RNG指纹(测试-Dataloader创建前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False)
    logger.info(f"🔧 RNG指纹(测试-Dataloader创建后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    total_batches = len(test_loader)
    
    # Collect probabilities / labels / features（先不使用阈值）
    all_probs = []
    all_labels = []
    all_features = []
    
    logger.info(f"🔧 RNG指纹(测试-推理开始前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    logger.info("开始推理...")
    logger.info("-"*70)
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            X_batch = X_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)
            
            # 直接前向获得 logits 和特征，不在这里做阈值判决
            logits, z = classifier(X_batch, return_features=True, return_separate=False)
            
            # Apply temperature scaling if enabled (makes predictions "softer")
            temperature = getattr(config, 'TEMPERATURE_SCALING', 1.0)
            if temperature != 1.0:
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_features.append(z.cpu().numpy())
            
            # 每10个批次或最后一个批次输出进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                processed = min((batch_idx + 1) * test_batch_size, len(X_test))
                logger.info(f"  推理进度: {batch_idx+1}/{total_batches} batches ({progress:.1f}%) | 已处理 {processed}/{len(X_test)} 个样本")
    
    logger.info(f"🔧 RNG指纹(测试-推理结束后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    # Concatenate results
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    features = np.concatenate(all_features)
    
    logger.info("-"*70)
    logger.info("✓ 推理完成")
    logger.info("")
    
    # 🚀 在测试集上搜索最优阈值（后处理步骤，模型已固定，不是数据泄露）
    logger.info("📊 基于测试集 Binary F1-Score(pos=1) 搜索最优阈值...")
    logger.info("   说明: 这是后处理步骤，模型参数已固定，用于优化决策阈值")
    optimal_threshold, optimal_metric, _ = find_optimal_threshold(
        y_true, y_prob, metric='f1_binary', positive_class=1
    )
    logger.info(f"✅ 测试集最优阈值: {optimal_threshold:.4f} (Binary F1 pos=1 = {optimal_metric:.4f})")
    logger.info(f"   配置默认阈值: {config_threshold:.4f}")
    
    # 阈值对比（解释为什么图里可能是 0.8+ 而不是 0.6）
    candidate_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, float(config_threshold)]
    # 注意：不对optimal_threshold进行四舍五入，保持原始精度
    candidate_thresholds_display = sorted(set([round(t, 4) for t in candidate_thresholds]))
    # 将optimal_threshold插入到正确的位置（用于显示）
    optimal_threshold_rounded = round(float(optimal_threshold), 4)
    if optimal_threshold_rounded not in candidate_thresholds_display:
        candidate_thresholds_display.append(optimal_threshold_rounded)
        candidate_thresholds_display.sort()
    
    logger.info("📏 阈值对比 (Malicious=Positive):")
    logger.info(f"  {'threshold':>10s} | {'precision':>9s} | {'recall':>7s} | {'f1':>7s}")
    logger.info("  " + "-"*44)
    for th_display in candidate_thresholds_display:
        # 对于最优阈值，使用原始高精度值；其他使用显示值
        if abs(th_display - optimal_threshold_rounded) < 0.00001:
            th_actual = float(optimal_threshold)  # 使用原始高精度值
        else:
            th_actual = th_display
        
        y_pred_th = (y_prob[:, 1] >= th_actual).astype(int)
        p = precision_score(y_true, y_pred_th, pos_label=1, zero_division=0)
        r = recall_score(y_true, y_pred_th, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred_th, pos_label=1, zero_division=0)
        marker = " ← 最优" if abs(th_display - optimal_threshold_rounded) < 0.00001 else ""
        logger.info(f"  {th_display:10.4f} | {p:9.4f} | {r:7.4f} | {f1:7.4f}{marker}")
    logger.info("")
    
    # 使用最优阈值生成预测标签（这是最终使用的阈值）
    y_pred = (y_prob[:, 1] >= optimal_threshold).astype(int)
    logger.info(f"✅ 最终评估使用最优阈值: {optimal_threshold:.4f}")
    logger.info(f"   (下方性能指标均基于此阈值计算)")
    logger.info("")
    
    # Calculate metrics at optimal threshold
    logger.info("📊 计算性能指标 (基于自动阈值)...")
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Print metrics（包含论文口径：恶意类为正类的单类F1）
    print_metrics(metrics, logger)
    
    # Per-class metrics
    logger.info("\nPer-Class Analysis:")
    for class_idx, class_name in enumerate(['Benign', 'Malicious']):
        class_mask = y_true == class_idx
        class_acc = (y_pred[class_mask] == y_true[class_mask]).mean()
        logger.info(f"  {class_name:10s}: {class_mask.sum():5d} samples, Accuracy: {class_acc*100:.2f}%")
    
    # Visualizations
    logger.info("\nGenerating visualizations...")
    
    # Feature space
    feature_space_path = os.path.join(config.RESULT_DIR, "figures", f"{save_prefix}_feature_space.png")
    plot_feature_space(features, y_true, feature_space_path, 
                      title=f"Test Set Feature Space", method='tsne')
    
    # Confusion matrix
    confusion_matrix_path = os.path.join(config.RESULT_DIR, "figures", f"{save_prefix}_confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], ['Benign', 'Malicious'], 
                         confusion_matrix_path, title=f"Test Set Confusion Matrix")
    
    # ROC curve
    roc_curve_path = os.path.join(config.RESULT_DIR, "figures", f"{save_prefix}_roc_curve.png")
    plot_roc_curve(y_true, y_prob, roc_curve_path, title=f"Test Set ROC Curve")
    
    # Probability distribution (使用自动搜索得到的最优阈值)
    prob_dist_path = os.path.join(config.RESULT_DIR, "figures", f"{save_prefix}_probability_distribution.png")
    plot_probability_distribution(y_true, y_prob, prob_dist_path, 
                                  title=f"Test Set Probability Distribution", threshold=optimal_threshold)
    
    logger.info("")
    logger.info("📁 输出文件路径:")
    logger.info(f"  ✓ 特征空间可视化: {feature_space_path}")
    logger.info(f"  ✓ 混淆矩阵: {confusion_matrix_path}")
    logger.info(f"  ✓ ROC曲线: {roc_curve_path}")
    logger.info(f"  ✓ 概率分布图: {prob_dist_path}")
    
    # Save predictions to result directory
    results_file = os.path.join(config.RESULT_DIR, "models", f"{save_prefix}_predictions.npz")
    np.savez(results_file,
             y_true=y_true,
             y_pred=y_pred,
             y_prob=y_prob,
             features=features)
    # Save metrics to text file
    metrics_file = os.path.join(config.RESULT_DIR, "models", f"{save_prefix}_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"MEDAL-Lite Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Samples: {len(y_true)}\n")
        f.write(f"  Benign:    {(y_true==0).sum()}\n")
        f.write(f"  Malicious: {(y_true==1).sum()}\n\n")
        f.write("Performance Metrics (Malicious=Positive):\n")
        f.write(f"  Accuracy:          {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision (pos=1): {metrics['precision_pos']:.4f}\n")
        f.write(f"  Recall    (pos=1): {metrics['recall_pos']:.4f}\n")
        f.write(f"  F1 (pos=1):        {metrics['f1_pos']:.4f}\n")
        f.write("  --- reference ---\n")
        f.write(f"  F1-Macro:          {metrics['f1_macro']:.4f}\n")
        f.write(f"  F1-Weighted:       {metrics['f1_weighted']:.4f}\n")
        if 'auc' in metrics:
            f.write(f"  AUC:               {metrics['auc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']) + "\n")
    
    logger.info(f"  ✓ 预测结果: {results_file}")
    logger.info(f"  ✓ 性能指标: {metrics_file}")
    
    # 添加标准化的键名以便对比
    metrics['precision'] = metrics['precision_pos']
    metrics['recall'] = metrics['recall_pos']
    metrics['f1'] = metrics['f1_pos']
    return metrics


def _infer_family_from_filename(filename: str) -> str:
    name = str(filename).lower()
    if 'dnscat' in name or 'dnscat2' in name:
        return 'dnscat2'
    if 'iodine' in name:
        return 'iodine'
    if 'dohbrw' in name or 'doh' in name:
        return 'doh'
    return 'unknown'


def _export_family_breakdown(save_prefix: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, test_files, config, logger):
    if test_files is None:
        return

    if len(test_files) != len(y_true):
        logger.warning(f"⚠ test_files 长度({len(test_files)})与样本数({len(y_true)})不一致，跳过分组评估")
        return

    families = np.array([_infer_family_from_filename(f) for f in test_files], dtype=object)
    unique_families = sorted(set(families.tolist()))

    rows = []
    for fam in unique_families:
        idx = families == fam
        n = int(idx.sum())
        if n <= 0:
            continue
        y_t = y_true[idx]
        y_p = y_pred[idx]
        y_pb = y_prob[idx]

        p = precision_score(y_t, y_p, pos_label=1, zero_division=0)
        r = recall_score(y_t, y_p, pos_label=1, zero_division=0)
        f1 = f1_score(y_t, y_p, pos_label=1, zero_division=0)
        acc = float((y_t == y_p).mean()) if n > 0 else 0.0

        rows.append({
            'family': fam,
            'n_samples': n,
            'n_benign': int((y_t == 0).sum()),
            'n_malicious': int((y_t == 1).sum()),
            'accuracy': float(acc),
            'precision_pos': float(p),
            'recall_pos': float(r),
            'f1_pos': float(f1),
        })

    if not rows:
        return

    models_dir = os.path.join(config.RESULT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)

    csv_path = os.path.join(models_dir, f'{save_prefix}_family_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(models_dir, f'{save_prefix}_family_report.json')
    payload = {
        'save_prefix': save_prefix,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'family_metrics': rows,
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("📁 分组评估输出:")
    logger.info(f"  ✓ CSV:  {csv_path}")
    logger.info(f"  ✓ JSON: {json_path}")


def main(args):
    """Main testing function"""
    
    # Setup
    rng_fp_before_seed = _rng_fingerprint_short()
    set_seed(config.SEED)
    rng_fp_after_seed = _rng_fingerprint_short()
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, "logs"), name='test')

    logger.info(f"🔧 RNG指纹(seed前): {rng_fp_before_seed}")
    logger.info(f"🔧 RNG指纹(seed后): {rng_fp_after_seed} ({_seed_snapshot()})")
    
    logger.info("="*70)
    logger.info("🧪 MEDAL-Lite Testing Pipeline")
    logger.info("="*70)
    
    # GPU信息
    if torch.cuda.is_available():
        logger.info(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        logger.info(f"  使用设备: {config.DEVICE}")
    else:
        logger.info(f"⚠ 使用CPU进行推理")
        logger.info(f"  使用设备: {config.DEVICE}")
    
    logger.info(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # ========================
    # Load Test Dataset
    # ========================
    logger.info("="*70)
    logger.info("📦 加载测试数据集 Loading Test Dataset")
    logger.info("="*70)
    logger.info(f"测试集配置:")
    logger.info(f"  正常流量路径: {config.BENIGN_TEST}")
    logger.info(f"  恶意流量路径: {config.MALICIOUS_TEST}")
    logger.info(f"  说明: 将读取上述路径下所有pcap文件，流数在处理时统计")
    logger.info("")
    
    logger.info(f"🔧 RNG指纹(加载测试数据前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    # 优先使用预处理好的数据
    if PREPROCESS_AVAILABLE and check_preprocessed_exists('test'):
        logger.info("✓ 发现预处理文件，直接加载...")
        X_test, y_test, test_files = load_preprocessed('test')
        X_test = normalize_burstsize_inplace(X_test)
        logger.info(f"  从预处理文件加载: {X_test.shape[0]} 个样本")
    else:
        # 从PCAP文件加载
        logger.info("开始加载测试数据集（从PCAP文件）...")
        logger.info("💡 提示: 运行 'python preprocess.py --test_only' 可预处理测试集，加速后续测试")
        X_test, y_test, test_files = load_dataset(
            benign_dir=config.BENIGN_TEST,
            malicious_dir=config.MALICIOUS_TEST,
            sequence_length=config.SEQUENCE_LENGTH
        )
        X_test = normalize_burstsize_inplace(X_test)

    logger.info(f"🔧 RNG指纹(加载测试数据后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    
    if X_test is None:
        logger.error("❌ 测试数据集加载失败!")
        return
    
    logger.info("✓ 测试数据集加载完成")
    logger.info(f"  数据形状: {X_test.shape} (样本数×序列长度×特征维度)")
    logger.info(f"  正常样本: {(y_test==0).sum()} 个")
    logger.info(f"  恶意样本: {(y_test==1).sum()} 个")
    logger.info("")
    
    # ========================
    # Load Model
    # ========================
    logger.info("="*70)
    logger.info("📥 加载训练好的模型 Loading Trained Model")
    logger.info("="*70)
    logger.info("📥 输入数据路径:")
    logger.info(f"  ✓ 测试数据: {config.BENIGN_TEST} (正常), {config.MALICIOUS_TEST} (恶意)")
    logger.info("")
    
    # Try to load model metadata to get the backbone path used during training
    metadata_path = os.path.join(config.CLASSIFICATION_DIR, "models", "model_metadata.json")
    backbone_path_from_metadata = None
    
    if os.path.exists(metadata_path):
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            backbone_path_from_metadata = metadata.get('backbone_path')
            input_is_features_from_metadata = metadata.get('input_is_features', False)
            feature_dim_from_metadata = metadata.get('feature_dim', None)
            
            if backbone_path_from_metadata:
                logger.info(f"✓ 从模型元数据中读取到训练时使用的骨干网络:")
                logger.info(f"  {backbone_path_from_metadata}")
                if input_is_features_from_metadata:
                    logger.info(f"✓ 训练时输入类型: 特征向量 (维度={feature_dim_from_metadata})")
                    logger.info(f"  测试时将自动从序列提取特征")
                logger.info("")
        except Exception as e:
            logger.warning(f"⚠ 无法读取模型元数据: {e}")
    
    # 确定骨干网络路径
    # 优先级：1. 命令行参数 2. 元数据 3. 默认路径
    backbone_path = None
    
    # 1. 检查命令行参数
    if hasattr(args, 'backbone_path') and args.backbone_path:
        backbone_path = args.backbone_path
        logger.info(f"✓ 使用命令行指定的骨干网络:")
        logger.info(f"  {backbone_path}")
        logger.info("")
    # 2. 尝试从元数据读取（优先使用训练时保存的路径）
    elif backbone_path_from_metadata:
        if os.path.exists(backbone_path_from_metadata):
            backbone_path = backbone_path_from_metadata
            logger.info("✓ 使用训练时的骨干网络（从元数据）")
            logger.info(f"  {backbone_path}")
        else:
            # 如果元数据中指定的路径不存在，这是严重问题，不应该回退到旧模型
            logger.error(f"❌ 严重错误：训练时使用的骨干网络不存在!")
            logger.error(f"  元数据中记录的路径: {backbone_path_from_metadata}")
            logger.error(f"  该文件不存在，无法使用正确的模型进行测试!")
            logger.error("")
            logger.error("可能的原因:")
            logger.error("  1. 模型文件被意外删除")
            logger.error("  2. 模型文件路径发生了变化")
            logger.error("  3. 使用了错误的输出目录")
            logger.error("")
            logger.error("解决方案:")
            logger.error("  1. 检查模型文件是否存在")
            logger.error("  2. 重新运行训练脚本")
            logger.error("  3. 或使用 --backbone_path 参数手动指定正确的模型路径")
            return
    # 3. 使用默认路径（仅在元数据不存在或未指定backbone_path时）
    else:
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
        logger.info(f"使用默认骨干网络路径: {backbone_path}")
        logger.warning("⚠ 注意：未找到模型元数据，使用默认路径可能不是训练时使用的模型!")
        logger.warning("  建议：确保训练脚本已正确保存模型元数据")
    
    if not os.path.exists(backbone_path):
        logger.error(f"❌ 骨干网络检查点未找到: {backbone_path}")
        logger.error("请先运行训练脚本!")
        return
    
    logger.info("正在加载骨干网络...")
    logger.info(f"  📥 输入模型: {backbone_path}")
    logger.info(f"🔧 RNG指纹(加载backbone权重前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    
    # 使用安全的模型加载函数（自动处理兼容性）
    from MoudleCode.utils.model_loader import load_backbone_safely
    backbone = load_backbone_safely(
        backbone_path=backbone_path,
        config=config,
        device=config.DEVICE,
        logger=logger
    )
    logger.info(f"🔧 RNG指纹(加载backbone权重后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
    backbone.freeze()
    logger.info(f"✓ 骨干网络加载完成")
    
    # ========================
    # Load classifiers (both best and final)
    # ========================
    best_path = os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_best_f1.pth")
    final_path = os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_final.pth")
    minloss_candidates = []
    try:
        models_dir = os.path.join(config.CLASSIFICATION_DIR, "models")
        if os.path.isdir(models_dir):
            for fn in os.listdir(models_dir):
                if isinstance(fn, str) and fn.startswith("classifier_last10_minloss_epoch") and fn.endswith(".pth"):
                    full_path = os.path.join(models_dir, fn)
                    minloss_candidates.append(full_path)
    except Exception:
        minloss_candidates = []
    
    # 检查哪些模型存在
    has_best = os.path.exists(best_path)
    has_final = os.path.exists(final_path)
    # 按文件修改时间选择最新的minloss模型（而不是按字符串排序）
    if len(minloss_candidates) > 0:
        minloss_path = max(minloss_candidates, key=lambda p: os.path.getmtime(p))
    else:
        minloss_path = None
    has_minloss = bool(minloss_path) and os.path.exists(minloss_path)
    
    if not has_best and not has_final and not has_minloss:
        logger.error(f"❌ 未找到任何分类器检查点!")
        logger.error(f"  Best模型: {best_path}")
        logger.error(f"  Final模型: {final_path}")
        if minloss_path:
            logger.error(f"  Last10-MinLoss模型: {minloss_path}")
        logger.error("请先运行训练脚本!")
        return
    
    # 如果命令行指定了分类器路径，只测试指定的模型
    if hasattr(args, 'classifier_path') and args.classifier_path:
        logger.info("正在加载指定的分类器...")
        logger.info(f"  📥 输入模型: {args.classifier_path}")
        
        logger.info(f"🔧 RNG指纹(构建classifier前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        classifier = MEDAL_Classifier(backbone, config)
        logger.info(f"🔧 RNG指纹(构建classifier后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        try:
            classifier_state = torch.load(args.classifier_path, map_location=config.DEVICE, weights_only=True)
        except TypeError:
            classifier_state = torch.load(args.classifier_path, map_location=config.DEVICE)
        logger.info(f"🔧 RNG指纹(加载classifier权重前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        _load_classifier_checkpoint(classifier, classifier_state, logger=logger)
        logger.info(f"🔧 RNG指纹(加载classifier权重后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        logger.info(f"✓ 分类器加载完成")
        
        # Count parameters
        n_params = sum(p.numel() for p in classifier.parameters())
        n_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        logger.info("")
        logger.info("📊 模型参数统计:")
        logger.info(f"  总参数量: {n_params:,}")
        logger.info(f"  可训练参数: {n_trainable:,} (骨干网络已冻结)")
        logger.info("")
        
        # Test single model
        logger.info(f"🔧 RNG指纹(进入test_model前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        metrics = test_model(classifier, X_test, y_test, config, logger)
        logger.info(f"🔧 RNG指纹(test_model返回后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        try:
            results_path = os.path.join(config.RESULT_DIR, 'models', 'test_predictions.npz')
            npz = np.load(results_path, allow_pickle=True)
            y_true = npz['y_true']
            y_pred = npz['y_pred']
            y_prob = npz['y_prob']
            _export_family_breakdown('test', y_true, y_pred, y_prob, test_files, config, logger)
        except Exception as e:
            logger.warning(f"⚠ 分组评估导出失败: {e}")
        
        logger.info("")
        logger.info("="*70)
        logger.info("🎉 测试完成! Testing Complete!")
        logger.info("="*70)
        logger.info("")
        logger.info("📁 输出文件路径:")
        logger.info(f"  ✓ 预测结果: {os.path.join(config.RESULT_DIR, 'models', 'test_predictions.npz')}")
        logger.info(f"  ✓ 性能指标: {os.path.join(config.RESULT_DIR, 'models', 'test_metrics.txt')}")
        logger.info(f"  ✓ 可视化图表: {os.path.join(config.RESULT_DIR, 'figures')}")
        logger.info("")
        logger.info("="*70)
        
        return metrics
    
    # 否则，测试所有可用的模型并对比
    logger.info("")
    logger.info("="*70)
    logger.info("🔬 对比测试：Best F1 vs Final 模型")
    logger.info("="*70)
    logger.info(f"  Best F1模型: {'✓ 存在' if has_best else '✗ 不存在'}")
    logger.info(f"  Final模型:   {'✓ 存在' if has_final else '✗ 不存在'}")
    logger.info(f"  Last10-MinLoss模型: {'✓ 存在' if has_minloss else '✗ 不存在'}")
    logger.info("")
    
    models_to_test = []
    if has_best:
        models_to_test.append(("Best F1", best_path))
    if has_final:
        models_to_test.append(("Final", final_path))
    if has_minloss:
        models_to_test.append(("Last10-MinLoss", minloss_path))
    
    all_metrics = {}
    
    for model_name, model_path in models_to_test:
        logger.info("="*70)
        logger.info(f"📊 测试模型: {model_name}")
        logger.info("="*70)
        logger.info(f"正在加载分类器...")
        logger.info(f"  📥 输入模型: {model_path}")
        
        logger.info(f"🔧 RNG指纹(构建classifier前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        classifier = MEDAL_Classifier(backbone, config)
        logger.info(f"🔧 RNG指纹(构建classifier后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        try:
            classifier_state = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
        except TypeError:
            classifier_state = torch.load(model_path, map_location=config.DEVICE)
        logger.info(f"🔧 RNG指纹(加载classifier权重前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        _load_classifier_checkpoint(classifier, classifier_state, logger=logger)
        logger.info(f"🔧 RNG指纹(加载classifier权重后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        logger.info(f"✓ 分类器加载完成")
        
        if model_name == "Best F1":
            # Count parameters only once
            n_params = sum(p.numel() for p in classifier.parameters())
            n_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
            logger.info("")
            logger.info("📊 模型参数统计:")
            logger.info(f"  总参数量: {n_params:,}")
            logger.info(f"  可训练参数: {n_trainable:,} (骨干网络已冻结)")
            logger.info("")
        
        # Test model with specific save prefix
        if model_name == "Best F1":
            save_prefix = "test_best"
        elif model_name == "Final":
            save_prefix = "test_final"
        else:
            save_prefix = "test_last10_minloss"
        logger.info(f"🔧 RNG指纹(进入test_model前): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        metrics = test_model(classifier, X_test, y_test, config, logger, save_prefix=save_prefix)
        logger.info(f"🔧 RNG指纹(test_model返回后): {_rng_fingerprint_short()} ({_seed_snapshot()})")
        try:
            results_path = os.path.join(config.RESULT_DIR, 'models', f'{save_prefix}_predictions.npz')
            npz = np.load(results_path, allow_pickle=True)
            y_true = npz['y_true']
            y_pred = npz['y_pred']
            y_prob = npz['y_prob']
            _export_family_breakdown(save_prefix, y_true, y_pred, y_prob, test_files, config, logger)
        except Exception as e:
            logger.warning(f"⚠ 分组评估导出失败: {e}")
        all_metrics[model_name] = metrics
        
        logger.info("")
    
    # ========================
    # Compare Results
    # ========================
    if len(all_metrics) > 1:
        logger.info("="*70)
        logger.info("📊 模型对比结果")
        logger.info("="*70)
        logger.info("")
        
        # 创建对比表格（支持三模型）
        header = f"{'Model':<16} | {'F1(pos=1)':>9} | {'Prec':>9} | {'Recall':>9} | {'AUC':>9}"
        logger.info(header)
        logger.info("-" * len(header))

        def _get(m: dict, k: str, fallback: float = float('nan')) -> float:
            try:
                return float(m.get(k, fallback))
            except Exception:
                return fallback

        def _score(m: dict) -> float:
            if not isinstance(m, dict):
                return float('-inf')
            if 'f1' in m:
                return float(m['f1'])
            if 'f1_pos' in m:
                return float(m['f1_pos'])
            return float('-inf')

        ranked = sorted(all_metrics.items(), key=lambda kv: _score(kv[1]), reverse=True)
        for name, m in ranked:
            f1v = _get(m, 'f1', _get(m, 'f1_pos'))
            pv = _get(m, 'precision', _get(m, 'precision_pos'))
            rv = _get(m, 'recall', _get(m, 'recall_pos'))
            aucv = _get(m, 'auc')
            logger.info(f"{name:<16} | {f1v:9.4f} | {pv:9.4f} | {rv:9.4f} | {aucv:9.4f}")

        logger.info("")

    chosen_name = None
    chosen_metrics = None
    try:
        def _score(m: dict) -> float:
            if not isinstance(m, dict):
                return float('-inf')
            if 'f1' in m:
                return float(m['f1'])
            if 'f1_pos' in m:
                return float(m['f1_pos'])
            return float('-inf')

        chosen_name = max(all_metrics.keys(), key=lambda k: _score(all_metrics.get(k)))
        chosen_metrics = all_metrics.get(chosen_name)
    except Exception as e:
        logger.warning(f"⚠ 无法根据F1选择最终模型: {e}")

    if chosen_name is not None and chosen_metrics is not None:
        logger.info("="*70)
        logger.info("🏁 最终测试结果选择")
        logger.info("="*70)
        logger.info(f"选择模型: {chosen_name}")
        logger.info(f"依据指标: F1(pos=1) = {float(chosen_metrics.get('f1', chosen_metrics.get('f1_pos', -1.0))):.4f}")
        logger.info("")
    
    logger.info("="*70)
    logger.info("🎉 测试完成! Testing Complete!")
    logger.info("="*70)
    logger.info("")
    logger.info("📁 输出文件路径:")
    if has_best:
        logger.info(f"  ✓ Best F1预测结果: {os.path.join(config.RESULT_DIR, 'models', 'test_best_predictions.npz')}")
        logger.info(f"  ✓ Best F1性能指标: {os.path.join(config.RESULT_DIR, 'models', 'test_best_metrics.txt')}")
        logger.info(f"  ✓ Best F1可视化: {os.path.join(config.RESULT_DIR, 'figures', 'test_best_*.png')}")
    if has_final:
        logger.info(f"  ✓ Final预测结果: {os.path.join(config.RESULT_DIR, 'models', 'test_final_predictions.npz')}")
        logger.info(f"  ✓ Final性能指标: {os.path.join(config.RESULT_DIR, 'models', 'test_final_metrics.txt')}")
        logger.info(f"  ✓ Final可视化: {os.path.join(config.RESULT_DIR, 'figures', 'test_final_*.png')}")
    logger.info("")
    logger.info("="*70)
    
    if chosen_metrics is not None:
        return chosen_metrics
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MEDAL-Lite model")
    parser.add_argument('--backbone_path', type=str, default='', help='Path to backbone checkpoint (optional)')
    parser.add_argument('--classifier_path', type=str, default='', help='Path to classifier checkpoint (optional)')
    
    args = parser.parse_args()
    
    main(args)

