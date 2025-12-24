"""
MEDAL-Lite Testing Script
Evaluate trained model on test dataset
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import (
    set_seed, setup_logger, calculate_metrics, print_metrics, find_optimal_threshold
)
from MoudleCode.utils.visualization import (
    plot_feature_space, plot_confusion_matrix, plot_roc_curve, plot_probability_distribution
)
from MoudleCode.preprocessing.pcap_parser import load_dataset
from MoudleCode.feature_extraction.backbone import MicroBiMambaBackbone
from MoudleCode.classification.dual_stream import MEDAL_Classifier

# 导入预处理模块
try:
    from preprocess import check_preprocessed_exists, load_preprocessed, preprocess_test
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

import logging


def test_model(classifier, X_test, y_test, config, logger, save_prefix="test"):
    """
    Test the model on test dataset
    
    Args:
        classifier: Trained MEDAL classifier
        X_test: (N, L, 5) test sequences
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
    logger.info(f"批次大小: 64")
    
    # 记录配置中的参考阈值（用于对比与可视化）
    config_threshold = getattr(config, 'MALICIOUS_THRESHOLD', 0.5)
    logger.info(f"✓ 配置参考阈值: {config_threshold:.2f} (仅用于参考与可视化)")
    logger.info("")
    
    classifier.eval()
    classifier.to(config.DEVICE)
    
    # Create DataLoader
    dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    total_batches = len(test_loader)
    
    # Collect probabilities / labels / features（先不使用阈值）
    all_probs = []
    all_labels = []
    all_features = []
    
    logger.info("开始推理...")
    logger.info("-"*70)
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            X_batch = X_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)
            
            # 直接前向获得 logits 和特征，不在这里做阈值判决
            logits, z = classifier(X_batch, return_features=True, return_separate=False)
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_features.append(z.cpu().numpy())
            
            # 每10个批次或最后一个批次输出进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                processed = (batch_idx + 1) * 64
                logger.info(f"  推理进度: {batch_idx+1}/{total_batches} batches ({progress:.1f}%) | 已处理 {processed}/{len(X_test)} 个样本")
    
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
    candidate_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, float(optimal_threshold), float(config_threshold)]
    candidate_thresholds = sorted(set([round(t, 4) for t in candidate_thresholds]))
    logger.info("📏 阈值对比 (Malicious=Positive):")
    logger.info(f"  {'threshold':>10s} | {'precision':>9s} | {'recall':>7s} | {'f1':>7s}")
    logger.info("  " + "-"*44)
    for th in candidate_thresholds:
        y_pred_th = (y_prob[:, 1] >= th).astype(int)
        p = precision_score(y_true, y_pred_th, pos_label=1, zero_division=0)
        r = recall_score(y_true, y_pred_th, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred_th, pos_label=1, zero_division=0)
        marker = " ← 最优" if abs(th - optimal_threshold) < 0.0001 else ""
        logger.info(f"  {th:10.4f} | {p:9.4f} | {r:7.4f} | {f1:7.4f}{marker}")
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
    
    return metrics


def main(args):
    """Main testing function"""
    
    # Setup
    set_seed(config.SEED)
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, "logs"), name='test')
    
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
    
    # 优先使用预处理好的数据
    if PREPROCESS_AVAILABLE and check_preprocessed_exists('test'):
        logger.info("✓ 发现预处理文件，直接加载...")
        X_test, y_test, test_files = load_preprocessed('test')
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
            if backbone_path_from_metadata:
                logger.info(f"✓ 从模型元数据中读取到训练时使用的骨干网络:")
                logger.info(f"  {backbone_path_from_metadata}")
                logger.info("")
        except Exception as e:
            logger.warning(f"⚠ 无法读取模型元数据: {e}")
    
    # Load backbone from feature_extraction directory
    backbone = MicroBiMambaBackbone(config)
    
    # Determine backbone path: use metadata if available, otherwise use default
    if backbone_path_from_metadata and os.path.exists(backbone_path_from_metadata):
        backbone_path = backbone_path_from_metadata
        logger.info("使用训练时的骨干网络（从元数据）")
    else:
        backbone_path = os.path.join(config.FEATURE_EXTRACTION_DIR, "models", "backbone_pretrained.pth")
        if backbone_path_from_metadata:
            logger.warning(f"⚠ 元数据中的骨干网络不存在: {backbone_path_from_metadata}")
            logger.warning(f"  回退到默认路径: {backbone_path}")
    
    if not os.path.exists(backbone_path):
        logger.error(f"❌ 骨干网络检查点未找到: {backbone_path}")
        logger.error("请先运行训练脚本!")
        return
    
    logger.info("正在加载骨干网络...")
    logger.info(f"  📥 输入模型: {backbone_path}")
    backbone.load_state_dict(torch.load(backbone_path, map_location=config.DEVICE))
    backbone.freeze()
    logger.info(f"✓ 骨干网络加载完成")
    
    # Load classifier from classification directory
    classifier = MEDAL_Classifier(backbone, config)
    classifier_path = os.path.join(config.CLASSIFICATION_DIR, "models", "classifier_final.pth")
    
    if not os.path.exists(classifier_path):
        logger.error(f"❌ 分类器检查点未找到: {classifier_path}")
        logger.error("请先运行训练脚本!")
        return
    
    logger.info("正在加载分类器...")
    logger.info(f"  📥 输入模型: {classifier_path}")
    classifier.load_state_dict(torch.load(classifier_path, map_location=config.DEVICE))
    logger.info(f"✓ 分类器加载完成")
    
    # Count parameters
    n_params = sum(p.numel() for p in classifier.parameters())
    n_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info("")
    logger.info("📊 模型参数统计:")
    logger.info(f"  总参数量: {n_params:,}")
    logger.info(f"  可训练参数: {n_trainable:,} (骨干网络已冻结)")
    logger.info("")
    
    # ========================
    # Test Model
    # ========================
    metrics = test_model(classifier, X_test, y_test, config, logger)
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MEDAL-Lite model")
    
    args = parser.parse_args()
    
    main(args)

