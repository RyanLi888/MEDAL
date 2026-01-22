"""
CL+AUM 双重校验标签矫正 - 示例脚本
====================================

演示如何使用 CL+AUM 双重校验机制进行标签矫正。

使用方法:
    python scripts/run_cl_aum_correction.py --config configs/your_config.yaml

核心流程:
1. 加载预训练的 SimMTM 骨干网络（无监督训练）
2. 提取特征
3. 计算 CL 置信度（静态特征度量）
4. 训练探针并计算 AUM 分数（动态训练度量）
5. 双重校验决策（Keep/Flip/Drop）
6. 评估矫正效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import logging
from pathlib import Path

from MoudleCode.label_correction import HybridCourt
from MoudleCode.utils.config import config
from MoudleCode.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CL+AUM 双重校验标签矫正')
    
    # 数据参数
    parser.add_argument('--features', type=str, required=True,
                        help='特征文件路径 (.npy)')
    parser.add_argument('--noisy_labels', type=str, required=True,
                        help='噪声标签文件路径 (.npy)')
    parser.add_argument('--true_labels', type=str, default=None,
                        help='真实标签文件路径 (.npy, 可选)')
    
    # CL 参数
    parser.add_argument('--cl_threshold', type=float, default=0.7,
                        help='CL 置信度阈值')
    parser.add_argument('--cl_projection_dim', type=int, default=128,
                        help='CL 投影头维度')
    
    # AUM 参数
    parser.add_argument('--aum_threshold', type=float, default=0.0,
                        help='AUM 分数阈值')
    parser.add_argument('--aum_epochs', type=int, default=30,
                        help='AUM 训练轮数')
    parser.add_argument('--aum_batch_size', type=int, default=128,
                        help='AUM 批次大小')
    parser.add_argument('--aum_lr', type=float, default=0.01,
                        help='AUM 学习率')
    
    # KNN 参数
    parser.add_argument('--knn_k', type=int, default=10,
                        help='KNN 邻居数')
    parser.add_argument('--knn_purity_threshold', type=float, default=0.8,
                        help='KNN 纯度阈值')
    
    # 决策参数
    parser.add_argument('--use_drop', action='store_true',
                        help='是否使用 Drop 动作')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output/cl_aum_correction',
                        help='输出目录')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备')
    
    return parser.parse_args()


def load_data(args):
    """加载数据"""
    logger.info("="*70)
    logger.info("加载数据")
    logger.info("="*70)
    
    # 加载特征
    features = np.load(args.features)
    logger.info(f"  ✓ 特征: {features.shape}")
    
    # 加载噪声标签
    noisy_labels = np.load(args.noisy_labels)
    logger.info(f"  ✓ 噪声标签: {noisy_labels.shape}")
    
    # 加载真实标签（可选）
    true_labels = None
    if args.true_labels is not None:
        true_labels = np.load(args.true_labels)
        logger.info(f"  ✓ 真实标签: {true_labels.shape}")
        
        # 计算噪声率
        noise_rate = (noisy_labels != true_labels).mean()
        logger.info(f"  📊 噪声率: {noise_rate:.2%}")
    
    return features, noisy_labels, true_labels


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = output_dir / 'cl_aum_correction.log'
    setup_logger(log_file=str(log_file))
    
    logger.info("="*70)
    logger.info("CL+AUM 双重校验标签矫正")
    logger.info("="*70)
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  设备: {args.device}")
    
    # 加载数据
    features, noisy_labels, true_labels = load_data(args)
    
    # 初始化 HybridCourt
    logger.info("\n" + "="*70)
    logger.info("初始化 HybridCourt")
    logger.info("="*70)
    
    num_classes = len(np.unique(noisy_labels))

    config.NUM_CLASSES = int(num_classes)
    config.FEATURE_DIM = int(features.shape[1])
    config.KNN_NEIGHBORS = int(args.knn_k)
    config.DEVICE = args.device

    config.STAGE2_CL_THRESHOLD = float(args.cl_threshold)
    config.STAGE2_AUM_THRESHOLD = float(args.aum_threshold)
    config.AUM_EPOCHS = int(args.aum_epochs)
    config.AUM_BATCH_SIZE = int(args.aum_batch_size)
    config.AUM_LR = float(args.aum_lr)
    config.STAGE2_KNN_PURITY_THRESHOLD = float(args.knn_purity_threshold)
    config.STAGE2_USE_DROP = bool(args.use_drop)

    hybrid_court = HybridCourt(config)
    
    logger.info(f"  ✓ HybridCourt 初始化完成")
    logger.info(f"    类别数: {num_classes}")
    logger.info(f"    特征维度: {features.shape[1]}")
    logger.info(f"    KNN 邻居数: {args.knn_k}")
    
    # 执行 CL+AUM 双重校验
    logger.info("\n" + "="*70)
    logger.info("执行 CL+AUM 双重校验")
    logger.info("="*70)

    clean_labels, action_mask, confidence, correction_weight, aum_scores, neighbor_consistency, pred_probs = \
        hybrid_court.correct_labels(
            features=features,
            noisy_labels=noisy_labels,
            device=args.device,
            y_true=true_labels,
        )
    
    # 保存结果
    logger.info("\n" + "="*70)
    logger.info("保存结果")
    logger.info("="*70)
    
    np.save(output_dir / 'clean_labels.npy', clean_labels)
    np.save(output_dir / 'action_mask.npy', action_mask)
    np.save(output_dir / 'confidence.npy', confidence)
    np.save(output_dir / 'correction_weight.npy', correction_weight)
    np.save(output_dir / 'aum_scores.npy', aum_scores)
    np.save(output_dir / 'neighbor_consistency.npy', neighbor_consistency)
    np.save(output_dir / 'pred_probs.npy', pred_probs)
    
    logger.info(f"  ✓ 结果已保存到: {output_dir}")
    
    # 生成详细报告
    logger.info("\n" + "="*70)
    logger.info("生成详细报告")
    logger.info("="*70)
    
    report_file = output_dir / 'correction_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CL+AUM 双重校验标签矫正 - 详细报告\n")
        f.write("="*70 + "\n\n")
        
        # 参数配置
        f.write("参数配置:\n")
        f.write(f"  CL 阈值: {args.cl_threshold}\n")
        f.write(f"  AUM 阈值: {args.aum_threshold}\n")
        f.write(f"  AUM 训练轮数: {args.aum_epochs}\n")
        f.write(f"  KNN 邻居数: {args.knn_k}\n")
        f.write(f"  KNN 纯度阈值: {args.knn_purity_threshold}\n")
        f.write(f"  使用 Drop: {args.use_drop}\n\n")
        
        # 动作统计
        n_keep = (action_mask == 0).sum()
        n_flip = (action_mask == 1).sum()
        n_drop = (action_mask == 2).sum()
        n_total = len(action_mask)
        
        f.write("动作统计:\n")
        f.write(f"  Keep: {n_keep:5d} ({100*n_keep/n_total:.1f}%)\n")
        f.write(f"  Flip: {n_flip:5d} ({100*n_flip/n_total:.1f}%)\n")
        if args.use_drop:
            f.write(f"  Drop: {n_drop:5d} ({100*n_drop/n_total:.1f}%)\n")
        f.write(f"  Total: {n_total}\n\n")
        
        # 如果有真实标签，计算详细指标
        if true_labels is not None:
            is_noise = (noisy_labels != true_labels)
            
            # 整体纯度
            correct = (clean_labels == true_labels).sum()
            purity = 100.0 * correct / n_total
            
            original_correct = (noisy_labels == true_labels).sum()
            original_purity = 100.0 * original_correct / n_total
            
            f.write("纯度指标:\n")
            f.write(f"  原始纯度: {original_purity:.2f}%\n")
            f.write(f"  矫正纯度: {purity:.2f}%\n")
            f.write(f"  提升: {purity - original_purity:+.2f}%\n\n")
            
            # 各动作纯度
            f.write("各动作纯度:\n")
            for action_name, action_value in [('Keep', 0), ('Flip', 1), ('Drop', 2)]:
                mask = (action_mask == action_value)
                if mask.sum() > 0:
                    action_correct = ((clean_labels == true_labels) & mask).sum()
                    action_purity = 100.0 * action_correct / mask.sum()
                    f.write(f"  {action_name}: {action_purity:.2f}% ({action_correct}/{mask.sum()})\n")
            f.write("\n")
            
            # Flip 详细分析
            if n_flip > 0:
                flip_mask = (action_mask == 1)
                flip_correct = ((clean_labels == true_labels) & flip_mask).sum()
                flip_wrong = n_flip - flip_correct
                
                # 真阳性：正确识别并翻转噪声
                true_positive = ((clean_labels == true_labels) & flip_mask & is_noise).sum()
                # 假阳性：错误翻转干净样本
                false_positive = flip_wrong
                
                f.write("Flip 详细分析:\n")
                f.write(f"  翻转总数: {n_flip}\n")
                f.write(f"  翻转正确: {flip_correct} ({100*flip_correct/n_flip:.1f}%)\n")
                f.write(f"  翻转错误: {flip_wrong} ({100*flip_wrong/n_flip:.1f}%)\n")
                f.write(f"  真阳性 (正确识别噪声): {true_positive}\n")
                f.write(f"  假阳性 (误杀干净样本): {false_positive}\n\n")
            
            # AUM 分析
            clean_aum = aum_scores[~is_noise]
            noise_aum = aum_scores[is_noise]
            
            f.write("AUM 分数分析:\n")
            f.write(f"  干净样本 AUM: {clean_aum.mean():.4f} ± {clean_aum.std():.4f}\n")
            f.write(f"  噪声样本 AUM: {noise_aum.mean():.4f} ± {noise_aum.std():.4f}\n")
            f.write(f"  AUM 与噪声相关性: {np.corrcoef(aum_scores, is_noise.astype(int))[0, 1]:.4f}\n\n")
    
    logger.info(f"  ✓ 报告已保存到: {report_file}")
    
    logger.info("\n" + "="*70)
    logger.info("CL+AUM 双重校验完成！")
    logger.info("="*70)


if __name__ == '__main__':
    main()
