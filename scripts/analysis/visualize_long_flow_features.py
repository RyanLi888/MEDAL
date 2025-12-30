#!/usr/bin/env python3
"""
长流特征可视化脚本

用于生成论文图表，展示 BurstSize 和 CumulativeLen 特征的区分能力

使用方法：
    python scripts/analysis/visualize_long_flow_features.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from MoudleCode.utils.config import config
from MoudleCode.utils.logger import setup_logger

logger = setup_logger(__name__)

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_preprocessed_data():
    """加载预处理后的数据"""
    preprocessed_dir = os.path.join(config.OUTPUT_ROOT, "preprocessed")
    
    logger.info("加载预处理数据...")
    X_train = np.load(os.path.join(preprocessed_dir, "train_X.npy"))
    y_train = np.load(os.path.join(preprocessed_dir, "train_y.npy"))
    
    logger.info(f"训练集形状: {X_train.shape}")
    logger.info(f"良性样本: {(y_train==0).sum()}, 恶意样本: {(y_train==1).sum()}")
    
    return X_train, y_train


def extract_flow_statistics(X, y):
    """
    提取流级别的统计特征
    
    Args:
        X: (N, L, D) - 特征序列
        y: (N,) - 标签
        
    Returns:
        stats: dict - 统计特征
    """
    N, L, D = X.shape
    
    # 特征索引
    LENGTH_IDX = config.LENGTH_INDEX
    IAT_IDX = config.IAT_INDEX
    DIRECTION_IDX = config.DIRECTION_INDEX
    BURST_IDX = config.BURST_SIZE_INDEX
    CUMULATIVE_IDX = config.CUMULATIVE_LEN_INDEX
    VALID_IDX = config.VALID_MASK_INDEX
    
    # 提取有效包（ValidMask > 0.5）
    valid_mask = X[:, :, VALID_IDX] > 0.5  # (N, L)
    
    stats = {
        'benign': {'burst_mean': [], 'length_mean': [], 'packet_count': [], 'cumulative_final': []},
        'malicious': {'burst_mean': [], 'length_mean': [], 'packet_count': [], 'cumulative_final': []}
    }
    
    for i in range(N):
        label = 'benign' if y[i] == 0 else 'malicious'
        valid_packets = valid_mask[i]
        n_valid = valid_packets.sum()
        
        if n_valid == 0:
            continue
        
        # 提取有效包的特征
        burst_values = X[i, valid_packets, BURST_IDX]
        length_values = X[i, valid_packets, LENGTH_IDX]
        cumulative_values = X[i, valid_packets, CUMULATIVE_IDX]
        
        # 统计特征
        stats[label]['burst_mean'].append(np.mean(burst_values))
        stats[label]['length_mean'].append(np.mean(length_values))
        stats[label]['packet_count'].append(n_valid)
        stats[label]['cumulative_final'].append(cumulative_values[-1] if len(cumulative_values) > 0 else 0.0)
    
    # 转换为 numpy 数组
    for label in ['benign', 'malicious']:
        for key in stats[label]:
            stats[label][key] = np.array(stats[label][key])
    
    return stats


def plot_burst_vs_length_scatter(stats, output_dir):
    """
    绘制 BurstSize vs PacketLength 散点图
    
    展示不同流量类型在特征空间中的分布
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点
    ax.scatter(stats['benign']['length_mean'], stats['benign']['burst_mean'],
               alpha=0.6, s=50, label='Benign', marker='o', edgecolors='k', linewidths=0.5)
    ax.scatter(stats['malicious']['length_mean'], stats['malicious']['burst_mean'],
               alpha=0.6, s=50, label='Malicious', marker='^', edgecolors='k', linewidths=0.5)
    
    ax.set_xlabel('Average Packet Length (normalized)', fontsize=14)
    ax.set_ylabel('Average Burst Size (log scale)', fontsize=14)
    ax.set_title('Burst Size vs Packet Length Distribution', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'burst_vs_length_scatter.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'burst_vs_length_scatter.pdf'), bbox_inches='tight')
    logger.info(f"✓ 保存散点图: burst_vs_length_scatter.png")
    plt.close()


def plot_cumulative_length_curves(X, y, output_dir, n_samples=100):
    """
    绘制 CumulativeLen 曲线
    
    展示良性和恶意流量的累积模式差异
    """
    CUMULATIVE_IDX = config.CUMULATIVE_LEN_INDEX
    VALID_IDX = config.VALID_MASK_INDEX
    
    # 随机采样
    benign_idx = np.where(y == 0)[0]
    malicious_idx = np.where(y == 1)[0]
    
    benign_sample = np.random.choice(benign_idx, min(n_samples, len(benign_idx)), replace=False)
    malicious_sample = np.random.choice(malicious_idx, min(n_samples, len(malicious_idx)), replace=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 良性流量
    ax = axes[0]
    for idx in benign_sample:
        valid_mask = X[idx, :, VALID_IDX] > 0.5
        cumulative = X[idx, valid_mask, CUMULATIVE_IDX]
        if len(cumulative) > 0:
            ax.plot(cumulative, alpha=0.3, color='blue', linewidth=0.5)
    
    # 平均曲线
    benign_cumulative_all = []
    for idx in benign_idx:
        valid_mask = X[idx, :, VALID_IDX] > 0.5
        cumulative = X[idx, valid_mask, CUMULATIVE_IDX]
        if len(cumulative) > 0:
            benign_cumulative_all.append(cumulative)
    
    # 对齐到相同长度（插值）
    max_len = max(len(c) for c in benign_cumulative_all)
    benign_aligned = []
    for c in benign_cumulative_all:
        if len(c) < max_len:
            # 线性插值
            x_old = np.linspace(0, 1, len(c))
            x_new = np.linspace(0, 1, max_len)
            c_interp = np.interp(x_new, x_old, c)
            benign_aligned.append(c_interp)
        else:
            benign_aligned.append(c[:max_len])
    
    benign_mean = np.mean(benign_aligned, axis=0)
    ax.plot(benign_mean, color='darkblue', linewidth=3, label='Mean')
    
    ax.set_xlabel('Packet Index', fontsize=12)
    ax.set_ylabel('Cumulative Length Ratio', fontsize=12)
    ax.set_title('Benign Traffic', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 恶意流量
    ax = axes[1]
    for idx in malicious_sample:
        valid_mask = X[idx, :, VALID_IDX] > 0.5
        cumulative = X[idx, valid_mask, CUMULATIVE_IDX]
        if len(cumulative) > 0:
            ax.plot(cumulative, alpha=0.3, color='red', linewidth=0.5)
    
    # 平均曲线
    malicious_cumulative_all = []
    for idx in malicious_idx:
        valid_mask = X[idx, :, VALID_IDX] > 0.5
        cumulative = X[idx, valid_mask, CUMULATIVE_IDX]
        if len(cumulative) > 0:
            malicious_cumulative_all.append(cumulative)
    
    max_len = max(len(c) for c in malicious_cumulative_all)
    malicious_aligned = []
    for c in malicious_cumulative_all:
        if len(c) < max_len:
            x_old = np.linspace(0, 1, len(c))
            x_new = np.linspace(0, 1, max_len)
            c_interp = np.interp(x_new, x_old, c)
            malicious_aligned.append(c_interp)
        else:
            malicious_aligned.append(c[:max_len])
    
    malicious_mean = np.mean(malicious_aligned, axis=0)
    ax.plot(malicious_mean, color='darkred', linewidth=3, label='Mean')
    
    ax.set_xlabel('Packet Index', fontsize=12)
    ax.set_ylabel('Cumulative Length Ratio', fontsize=12)
    ax.set_title('Malicious Traffic', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_length_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cumulative_length_curves.pdf'), bbox_inches='tight')
    logger.info(f"✓ 保存累积曲线: cumulative_length_curves.png")
    plt.close()


def plot_feature_distributions(stats, output_dir):
    """
    绘制特征分布的 CDF 图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Burst Size CDF
    ax = axes[0, 0]
    benign_burst = np.sort(stats['benign']['burst_mean'])
    malicious_burst = np.sort(stats['malicious']['burst_mean'])
    ax.plot(benign_burst, np.linspace(0, 1, len(benign_burst)), label='Benign', linewidth=2)
    ax.plot(malicious_burst, np.linspace(0, 1, len(malicious_burst)), label='Malicious', linewidth=2)
    ax.set_xlabel('Average Burst Size (log scale)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Burst Size Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Packet Length CDF
    ax = axes[0, 1]
    benign_length = np.sort(stats['benign']['length_mean'])
    malicious_length = np.sort(stats['malicious']['length_mean'])
    ax.plot(benign_length, np.linspace(0, 1, len(benign_length)), label='Benign', linewidth=2)
    ax.plot(malicious_length, np.linspace(0, 1, len(malicious_length)), label='Malicious', linewidth=2)
    ax.set_xlabel('Average Packet Length (normalized)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Packet Length Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Packet Count CDF
    ax = axes[1, 0]
    benign_count = np.sort(stats['benign']['packet_count'])
    malicious_count = np.sort(stats['malicious']['packet_count'])
    ax.plot(benign_count, np.linspace(0, 1, len(benign_count)), label='Benign', linewidth=2)
    ax.plot(malicious_count, np.linspace(0, 1, len(malicious_count)), label='Malicious', linewidth=2)
    ax.set_xlabel('Packet Count', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Flow Length Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative Final Value CDF
    ax = axes[1, 1]
    benign_cumulative = np.sort(stats['benign']['cumulative_final'])
    malicious_cumulative = np.sort(stats['malicious']['cumulative_final'])
    ax.plot(benign_cumulative, np.linspace(0, 1, len(benign_cumulative)), label='Benign', linewidth=2)
    ax.plot(malicious_cumulative, np.linspace(0, 1, len(malicious_cumulative)), label='Malicious', linewidth=2)
    ax.set_xlabel('Final Cumulative Length Ratio', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Flow Completeness Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Complete Flow')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions_cdf.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'feature_distributions_cdf.pdf'), bbox_inches='tight')
    logger.info(f"✓ 保存 CDF 图: feature_distributions_cdf.png")
    plt.close()


def print_statistics(stats):
    """打印统计信息"""
    logger.info("\n" + "="*70)
    logger.info("特征统计摘要")
    logger.info("="*70)
    
    for label in ['benign', 'malicious']:
        logger.info(f"\n{label.upper()} 流量:")
        logger.info(f"  样本数: {len(stats[label]['burst_mean'])}")
        logger.info(f"  平均 Burst Size: {np.mean(stats[label]['burst_mean']):.4f} ± {np.std(stats[label]['burst_mean']):.4f}")
        logger.info(f"  平均 Packet Length: {np.mean(stats[label]['length_mean']):.4f} ± {np.std(stats[label]['length_mean']):.4f}")
        logger.info(f"  平均 Packet Count: {np.mean(stats[label]['packet_count']):.1f} ± {np.std(stats[label]['packet_count']):.1f}")
        logger.info(f"  平均 Cumulative Final: {np.mean(stats[label]['cumulative_final']):.4f} ± {np.std(stats[label]['cumulative_final']):.4f}")
        
        # 长流比例（Cumulative Final < 1.0 表示被截断）
        long_flow_ratio = (stats[label]['cumulative_final'] < 0.99).sum() / len(stats[label]['cumulative_final'])
        logger.info(f"  长流比例 (>1024包): {long_flow_ratio*100:.1f}%")
    
    logger.info("\n" + "="*70)


def main():
    """主函数"""
    logger.info("="*70)
    logger.info("长流特征可视化")
    logger.info("="*70)
    
    # 创建输出目录
    output_dir = os.path.join(config.OUTPUT_ROOT, "feature_analysis", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    X_train, y_train = load_preprocessed_data()
    
    # 提取统计特征
    logger.info("\n提取流级别统计特征...")
    stats = extract_flow_statistics(X_train, y_train)
    
    # 打印统计信息
    print_statistics(stats)
    
    # 生成图表
    logger.info("\n生成可视化图表...")
    plot_burst_vs_length_scatter(stats, output_dir)
    plot_cumulative_length_curves(X_train, y_train, output_dir)
    plot_feature_distributions(stats, output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("✓ 可视化完成!")
    logger.info("="*70)
    logger.info(f"输出目录: {output_dir}")
    logger.info("生成的图表:")
    logger.info("  - burst_vs_length_scatter.png/pdf")
    logger.info("  - cumulative_length_curves.png/pdf")
    logger.info("  - feature_distributions_cdf.png/pdf")
    logger.info("="*70)


if __name__ == "__main__":
    main()
