"""
分析难以分离的恶意流量样本
识别特征空间中与正常流量重叠的恶意样本
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from tqdm import tqdm

from MoudleCode.feature_extraction.backbone import build_backbone
from MoudleCode.utils.config import Config
from MoudleCode.utils.model_loader import load_backbone_safely
from MoudleCode.utils.visualization import _configure_matplotlib_chinese_fonts

# 配置matplotlib中文字体
_configure_matplotlib_chinese_fonts()


class HardSampleAnalyzer:
    """难以分离样本分析器"""
    
    def __init__(self, backbone_path, data_root, device='cuda'):
        """
        Args:
            backbone_path: 骨干网络权重路径
            data_root: 预处理数据目录
            device: 计算设备
        """
        self.device = device
        self.backbone_path = backbone_path
        
        # 加载配置
        config = Config()
        
        # 加载骨干网络
        print(f"[加载] 骨干网络: {backbone_path}")
        self.backbone = load_backbone_safely(
            backbone_path=backbone_path,
            config=config,
            device=device,
            logger=None
        )
        
        # 加载预处理数据
        print(f"[加载] 预处理数据: {data_root}")
        train_X = np.load(f"{data_root}/train_X.npy")
        train_y = np.load(f"{data_root}/train_y.npy")
        
        self.train_X = torch.from_numpy(train_X).float()
        self.train_y = torch.from_numpy(train_y).long()
        self.clean_labels = train_y
        
        print(f"✓ 数据加载完成: {len(self.train_X)} 个样本")
        print(f"  - 正常样本: {np.sum(train_y == 0)}")
        print(f"  - 恶意样本: {np.sum(train_y == 1)}")
        
    def extract_features(self):
        """提取特征"""
        print("\n[提取] 特征向量...")
        
        all_features = []
        batch_size = 64
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.train_X), batch_size)):
                batch_X = self.train_X[i:i+batch_size].to(self.device)
                features = self.backbone(batch_X, return_sequence=False)
                all_features.append(features.cpu().numpy())
        
        self.features = np.vstack(all_features)
        print(f"✓ 特征提取完成: {self.features.shape}")
        
        return self.features
    
    def identify_hard_samples(self, k=20, threshold=0.5):
        """
        识别难以分离的恶意样本
        
        Args:
            k: KNN的K值
            threshold: 判断为难以分离的阈值（KNN预测为正常的比例）
        
        Returns:
            hard_samples: 难以分离的样本索引
        """
        print(f"\n[分析] 识别难以分离的恶意样本 (K={k}, threshold={threshold})...")
        
        # 只分析恶意样本（标签为1）
        malicious_mask = self.clean_labels == 1
        malicious_indices = np.where(malicious_mask)[0]
        malicious_features = self.features[malicious_indices]
        
        # 使用所有样本训练KNN
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(self.features, self.clean_labels)
        
        # 对恶意样本进行预测
        malicious_predictions = knn.predict(malicious_features)
        malicious_proba = knn.predict_proba(malicious_features)
        
        # 计算每个恶意样本的K近邻中正常样本的比例
        distances, indices = knn.kneighbors(malicious_features)
        neighbor_labels = self.clean_labels[indices]
        benign_ratio = np.mean(neighbor_labels == 0, axis=1)  # 正常样本比例
        
        # 识别难以分离的样本（被预测为正常，或邻居中正常样本比例高）
        hard_mask = (malicious_predictions == 0) | (benign_ratio >= threshold)
        hard_indices = malicious_indices[hard_mask]
        
        print(f"  - 总恶意样本数: {len(malicious_indices)}")
        print(f"  - 难以分离的恶意样本数: {len(hard_indices)} ({len(hard_indices)/len(malicious_indices)*100:.1f}%)")
        
        # 统计信息
        stats = {
            'total_malicious': int(len(malicious_indices)),
            'hard_samples_count': int(len(hard_indices)),
            'hard_samples_ratio': float(len(hard_indices) / len(malicious_indices)),
            'predicted_as_benign': int(np.sum(malicious_predictions == 0)),
            'high_benign_neighbor_ratio': int(np.sum(benign_ratio >= threshold))
        }
        
        return hard_indices, benign_ratio[hard_mask], stats
    
    def analyze_hard_samples_features(self, hard_indices):
        """分析难以分离样本的原始特征"""
        print("\n[分析] 难以分离样本的原始特征统计...")
        
        # 提取难以分离样本的原始特征
        hard_X = self.train_X[hard_indices].numpy()
        
        # 提取所有恶意样本的原始特征（作为对比）
        all_malicious_mask = self.clean_labels == 1
        all_malicious_X = self.train_X[all_malicious_mask].numpy()
        
        # 提取正常样本的原始特征（作为对比）
        benign_mask = self.clean_labels == 0
        benign_X = self.train_X[benign_mask].numpy()
        
        # 特征名称
        feature_names = ['Length', 'Direction', 'BurstSize', 'LogIAT', 'ValidMask']
        
        stats = {}
        
        for i, feat_name in enumerate(feature_names):
            hard_feat = hard_X[:, :, i].flatten()  # 展平序列
            all_malicious_feat = all_malicious_X[:, :, i].flatten()
            benign_feat = benign_X[:, :, i].flatten()
            
            stats[feat_name] = {
                'hard_samples': {
                    'mean': float(np.mean(hard_feat)),
                    'std': float(np.std(hard_feat)),
                    'min': float(np.min(hard_feat)),
                    'max': float(np.max(hard_feat)),
                    'median': float(np.median(hard_feat))
                },
                'all_malicious': {
                    'mean': float(np.mean(all_malicious_feat)),
                    'std': float(np.std(all_malicious_feat)),
                    'median': float(np.median(all_malicious_feat))
                },
                'benign': {
                    'mean': float(np.mean(benign_feat)),
                    'std': float(np.std(benign_feat)),
                    'median': float(np.median(benign_feat))
                }
            }
            
            print(f"\n  {feat_name}:")
            print(f"    难以分离样本 - 均值: {stats[feat_name]['hard_samples']['mean']:.4f}, "
                  f"中位数: {stats[feat_name]['hard_samples']['median']:.4f}")
            print(f"    所有恶意样本 - 均值: {stats[feat_name]['all_malicious']['mean']:.4f}, "
                  f"中位数: {stats[feat_name]['all_malicious']['median']:.4f}")
            print(f"    正常样本 - 均值: {stats[feat_name]['benign']['mean']:.4f}, "
                  f"中位数: {stats[feat_name]['benign']['median']:.4f}")
        
        return stats
    
    def visualize_hard_samples(self, hard_indices, save_dir='./output/backbone_eval'):
        """可视化难以分离的样本在t-SNE图中的位置"""
        print("\n[可视化] 生成t-SNE图（标注难以分离的样本）...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # t-SNE降维
        try:
            n_samples = len(self.features)
            perplexity = min(30, max(5, n_samples // 4))
            
            print(f"  - 进行t-SNE降维 (perplexity={perplexity})...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                       max_iter=1000, verbose=0)
            features_2d = tsne.fit_transform(self.features)
        except Exception as e:
            print(f"[错误] t-SNE降维失败: {e}")
            return None
        
        # 创建标记数组
        is_hard = np.zeros(len(self.clean_labels), dtype=bool)
        is_hard[hard_indices] = True
        
        # 绘制
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左图：原始分类（正常/恶意）
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=self.clean_labels,
            cmap='coolwarm',
            alpha=0.6,
            s=20
        )
        ax1.set_title('特征空间分布（正常/恶意）', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dim 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dim 2', fontsize=12)
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Label (0=正常, 1=恶意)', rotation=270, labelpad=20)
        
        # 右图：标注难以分离的样本
        ax2 = axes[1]
        
        # 先绘制所有样本（灰色）
        ax2.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c='lightgray',
            alpha=0.3,
            s=15,
            label='其他样本'
        )
        
        # 绘制难以分离的恶意样本（红色，大点）
        hard_2d = features_2d[is_hard]
        ax2.scatter(
            hard_2d[:, 0], hard_2d[:, 1],
            c='red',
            alpha=0.8,
            s=100,
            marker='x',
            linewidths=2,
            label=f'难以分离的恶意样本 ({len(hard_indices)}个)'
        )
        
        # 绘制正常样本（蓝色，小点）
        benign_mask = self.clean_labels == 0
        benign_2d = features_2d[benign_mask]
        ax2.scatter(
            benign_2d[:, 0], benign_2d[:, 1],
            c='blue',
            alpha=0.4,
            s=20,
            label='正常样本'
        )
        
        ax2.set_title('难以分离的恶意样本位置', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Dim 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dim 2', fontsize=12)
        ax2.legend(loc='best')
        
        plt.tight_layout()
        save_path = save_dir / 'hard_samples_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化图保存至: {save_path}")
        plt.close()
        
        return features_2d
    
    def generate_report(self, hard_indices, benign_ratios, feature_stats, 
                       hard_samples_stats, save_dir='./output/backbone_eval'):
        """生成分析报告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'backbone_path': str(self.backbone_path),
            'total_samples': int(len(self.clean_labels)),
            'total_malicious': int(hard_samples_stats['total_malicious']),
            'hard_samples': {
                'count': int(hard_samples_stats['hard_samples_count']),
                'ratio': float(hard_samples_stats['hard_samples_ratio']),
                'indices': hard_indices.tolist(),
                'benign_neighbor_ratios': benign_ratios.tolist()
            },
            'prediction_stats': {
                'predicted_as_benign': int(hard_samples_stats['predicted_as_benign']),
                'high_benign_neighbor_ratio': int(hard_samples_stats['high_benign_neighbor_ratio'])
            },
            'feature_statistics': feature_stats
        }
        
        # 保存JSON
        json_path = save_dir / 'hard_samples_analysis.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可读报告
        log_path = save_dir / 'hard_samples_analysis.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("难以分离的恶意样本分析报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"骨干网络路径: {self.backbone_path}\n")
            f.write(f"分析时间: {Path(save_dir).name}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("统计摘要\n")
            f.write("-"*80 + "\n")
            f.write(f"总样本数: {len(self.clean_labels)}\n")
            f.write(f"恶意样本总数: {hard_samples_stats['total_malicious']}\n")
            f.write(f"难以分离的恶意样本数: {hard_samples_stats['hard_samples_count']}\n")
            f.write(f"难以分离比例: {hard_samples_stats['hard_samples_ratio']*100:.2f}%\n\n")
            
            f.write(f"预测统计:\n")
            f.write(f"  - 被KNN预测为正常: {hard_samples_stats['predicted_as_benign']} 个\n")
            f.write(f"  - K近邻中正常样本比例≥50%: {hard_samples_stats['high_benign_neighbor_ratio']} 个\n\n")
            
            f.write("-"*80 + "\n")
            f.write("难以分离样本的原始特征分析\n")
            f.write("-"*80 + "\n")
            
            for feat_name, stats in feature_stats.items():
                f.write(f"\n{feat_name}:\n")
                f.write(f"  难以分离样本:\n")
                f.write(f"    均值: {stats['hard_samples']['mean']:.4f}\n")
                f.write(f"    中位数: {stats['hard_samples']['median']:.4f}\n")
                f.write(f"    标准差: {stats['hard_samples']['std']:.4f}\n")
                f.write(f"  所有恶意样本:\n")
                f.write(f"    均值: {stats['all_malicious']['mean']:.4f}\n")
                f.write(f"    中位数: {stats['all_malicious']['median']:.4f}\n")
                f.write(f"  正常样本:\n")
                f.write(f"    均值: {stats['benign']['mean']:.4f}\n")
                f.write(f"    中位数: {stats['benign']['median']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("难以分离样本索引（前50个）\n")
            f.write("="*80 + "\n")
            for i, idx in enumerate(hard_indices[:50]):
                f.write(f"{idx:4d} ")
                if (i + 1) % 10 == 0:
                    f.write("\n")
            if len(hard_indices) > 50:
                f.write(f"\n... (共{len(hard_indices)}个样本)\n")
            f.write("\n")
        
        print(f"\n✓ 分析报告已保存:")
        print(f"  - JSON格式: {json_path}")
        print(f"  - 日志格式: {log_path}")
        
        return report
    
    def run_analysis(self, k=20, threshold=0.5, save_dir='./output/backbone_eval'):
        """运行完整分析"""
        print("\n" + "="*70)
        print("难以分离的恶意样本分析 - 开始")
        print("="*70)
        
        # 1. 提取特征
        self.extract_features()
        
        # 2. 识别难以分离的样本
        hard_indices, benign_ratios, stats = self.identify_hard_samples(k=k, threshold=threshold)
        
        if len(hard_indices) == 0:
            print("\n✓ 未发现难以分离的恶意样本，特征空间分离良好！")
            return None
        
        # 3. 分析原始特征
        feature_stats = self.analyze_hard_samples_features(hard_indices)
        
        # 4. 可视化
        features_2d = self.visualize_hard_samples(hard_indices, save_dir)
        
        # 5. 生成报告
        report = self.generate_report(
            hard_indices, benign_ratios, feature_stats, stats, save_dir
        )
        
        print("\n" + "="*70)
        print("分析完成！")
        print("="*70)
        
        return report


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析难以分离的恶意流量样本')
    parser.add_argument('--backbone', type=str, required=True,
                        help='骨干网络权重路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='预处理数据目录')
    parser.add_argument('--output', type=str, default='./output/backbone_eval',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--k', type=int, default=20,
                        help='KNN的K值')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='判断为难以分离的阈值（K近邻中正常样本比例）')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = HardSampleAnalyzer(
        backbone_path=args.backbone,
        data_root=args.data_root,
        device=args.device
    )
    
    # 运行分析
    report = analyzer.run_analysis(
        k=args.k,
        threshold=args.threshold,
        save_dir=args.output
    )
    
    if report:
        print(f"\n发现 {report['hard_samples']['count']} 个难以分离的恶意样本")
        print(f"比例: {report['hard_samples']['ratio']*100:.2f}%")


if __name__ == '__main__':
    main()
