"""
骨干网络"上帝视角"评估脚本
Stage1 训练结束后，用真值标签评估特征空间质量
决定是否需要 Stage 2.5 (SupCon 微调)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

from MoudleCode.feature_extraction.backbone import MicroBiMambaBackbone, build_backbone
from MoudleCode.utils.config import Config
from MoudleCode.utils.visualization import _configure_matplotlib_chinese_fonts

# 配置matplotlib中文字体
_configure_matplotlib_chinese_fonts()


class BackboneEvaluator:
    """骨干网络特征空间评估器"""
    
    def __init__(self, backbone_path, data_root, device='cuda'):
        """
        Args:
            backbone_path: Stage1 训练好的骨干网络权重路径
            data_root: 预处理数据目录（包含 train_X.npy 和 train_y.npy）
            device: 计算设备
        """
        self.device = device
        self.backbone_path = backbone_path
        
        # 加载配置
        config = Config()
        
        # 加载骨干网络
        print(f"[加载] 骨干网络: {backbone_path}")
        self.backbone = build_backbone(config)

        load_location = 'cpu' if str(device) != 'cpu' else 'cpu'
        try:
            checkpoint = torch.load(backbone_path, map_location=load_location, weights_only=True)
        except TypeError:
            checkpoint = torch.load(backbone_path, map_location=load_location)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.to(device)
        self.backbone.eval()
        
        # 加载预处理数据
        print(f"[加载] 预处理数据: {data_root}")
        train_X = np.load(f"{data_root}/train_X.npy")
        train_y = np.load(f"{data_root}/train_y.npy")
        
        # 转换为 PyTorch 张量
        self.train_X = torch.from_numpy(train_X).float()
        self.train_y = torch.from_numpy(train_y).long()
        
        # train_y 就是真实标签
        self.clean_labels_array = train_y
        
        print(f"✓ 数据加载完成: {len(self.train_X)} 个样本, 特征维度 {self.train_X.shape}")
        print(f"  - 真实标签分布: {np.bincount(train_y)}")
        
    def extract_features(self):
        """提取所有样本的特征"""
        print("\n[提取] 特征向量...")
        
        all_features = []
        
        # 分批处理
        batch_size = 64
        with torch.no_grad():
            for i in tqdm(range(0, len(self.train_X), batch_size)):
                batch_X = self.train_X[i:i+batch_size].to(self.device)
                
                # 提取特征 (MicroBiMamba 输出是 [B, d_model])
                features = self.backbone(batch_X, return_sequence=False)
                all_features.append(features.cpu().numpy())
        
        # 合并所有批次
        self.features = np.vstack(all_features)  # [N, d_model]
        
        print(f"✓ 提取完成: {self.features.shape[0]} 个样本, 特征维度 {self.features.shape[1]}")
        print(f"  - 真实标签分布: {np.bincount(self.clean_labels_array)}")
        
        return self.features, self.clean_labels_array
    
    def visualize_tsne(self, save_dir='./output/backbone_eval'):
        """t-SNE 可视化：用真实标签着色"""
        print("\n[可视化] t-SNE 降维...")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(self.features)
        
        # 绘制特征空间分布
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        scatter = ax.scatter(
            features_2d[:, 0], 
            features_2d[:, 1],
            c=self.clean_labels_array,
            cmap='coolwarm',
            alpha=0.6,
            s=20
        )
        ax.set_title('骨干网络特征空间 (真实标签)', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dim 1', fontsize=12)
        ax.set_ylabel('t-SNE Dim 2', fontsize=12)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Label (0=benign, 1=malicious)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        save_path = save_dir / 'tsne_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ t-SNE 图保存至: {save_path}")
        plt.close()
        
    def knn_purity_test(self, k_values=[5, 10, 20, 50]):
        """KNN 纯净度测试：评估特征空间的可分性"""
        print("\n[评估] KNN 纯净度测试...")
        
        results = {}
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
            knn.fit(self.features, self.clean_labels_array)
            
            # 预测
            pred_labels = knn.predict(self.features)
            acc = accuracy_score(self.clean_labels_array, pred_labels)
            
            # 混淆矩阵
            cm = confusion_matrix(self.clean_labels_array, pred_labels)
            
            results[k] = {
                'accuracy': acc,
                'confusion_matrix': cm.tolist()
            }
            
            print(f"  K={k:2d}: 准确率 = {acc*100:.2f}%")
        
        return results

    def knn_split_test(self, k_values=[5, 10, 20, 50], test_size=0.3, random_state=42):
        X = np.asarray(self.features)
        y = np.asarray(self.clean_labels_array)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        results = {}
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
            knn.fit(X_tr, y_tr)
            pred = knn.predict(X_te)
            acc = accuracy_score(y_te, pred)
            cm = confusion_matrix(y_te, pred)
            results[k] = {
                'accuracy': acc,
                'confusion_matrix': cm.tolist(),
                'test_size': float(test_size),
                'random_state': int(random_state),
                'n_train': int(len(y_tr)),
                'n_test': int(len(y_te)),
            }
        return results
    
    def make_decision(self, knn_results):
        """根据 KNN 结果给出决策建议"""
        print("\n" + "="*70)
        print("【决策矩阵】骨干网络质量诊断")
        print("="*70)
        
        # 取 K=20 作为主要参考
        main_acc = knn_results[20]['accuracy']
        
        print(f"\n核心指标: KNN-20 准确率 (真值) = {main_acc*100:.2f}%\n")
        
        if main_acc >= 0.85:
            grade = "S级 (完美)"
            diagnosis = "骨干网络非常强健，完全具备了抗噪能力，已经提取出了本质特征。"
            action = "✓ 直接进入 Stage 3\n  无需任何微调。现在的特征 + 双流 MiniMLP 足以拿到很好的结果。"
            color = "\033[92m"  # 绿色
            need_supcon = False
            
        elif main_acc >= 0.70:
            grade = "A级 (及格)"
            diagnosis = "学到了大体特征，但受噪声影响，边界不够锐利。"
            action = "✓ 直接进入 Stage 3\n  依靠 MiniMLP 的非线性能力和正交约束去切分边界。\n  Dual-Stream 的 Sample Weights 会帮你处理边界难样本。"
            color = "\033[93m"  # 黄色
            need_supcon = False
            
        else:
            grade = "C级 (崩塌)"
            diagnosis = "骨干网络过拟合了噪声。它学的不是病灶，是噪声的分布。"
            action = "✗ 必须加入 Stage 2.5 (SupCon 微调)\n  Stage 3 救不回来了。必须利用筛选出的干净样本，\n  通过有监督对比学习重塑特征空间，把两类强行拉开。"
            color = "\033[91m"  # 红色
            need_supcon = True
        
        print(f"{color}等级: {grade}\033[0m")
        print(f"诊断: {diagnosis}")
        print(f"\n推荐策略:\n{action}")
        print("\n" + "="*70)
        
        return {
            'grade': grade,
            'main_accuracy': main_acc,
            'need_supcon': need_supcon,
            'diagnosis': diagnosis,
            'action': action
        }
    
    def generate_report(self, knn_results, decision, save_dir='./output/backbone_eval', knn_split_results=None):
        """生成完整的评估报告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'backbone_path': str(self.backbone_path),
            'total_samples': int(self.features.shape[0]),
            'feature_dim': int(self.features.shape[1]),
            'label_distribution': {
                int(k): int(v) for k, v in enumerate(np.bincount(self.clean_labels_array))
            },
            'knn_results': {
                str(k): {
                    'accuracy': float(v['accuracy']),
                    'confusion_matrix': v['confusion_matrix']
                } for k, v in knn_results.items()
            },
            'knn_split_results': {
                str(k): {
                    'accuracy': float(v['accuracy']),
                    'confusion_matrix': v['confusion_matrix'],
                    'test_size': float(v.get('test_size', 0.0)),
                    'random_state': int(v.get('random_state', 0)),
                    'n_train': int(v.get('n_train', 0)),
                    'n_test': int(v.get('n_test', 0)),
                } for k, v in (knn_split_results or {}).items()
            },
            'decision': {
                'grade': decision['grade'],
                'main_accuracy': float(decision['main_accuracy']),
                'need_supcon': decision['need_supcon'],
                'diagnosis': decision['diagnosis'],
                'action': decision['action']
            }
        }
        
        # 保存 JSON
        report_path = save_dir / 'evaluation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可读的日志文件
        log_path = save_dir / 'evaluation_report.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("骨干网络「上帝视角」评估报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"骨干网络路径: {self.backbone_path}\n")
            f.write(f"评估时间: {Path(save_dir).name}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("数据集信息\n")
            f.write("-"*80 + "\n")
            f.write(f"总样本数: {self.features.shape[0]}\n")
            f.write(f"特征维度: {self.features.shape[1]}\n")
            f.write(f"标签分布:\n")
            for label, count in enumerate(np.bincount(self.clean_labels_array)):
                label_name = "benign" if label == 0 else "malicious"
                f.write(f"  - {label_name} ({label}): {count} 个样本\n")
            f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("KNN 纯净度测试结果\n")
            f.write("-"*80 + "\n")
            for k, result in knn_results.items():
                acc = result['accuracy']
                cm = np.array(result['confusion_matrix'])
                f.write(f"\nK = {k}:\n")
                f.write(f"  准确率: {acc*100:.2f}%\n")
                f.write(f"  混淆矩阵:\n")
                f.write(f"              预测benign  预测malicious\n")
                f.write(f"  真实benign      {cm[0,0]:4d}        {cm[0,1]:4d}\n")
                f.write(f"  真实malicious   {cm[1,0]:4d}        {cm[1,1]:4d}\n")
            f.write("\n")

            if knn_split_results:
                f.write("-"*80 + "\n")
                f.write("KNN Split 测试结果 (train/test split)\n")
                f.write("-"*80 + "\n")
                for k, result in knn_split_results.items():
                    acc = result['accuracy']
                    cm = np.array(result['confusion_matrix'])
                    f.write(f"\nK = {k}:\n")
                    f.write(f"  测试集准确率: {acc*100:.2f}%\n")
                    f.write(f"  split: test_size={result.get('test_size', 0.0)}, random_state={result.get('random_state', 0)}\n")
                    f.write(f"  n_train={result.get('n_train', 0)}, n_test={result.get('n_test', 0)}\n")
                    f.write(f"  混淆矩阵:\n")
                    f.write(f"              预测benign  预测malicious\n")
                    f.write(f"  真实benign      {cm[0,0]:4d}        {cm[0,1]:4d}\n")
                    f.write(f"  真实malicious   {cm[1,0]:4d}        {cm[1,1]:4d}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("决策建议\n")
            f.write("="*80 + "\n\n")
            f.write(f"等级: {decision['grade']}\n")
            f.write(f"核心指标 (KNN-20): {decision['main_accuracy']*100:.2f}%\n\n")
            f.write(f"诊断:\n{decision['diagnosis']}\n\n")
            f.write(f"推荐策略:\n{decision['action']}\n\n")
            
            if decision['need_supcon']:
                f.write("⚠️  需要执行 Stage 2.5 (SupCon 微调)\n")
            else:
                f.write("✓ 可以直接进入 Stage 3\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\n✓ 评估报告已保存:")
        print(f"  - JSON格式: {report_path}")
        print(f"  - 日志格式: {log_path}")
        
        return report
    
    def run_full_evaluation(self, save_dir='./output/backbone_eval', knn_test_size=0.3, knn_random_state=42):
        """运行完整的评估流程"""
        print("\n" + "="*70)
        print("骨干网络「上帝视角」评估 - 开始")
        print("="*70)
        
        # 1. 提取特征
        self.extract_features()
        
        # 2. t-SNE 可视化
        self.visualize_tsne(save_dir)
        
        # 3. KNN 纯净度测试（训练集自测，容易虚高，用于观察簇结构）
        knn_results = self.knn_purity_test()

        # 3b. KNN split 测试（更接近泛化能力）
        print("\n[评估] KNN Split 测试 (train/test split)...")
        knn_split_results = self.knn_split_test(test_size=knn_test_size, random_state=knn_random_state)
        for k in sorted(knn_split_results.keys()):
            acc = knn_split_results[k]['accuracy']
            print(f"  K={k:2d}: 测试集准确率 = {acc*100:.2f}%")
        
        # 4. 决策（仍然沿用 KNN-20 自测作为原决策矩阵的核心指标，split 用于辅助判断）
        decision = self.make_decision(knn_results)
        
        # 5. 生成报告
        report = self.generate_report(knn_results, decision, save_dir, knn_split_results=knn_split_results)
        
        print("\n" + "="*70)
        print("评估完成！")
        print("="*70)
        
        return report


def main():
    """主函数：示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='骨干网络特征空间评估')
    parser.add_argument('--backbone', type=str, required=True,
                        help='Stage1 训练好的骨干网络权重路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='预处理数据目录（包含 train_X.npy 和 train_y.npy）')
    parser.add_argument('--output', type=str, default='./output/backbone_eval',
                        help='评估结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--knn_test_size', type=float, default=0.3,
                        help='KNN split 测试的测试集比例')
    parser.add_argument('--knn_random_state', type=int, default=42,
                        help='KNN split 测试的随机种子')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = BackboneEvaluator(
        backbone_path=args.backbone,
        data_root=args.data_root,
        device=args.device
    )
    
    # 运行完整评估
    report = evaluator.run_full_evaluation(save_dir=args.output, knn_test_size=args.knn_test_size, knn_random_state=args.knn_random_state)
    
    # 打印最终建议
    if report['decision']['need_supcon']:
        print("\n⚠️  建议执行 Stage 2.5: 运行 supcon_finetune.py")
    else:
        print("\n✓ 可以直接进入 Stage 3: 运行 stage3_dual_stream.py")


if __name__ == '__main__':
    main()
