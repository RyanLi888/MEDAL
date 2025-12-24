"""
测试脚本：验证 TabDDPM 生成质量诊断功能

运行此脚本可以快速验证增强数据的质量
"""
import sys
import os
# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from MoudleCode.utils.config import config
from MoudleCode.utils.visualization import plot_real_vs_synthetic_comparison
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_augmented_data():
    """加载增强数据"""
    augmented_data_path = os.path.join(config.DATA_AUGMENTATION_DIR, "models", "augmented_data.npz")
    
    if not os.path.exists(augmented_data_path):
        logger.error(f"❌ 找不到增强数据: {augmented_data_path}")
        logger.error("   请先运行 train.py 完成 Stage 2")
        return None
    
    logger.info(f"✓ 加载增强数据: {augmented_data_path}")
    data = np.load(augmented_data_path)
    
    X_augmented = data['X_augmented']
    y_augmented = data['y_augmented']
    is_original = data['is_original']
    n_original = int(data['n_original'])
    
    logger.info(f"  总样本数: {len(X_augmented)}")
    logger.info(f"  原始样本: {n_original}")
    logger.info(f"  合成样本: {len(X_augmented) - n_original}")
    
    return X_augmented, y_augmented, is_original, n_original


def evaluate_generation_quality(X_augmented, y_augmented, n_original):
    """评估生成质量"""
    logger.info("\n" + "="*70)
    logger.info("生成质量诊断")
    logger.info("="*70)
    
    # 分离原始数据和合成数据
    X_original = X_augmented[:n_original]
    X_synthetic = X_augmented[n_original:]
    y_original = y_augmented[:n_original]
    y_synthetic = y_augmented[n_original:]
    
    logger.info(f"原始数据: {len(X_original)} 个样本")
    logger.info(f"合成数据: {len(X_synthetic)} 个样本")
    logger.info("")
    
    # 1. Fidelity: 特征分布对比
    logger.info("1️⃣  Fidelity 检查: 特征分布对比")
    feature_names = ['Length', 'IAT', 'Window', 'Direction', 'Flags']
    logger.info(f"{'特征':<12} | {'真实均值':<12} | {'合成均值':<12} | {'差异%':<10} | {'真实标准差':<12} | {'合成标准差':<12}")
    logger.info("-"*85)
    
    fidelity_scores = []
    for i, name in enumerate(feature_names):
        real_data = X_original[:, :, i].flatten()
        syn_data = X_synthetic[:, :, i].flatten()
        
        real_mean = real_data.mean()
        syn_mean = syn_data.mean()
        real_std = real_data.std()
        syn_std = syn_data.std()
        
        diff_pct = abs(real_mean - syn_mean) / (abs(real_mean) + 1e-8) * 100
        fidelity_scores.append(diff_pct)
        
        quality_marker = "✓" if diff_pct < 10 else ("⚠" if diff_pct < 20 else "❌")
        
        logger.info(f"{name:<12} | {real_mean:>11.4f} | {syn_mean:>11.4f} | {diff_pct:>8.2f}% {quality_marker} | "
                   f"{real_std:>11.4f} | {syn_std:>11.4f}")
    
    avg_fidelity = np.mean(fidelity_scores)
    logger.info(f"\n  平均 Fidelity 差异: {avg_fidelity:.2f}%")
    logger.info("")
    
    # 2. Protocol Validity: 协议约束检查
    logger.info("2️⃣  Protocol Validity 检查: 物理约束验证")
    
    invalid_length = (X_synthetic[:, :, 0] < 0).sum()
    invalid_window = (X_synthetic[:, :, 4] < 0).sum()
    invalid_iat = (X_synthetic[:, :, 1] < 0).sum()
    invalid_direction = ((X_synthetic[:, :, 3] < 0) | (X_synthetic[:, :, 3] > 1)).sum()
    invalid_flags = ((X_synthetic[:, :, 4] < 0) | (X_synthetic[:, :, 4] > 255)).sum()
    
    total_values = X_synthetic.size
    invalid_total = invalid_length + invalid_window + invalid_iat + invalid_direction + invalid_flags
    validity_rate = (1 - invalid_total / total_values) * 100
    
    logger.info(f"  ❌ 负数包长 (Length < 0):     {invalid_length:>6} 个值")
    logger.info(f"  ❌ 负数窗口 (Window < 0):     {invalid_window:>6} 个值")
    logger.info(f"  ❌ 负数IAT (IAT < 0):         {invalid_iat:>6} 个值")
    logger.info(f"  ❌ 异常方向 (Direction ∉[0,1]): {invalid_direction:>6} 个值")
    logger.info(f"  ❌ 异常标志 (Flags ∉[0,255]):  {invalid_flags:>6} 个值")
    logger.info(f"  ✓ 总有效率: {validity_rate:.2f}%")
    logger.info("")
    
    # 3. Class-wise 分布检查
    logger.info("3️⃣  Class-wise 分布检查")
    logger.info(f"  原始数据 - 正常: {(y_original==0).sum()}, 恶意: {(y_original==1).sum()}")
    logger.info(f"  合成数据 - 正常: {(y_synthetic==0).sum()}, 恶意: {(y_synthetic==1).sum()}")
    logger.info(f"  增强后   - 正常: {(y_augmented==0).sum()}, 恶意: {(y_augmented==1).sum()}")
    logger.info("")
    
    # 4. 结构感知检查
    logger.info("4️⃣  Structure-Aware 检查: 依赖特征协方差")
    dep_indices = [0, 1, 4]  # Length, IAT, Window
    real_dep = X_original[:, :, dep_indices].reshape(-1, 3)
    syn_dep = X_synthetic[:, :, dep_indices].reshape(-1, 3)
    
    real_corr = np.corrcoef(real_dep.T)
    syn_corr = np.corrcoef(syn_dep.T)
    
    logger.info("  真实数据相关系数矩阵:")
    logger.info(f"    Length-IAT:    {real_corr[0, 1]:>7.4f}")
    logger.info(f"    Length-Window: {real_corr[0, 2]:>7.4f}")
    logger.info(f"    IAT-Window:    {real_corr[1, 2]:>7.4f}")
    logger.info("  合成数据相关系数矩阵:")
    logger.info(f"    Length-IAT:    {syn_corr[0, 1]:>7.4f}")
    logger.info(f"    Length-Window: {syn_corr[0, 2]:>7.4f}")
    logger.info(f"    IAT-Window:    {syn_corr[1, 2]:>7.4f}")
    
    corr_diff = np.abs(real_corr - syn_corr).mean()
    logger.info(f"  平均相关性差异: {corr_diff:.4f} {'✓' if corr_diff < 0.1 else '⚠'}")
    logger.info("")
    
    # 5. 总体评估
    logger.info("="*70)
    logger.info("总体评估")
    logger.info("="*70)
    
    fidelity_grade = "优秀" if avg_fidelity < 10 else ("良好" if avg_fidelity < 20 else "需改进")
    validity_grade = "优秀" if validity_rate > 99 else ("良好" if validity_rate > 95 else "需改进")
    structure_grade = "保持" if corr_diff < 0.1 else "部分保持"
    
    logger.info(f"  Fidelity (真实性):        {fidelity_grade} (平均差异 {avg_fidelity:.2f}%)")
    logger.info(f"  Protocol Validity (有效性): {validity_grade} (有效率 {validity_rate:.2f}%)")
    logger.info(f"  Structure (结构保持):      {structure_grade} (相关性差异 {corr_diff:.4f})")
    logger.info("")
    
    # 6. 生成可视化
    logger.info("5️⃣  生成可视化: Real vs Synthetic t-SNE 对比图")
    save_path = os.path.join(config.DATA_AUGMENTATION_DIR, "figures", "real_vs_synthetic_tsne_test.png")
    
    try:
        plot_real_vs_synthetic_comparison(
            X_original, X_synthetic,
            y_original, y_synthetic,
            save_path,
            title='TabDDPM Generation Quality Assessment',
            method='tsne'
        )
        logger.info(f"  ✓ t-SNE对比图已保存: {save_path}")
    except Exception as e:
        logger.error(f"  ❌ 生成t-SNE图失败: {e}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("✓ 生成质量诊断完成")
    logger.info("="*70)


def main():
    """主函数"""
    logger.info("="*70)
    logger.info("TabDDPM 生成质量测试")
    logger.info("="*70)
    logger.info("")
    
    # 加载数据
    result = load_augmented_data()
    if result is None:
        return
    
    X_augmented, y_augmented, is_original, n_original = result
    
    # 评估质量
    evaluate_generation_quality(X_augmented, y_augmented, n_original)
    
    logger.info("\n💡 提示:")
    logger.info("  - 查看详细说明: docs/TabDDPM_详细说明.md")
    logger.info("  - 查看可视化结果: output/data_augmentation/figures/")
    logger.info("  - 运行完整训练: python train.py")


if __name__ == "__main__":
    main()
