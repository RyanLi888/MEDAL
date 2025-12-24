"""
可视化对比学习训练过程
展示SimMTM和InfoNCE损失的收敛曲线
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# 从训练日志中提取的真实数据（最新实验 2025-12-23）
epochs = [1, 2, 3, 4, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 193, 194, 195, 200]
total_loss = [11.79, 6.03, 4.00, 3.26, 2.44, 2.05, 1.92, 1.85, 1.81, 1.52, 1.37, 1.36, 1.38, 1.35, 1.34, 1.36, 1.41, 1.37]
simmtm_loss = [10.52, 4.84, 2.92, 2.31, 1.55, 1.28, 1.18, 1.12, 1.13, 1.05, 1.01, 0.99, 0.97, 0.96, 0.95, 0.95, 0.96, 0.95]
infonce_loss = [4.23, 3.99, 3.57, 3.17, 2.96, 2.56, 2.47, 2.43, 2.26, 1.57, 1.21, 1.23, 1.37, 1.30, 1.28, 1.38, 1.53, 1.40]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('MEDAL Instance Contrastive Learning Training Curves (Latest Experiment)', fontsize=16, fontweight='bold')

# 1. 总损失曲线
ax1 = axes[0, 0]
ax1.plot(epochs, total_loss, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Total Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Total Loss Convergence', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xlim(0, 210)

# 添加关键点标注
ax1.annotate(f'Start: {total_loss[0]:.2f}', 
             xy=(epochs[0], total_loss[0]), 
             xytext=(20, total_loss[0]+1),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red')
ax1.annotate(f'Final: {total_loss[-1]:.2f}', 
             xy=(epochs[-1], total_loss[-1]), 
             xytext=(epochs[-1]-40, total_loss[-1]+1),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
             fontsize=10, color='green')

# 2. SimMTM vs InfoNCE
ax2 = axes[0, 1]
ax2.plot(epochs, simmtm_loss, 'o-', linewidth=2, markersize=6, color='#A23B72', label='SimMTM Loss')
ax2.plot(epochs, infonce_loss, 's-', linewidth=2, markersize=6, color='#F18F01', label='InfoNCE Loss')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('SimMTM vs InfoNCE Loss', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_xlim(0, 210)

# 3. 损失占比堆叠图
ax3 = axes[1, 0]
ax3.fill_between(epochs, 0, simmtm_loss, alpha=0.6, color='#A23B72', label='SimMTM')
ax3.fill_between(epochs, simmtm_loss, total_loss, alpha=0.6, color='#F18F01', label='InfoNCE')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Loss Component Stacking', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)
ax3.set_xlim(0, 210)

# 4. InfoNCE占比变化
ax4 = axes[1, 1]
infonce_ratio = [infonce_loss[i] / total_loss[i] * 100 for i in range(len(epochs))]
ax4.plot(epochs, infonce_ratio, 'o-', linewidth=2, markersize=6, color='#C73E1D')
ax4.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% Reference')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('InfoNCE Ratio (%)', fontsize=12)
ax4.set_title('InfoNCE Loss Proportion', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=11)
ax4.set_xlim(0, 210)
ax4.set_ylim(0, 100)

# 添加文本说明
textstr = '\n'.join([
    'Configuration:',
    f'• Temperature τ = 0.3',
    f'• Weight λ = 0.3',
    f'• Epochs = 200',
    f'• Batch Size = 64',
    '',
    'Final Results:',
    f'• Total Loss: {total_loss[-1]:.2f}',
    f'• SimMTM: {simmtm_loss[-1]:.2f}',
    f'• InfoNCE: {infonce_loss[-1]:.2f}',
    f'• InfoNCE Ratio: {infonce_ratio[-1]:.1f}%'
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax4.text(0.98, 0.02, textstr, transform=ax4.transAxes, fontsize=9,
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()

# 保存图表
output_dir = 'output/feature_extraction/figures'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'contrastive_learning_curves.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ 训练曲线已保存: {output_path}")

# 显示图表
plt.show()

# 打印统计信息
print("\n" + "="*70)
print("对比学习训练统计")
print("="*70)
print(f"\n初始状态 (Epoch 1):")
print(f"  总损失: {total_loss[0]:.2f}")
print(f"  SimMTM: {simmtm_loss[0]:.2f} ({simmtm_loss[0]/total_loss[0]*100:.1f}%)")
print(f"  InfoNCE: {infonce_loss[0]:.2f} ({infonce_loss[0]/total_loss[0]*100:.1f}%)")

print(f"\n最终状态 (Epoch 200):")
print(f"  总损失: {total_loss[-1]:.2f}")
print(f"  SimMTM: {simmtm_loss[-1]:.2f} ({simmtm_loss[-1]/total_loss[-1]*100:.1f}%)")
print(f"  InfoNCE: {infonce_loss[-1]:.2f} ({infonce_loss[-1]/total_loss[-1]*100:.1f}%)")

print(f"\n损失下降:")
print(f"  总损失: {total_loss[0]:.2f} → {total_loss[-1]:.2f} (↓{(1-total_loss[-1]/total_loss[0])*100:.1f}%)")
print(f"  SimMTM: {simmtm_loss[0]:.2f} → {simmtm_loss[-1]:.2f} (↓{(1-simmtm_loss[-1]/simmtm_loss[0])*100:.1f}%)")
print(f"  InfoNCE: {infonce_loss[0]:.2f} → {infonce_loss[-1]:.2f} (↓{(1-infonce_loss[-1]/infonce_loss[0])*100:.1f}%)")

print(f"\nInfoNCE占比变化:")
print(f"  初始: {infonce_ratio[0]:.1f}%")
print(f"  中期 (Epoch 100): {infonce_ratio[8]:.1f}%")
print(f"  最终: {infonce_ratio[-1]:.1f}%")

print("\n" + "="*70)
