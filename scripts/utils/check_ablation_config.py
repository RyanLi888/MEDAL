#!/usr/bin/env python
"""
检查消融实验配置是否使用最优策略

验证点：
1. train_clean_only_then_test.py 是否使用最优策略
2. train.py 的 Stage3 是否使用最优策略
3. 消融实验1、2、3是否都使用相同的最优策略
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from MoudleCode.utils.config import config


def check_optimal_config():
    """检查配置是否为最优策略"""
    print("=" * 70)
    print("检查消融实验配置")
    print("=" * 70)
    print()
    
    # 最优策略配置
    optimal_config = {
        'USE_FOCAL_LOSS': True,
        'USE_BCE_LOSS': False,
        'USE_LOGIT_MARGIN': False,
        'USE_MARGIN_LOSS': False,
        'USE_SOFT_F1_LOSS': False,
        'SOFT_ORTH_WEIGHT_START': 0.0,
        'SOFT_ORTH_WEIGHT_END': 0.0,
        'CONSISTENCY_WEIGHT_START': 0.0,
        'CONSISTENCY_WEIGHT_END': 0.0,
        'STAGE3_ONLINE_AUGMENTATION': False,
        'STAGE3_USE_ST_MIXUP': False,
        'FINETUNE_BACKBONE': True,
        'FINETUNE_BACKBONE_SCOPE': 'projection',
        'FINETUNE_BACKBONE_LR': 2e-5,
        'FINETUNE_BACKBONE_WARMUP_EPOCHS': 30,
        'FOCAL_ALPHA': 0.5,
        'FOCAL_GAMMA': 2.0,
    }
    
    print("📋 最优策略配置:")
    print("-" * 70)
    for key, expected_value in optimal_config.items():
        actual_value = getattr(config, key, None)
        status = "✓" if actual_value == expected_value else "✗"
        print(f"  {status} {key}: {actual_value} (期望: {expected_value})")
    print()
    
    # 检查是否所有配置都正确
    all_correct = all(
        getattr(config, key, None) == expected_value
        for key, expected_value in optimal_config.items()
    )
    
    if all_correct:
        print("✅ 所有配置项均符合最优策略！")
    else:
        print("⚠️  部分配置项不符合最优策略，请检查！")
    
    print()
    print("=" * 70)
    print("消融实验说明")
    print("=" * 70)
    print()
    print("消融实验1 - 特征提取:")
    print("  - 训练骨干网络(Stage1) 或 使用已有骨干")
    print("  - 用真实标签(无噪声)训练分类器")
    print("  - 使用最优策略: Focal Loss + projection微调")
    print()
    print("消融实验2 - 数据增强:")
    print("  - 使用真实标签(无噪声)提取特征")
    print("  - TabDDPM数据增强")
    print("  - 用真实+增强数据训练分类器")
    print("  - 使用最优策略: Focal Loss + projection微调")
    print()
    print("消融实验3 - 标签矫正:")
    print("  - 使用30%噪声提取特征")
    print("  - Hybrid Court标签矫正")
    print("  - 用矫正后的数据训练分类器")
    print("  - 使用最优策略: Focal Loss + projection微调")
    print()
    print("=" * 70)
    
    return all_correct


if __name__ == '__main__':
    success = check_optimal_config()
    sys.exit(0 if success else 1)
