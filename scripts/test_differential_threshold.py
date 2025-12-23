#!/usr/bin/env python3
"""
测试差异化阈值策略
验证"宽进(变良)严出(变恶)"的效果
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from MoudleCode.label_correction.hybrid_court import HybridCourt, TIER_CONFIG
from MoudleCode.utils.config import Config

def analyze_flip_direction(csv_path):
    """分析CSV中的翻转方向和准确率"""
    df = pd.read_csv(csv_path)
    
    # 筛选Tier 2 Flip样本
    flip_df = df[df['Tier分级'] == 'Tier 2: Flip'].copy()
    
    print(f"\n{'='*70}")
    print(f"翻转样本分析 (基于 {csv_path})")
    print(f"{'='*70}")
    print(f"总翻转样本数: {len(flip_df)}")
    
    # 判断翻转方向
    flip_df['翻转方向'] = flip_df.apply(
        lambda row: '翻转为正常' if row['矫正后标签'] == '正常' else '翻转为恶意',
        axis=1
    )
    
    # 按方向统计
    for direction in ['翻转为正常', '翻转为恶意']:
        subset = flip_df[flip_df['翻转方向'] == direction]
        if len(subset) == 0:
            continue
            
        correct = (subset['矫正是否正确'] == '正确').sum()
        total = len(subset)
        accuracy = 100 * correct / total
        
        print(f"\n{direction}:")
        print(f"  样本数: {total}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.1f}%")
        
        # 分析错误样本的CL置信度分布
        if direction == '翻转为恶意':
            error_subset = subset[subset['矫正是否正确'] == '错误']
            if len(error_subset) > 0:
                cl_target = error_subset['CL恶意概率']
                print(f"\n  错误样本的CL目标置信度分布:")
                print(f"    最小值: {cl_target.min():.3f}")
                print(f"    最大值: {cl_target.max():.3f}")
                print(f"    平均值: {cl_target.mean():.3f}")
                print(f"    中位数: {cl_target.median():.3f}")
                
                # 统计低于不同阈值的错误数
                for threshold in [0.55, 0.60, 0.61, 0.65, 0.70]:
                    count = (cl_target < threshold).sum()
                    pct = 100 * count / len(error_subset)
                    print(f"    CL < {threshold}: {count}/{len(error_subset)} ({pct:.1f}%)")

def test_differential_threshold():
    """测试差异化阈值配置"""
    print(f"\n{'='*70}")
    print(f"差异化阈值配置测试")
    print(f"{'='*70}")
    
    config = TIER_CONFIG['PHASE2_FILTERS']
    
    print(f"\n翻转为正常 (宽松标准):")
    print(f"  CL目标下限: {config['FLIP_TO_NORMAL_CL_MIN']}")
    print(f"  KNN下限:    {config['FLIP_TO_NORMAL_KNN_MIN']}")
    print(f"  策略理由:   数据显示翻转为正常的准确率高达98.4%，风险极低")
    
    print(f"\n翻转为恶意 (严格标准):")
    print(f"  CL目标下限: {config['FLIP_TO_MALICIOUS_CL_MIN']}")
    print(f"  KNN下限:    {config['FLIP_TO_MALICIOUS_KNN_MIN']}")
    print(f"  策略理由:   数据显示CL<0.61时翻转为恶意100%出错，必须严格把关")
    
    print(f"\n预期效果:")
    print(f"  1. 翻转为正常: 保持高召回率，拯救更多被误标的正常样本")
    print(f"  2. 翻转为恶意: 显著降低误判率，避免将正常样本错误标记为恶意")
    print(f"  3. 整体纯度: 在几乎不损失正确矫正数量的前提下，提升至98%以上")

if __name__ == '__main__':
    # 测试配置
    test_differential_threshold()
    
    # 分析现有数据
    csv_path = 'python/MEDAL/output/label_correction/analysis/documents/sample_analysis.csv'
    if os.path.exists(csv_path):
        analyze_flip_direction(csv_path)
    else:
        print(f"\n⚠ 未找到分析文件: {csv_path}")
        print(f"请先运行噪声分析生成sample_analysis.csv")
