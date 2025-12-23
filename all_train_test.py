"""
MEDAL-Lite All-in-One Script
Runs complete training and testing pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger

# Import train and test functions
from train import main as train_main
from test import main as test_main

# 导入预处理模块
try:
    from preprocess import check_preprocessed_exists, preprocess_train, preprocess_test
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

import logging


def main(args):
    """Main function to run both training and testing"""
    
    # Setup
    set_seed(config.SEED)
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, "logs"), name='all_train_test')
    
    logger.info("="*70)
    logger.info("🚀 MEDAL-Lite Complete Pipeline: Training + Testing")
    logger.info("="*70)
    logger.info(f"设备: {config.DEVICE}")
    logger.info(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("📋 实验配置:")
    logger.info(f"  训练集路径: {config.BENIGN_TRAIN} (正常), {config.MALICIOUS_TRAIN} (恶意)")
    logger.info(f"  测试集路径: {config.BENIGN_TEST} (正常), {config.MALICIOUS_TEST} (恶意)")
    logger.info(f"  说明: 将读取上述路径下所有pcap文件，流数在处理时统计")
    logger.info(f"  标签噪声率: {config.LABEL_NOISE_RATE*100:.0f}%")
    logger.info("")
    
    # Check start stage
    start_stage = getattr(args, 'start_stage', '1')
    
    # ========================
    # Phase 0: 检查预处理数据
    # ========================
    if PREPROCESS_AVAILABLE:
        logger.info("="*70)
        logger.info("📦 检查预处理数据")
        logger.info("="*70)
        
        train_exists = check_preprocessed_exists('train')
        test_exists = check_preprocessed_exists('test')
        
        if train_exists:
            logger.info("✓ 训练集预处理文件已存在")
        else:
            logger.info("⚠️  训练集预处理文件不存在，将从PCAP加载")
            logger.info("   💡 提示: 运行 'python preprocess.py --train_only' 可预处理训练集")
        
        if test_exists:
            logger.info("✓ 测试集预处理文件已存在")
        else:
            logger.info("⚠️  测试集预处理文件不存在，将从PCAP加载")
            logger.info("   💡 提示: 运行 'python preprocess.py --test_only' 可预处理测试集")
        
        logger.info("")
    
    # ========================
    # Phase 1: Training (if not starting from test)
    # ========================
    if start_stage != "test":
        logger.info("="*70)
        logger.info("📚 PHASE 1: TRAINING 训练阶段")
        logger.info("="*70)
        logger.info("")
        
        try:
            classifier = train_main(args)
            logger.info("")
            logger.info("✓ 训练阶段完成!")
        except Exception as e:
            logger.error(f"❌ 训练阶段失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        logger.info("="*70)
        logger.info("⏭️  跳过训练阶段 (从测试开始)")
        logger.info("="*70)
        logger.info("")
    
    # ========================
    # Phase 2: Testing
    # ========================
    logger.info("")
    logger.info("="*70)
    logger.info("🧪 PHASE 2: TESTING 测试阶段")
    logger.info("="*70)
    logger.info("")
    
    try:
        metrics = test_main(args)
        logger.info("")
        logger.info("✓ 测试阶段完成!")
    except Exception as e:
        logger.error(f"❌ 测试阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================
    # Summary
    # ========================
    logger.info("")
    logger.info("="*70)
    logger.info("🎉 完整流程完成 - 最终总结 PIPELINE COMPLETE - SUMMARY")
    logger.info("="*70)
    logger.info("")
    logger.info("📊 最终测试性能:")
    logger.info(f"  ✓ 准确率 (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  ✓ 精确率 (Precision): {metrics['precision_pos']:.4f} ({metrics['precision_pos']*100:.2f}%)")
    logger.info(f"  ✓ 召回率 (Recall):    {metrics['recall_pos']:.4f} ({metrics['recall_pos']*100:.2f}%)")
    logger.info(f"  ✓ F1分数 (F1-pos):    {metrics['f1_pos']:.4f} ({metrics['f1_pos']*100:.2f}%)  # 论文口径")
    logger.info(f"  ✓ F1-Macro:           {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)")
    if 'auc' in metrics:
        logger.info(f"  ✓ AUC值:              {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)")
    logger.info("")
    logger.info("📁 输出文件位置:")
    logger.info(f"  ✓ 模型文件: {config.CHECKPOINT_DIR}")
    logger.info(f"  ✓ 可视化图表: {os.path.join(config.OUTPUT_ROOT, 'figures')}")
    logger.info(f"  ✓ 测试结果: {os.path.join(config.OUTPUT_ROOT, 'result')}")
    logger.info(f"  ✓ 训练日志: {os.path.join(config.OUTPUT_ROOT, 'logs')}")
    logger.info("")
    logger.info("="*70)
    logger.info("感谢使用 MEDAL-Lite! Thank you for using MEDAL-Lite!")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete MEDAL-Lite training and testing")
    
    # Training arguments
    parser.add_argument("--noise_rate", type=float, default=0.30, 
                       help="Label noise rate (default: 0.30)")
    
    # Stage selection
    parser.add_argument("--start_stage", type=str, default="1", 
                       choices=["1", "2", "3", "test"],
                       help="Start from which stage (1=backbone pretrain, 2=label correction, 3=classifier finetune, test=testing only)")
    
    args = parser.parse_args()
    
    # Override config with arguments
    config.LABEL_NOISE_RATE = args.noise_rate
    
    main(args)

