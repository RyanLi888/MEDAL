"""
MEDAL-Lite All-in-One Script
Runs complete training and testing pipeline
"""
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path when running as a script
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger

# Import train and test functions
from scripts.training.train import main as train_main
from scripts.testing.test import main as test_main

# 导入预处理模块
try:
    from scripts.utils.preprocess import check_preprocessed_exists, preprocess_train, preprocess_test
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
    
    # Backbone selection
    parser.add_argument("--backbone_path", type=str, default=None,
                       help="Path to specific backbone model (e.g., backbone_SimCLR_500.pth)")
    
    # Backbone finetuning arguments (Stage 3)
    parser.add_argument("--finetune_backbone", action="store_true",
                       help="Enable backbone finetuning in Stage 3 (default: False)")
    parser.add_argument("--finetune_backbone_lr", type=float, default=None,
                       help="Learning rate for backbone finetuning (default: 2e-5)")
    parser.add_argument("--finetune_backbone_warmup", type=int, default=None,
                       help="Warmup epochs before backbone finetuning (default: 30)")
    parser.add_argument("--finetune_backbone_scope", type=str, default=None,
                       choices=["projection", "all"],
                       help="Scope of backbone finetuning: projection or all (default: projection)")
    
    args = parser.parse_args()
    
    # Override config with arguments
    config.LABEL_NOISE_RATE = args.noise_rate
    
    # Apply backbone finetuning settings from command line or environment variables
    # Priority: command line args > environment variables > config defaults
    if args.finetune_backbone or os.environ.get('MEDAL_FINETUNE_BACKBONE', '').strip().lower() in ('1', 'true', 'yes', 'y', 'on'):
        config.FINETUNE_BACKBONE = True
        
        # Set learning rate
        if args.finetune_backbone_lr is not None:
            config.FINETUNE_BACKBONE_LR = args.finetune_backbone_lr
        elif os.environ.get('MEDAL_FINETUNE_BACKBONE_LR', '').strip():
            try:
                config.FINETUNE_BACKBONE_LR = float(os.environ.get('MEDAL_FINETUNE_BACKBONE_LR'))
            except ValueError:
                pass
        else:
            config.FINETUNE_BACKBONE_LR = 2e-5  # Default value
        
        # Set warmup epochs
        if args.finetune_backbone_warmup is not None:
            config.FINETUNE_BACKBONE_WARMUP_EPOCHS = args.finetune_backbone_warmup
        elif os.environ.get('MEDAL_FINETUNE_BACKBONE_WARMUP_EPOCHS', '').strip():
            try:
                config.FINETUNE_BACKBONE_WARMUP_EPOCHS = int(float(os.environ.get('MEDAL_FINETUNE_BACKBONE_WARMUP_EPOCHS')))
            except ValueError:
                pass
        else:
            config.FINETUNE_BACKBONE_WARMUP_EPOCHS = 30  # Default value
        
        # Set scope
        if args.finetune_backbone_scope is not None:
            config.FINETUNE_BACKBONE_SCOPE = args.finetune_backbone_scope
        elif os.environ.get('MEDAL_FINETUNE_BACKBONE_SCOPE', '').strip().lower() in ('projection', 'all'):
            config.FINETUNE_BACKBONE_SCOPE = os.environ.get('MEDAL_FINETUNE_BACKBONE_SCOPE').strip().lower()
        else:
            config.FINETUNE_BACKBONE_SCOPE = 'projection'  # Default value
    
    main(args)

