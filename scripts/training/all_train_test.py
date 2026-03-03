"""
MEDAL-Lite 完整训练+测试脚本 (重构版)
=====================================
运行完整的训练和测试流程，复用主流程代码
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime
import json

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger
from MoudleCode.utils.logging_utils import (
    log_section_header, log_final_summary
)

from scripts.training.train import main as train_main
from scripts.training.common import (
    safe_makedirs,
    rng_fingerprint_short,
    seed_snapshot,
    add_finetune_backbone_cli_args,
    apply_finetune_backbone_override,
)
from scripts.testing.test import main as test_main

try:
    from scripts.utils.preprocess import check_preprocessed_exists
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

def main(args):
    """主函数：运行训练和测试"""

    seed = getattr(args, 'seed', None) or getattr(config, 'SEED', None) or 42
    config.SEED = seed
    rng_fp_before_seed = rng_fingerprint_short()
    set_seed(seed)
    rng_fp_after_seed = rng_fingerprint_short()
    config.create_dirs()
    startup_warnings = []
    
    # 如果指定了实验目录（用于测试），加载实验元数据并设置输出目录
    if getattr(args, 'experiment_dir', None):
        experiment_dir = args.experiment_dir
        metadata_path = os.path.join(experiment_dir, 'experiment_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                experiment_metadata = json.load(f)
            dirs = experiment_metadata.get('directories', {})
            config.FEATURE_EXTRACTION_DIR = dirs.get('feature_extraction', config.FEATURE_EXTRACTION_DIR)
            config.LABEL_CORRECTION_DIR = dirs.get('label_correction', config.LABEL_CORRECTION_DIR)
            config.DATA_AUGMENTATION_DIR = dirs.get('data_augmentation', config.DATA_AUGMENTATION_DIR)
            config.CLASSIFICATION_DIR = dirs.get('classification', config.CLASSIFICATION_DIR)
            config.RESULT_DIR = dirs.get('result', config.RESULT_DIR)
            run_tag = experiment_metadata.get('run_tag', os.path.basename(experiment_dir))
        else:
            startup_warnings.append(f"⚠ 实验元数据不存在: {metadata_path}")
            startup_warnings.append("  使用默认输出目录")
            run_tag = os.path.basename(experiment_dir)
    else:
        # 创建新的实验文件夹（基于时间戳）
        run_tag = getattr(args, 'run_tag', None) or datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = os.path.join(config.OUTPUT_ROOT, 'experiments', run_tag)
        safe_makedirs(experiment_dir)
        
        # 保存原始输出目录
        original_feature_extraction_dir = config.FEATURE_EXTRACTION_DIR
        original_label_correction_dir = config.LABEL_CORRECTION_DIR
        original_data_augmentation_dir = config.DATA_AUGMENTATION_DIR
        original_classification_dir = config.CLASSIFICATION_DIR
        original_result_dir = config.RESULT_DIR
        
        # 修改输出目录，使其指向实验文件夹
        config.FEATURE_EXTRACTION_DIR = os.path.join(experiment_dir, 'feature_extraction')
        config.LABEL_CORRECTION_DIR = os.path.join(experiment_dir, 'label_correction')
        config.DATA_AUGMENTATION_DIR = os.path.join(experiment_dir, 'data_augmentation')
        config.CLASSIFICATION_DIR = os.path.join(experiment_dir, 'classification')
        config.RESULT_DIR = os.path.join(experiment_dir, 'result')
        
        # 创建实验文件夹下的子目录
        for module_dir in [config.FEATURE_EXTRACTION_DIR, config.LABEL_CORRECTION_DIR,
                          config.DATA_AUGMENTATION_DIR, config.CLASSIFICATION_DIR, config.RESULT_DIR]:
            safe_makedirs(module_dir)
            safe_makedirs(os.path.join(module_dir, 'models'))
            safe_makedirs(os.path.join(module_dir, 'figures'))
            safe_makedirs(os.path.join(module_dir, 'logs'))
        
        # 保存实验元数据
        experiment_metadata = {
            'run_tag': run_tag,
            'experiment_dir': experiment_dir,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'seed': seed,
            'start_stage': getattr(args, 'start_stage', '1'),
            'backbone_path': getattr(args, 'backbone_path', None),
            'noise_rate': config.LABEL_NOISE_RATE,
            'directories': {
                'feature_extraction': config.FEATURE_EXTRACTION_DIR,
                'label_correction': config.LABEL_CORRECTION_DIR,
                'data_augmentation': config.DATA_AUGMENTATION_DIR,
                'classification': config.CLASSIFICATION_DIR,
                'result': config.RESULT_DIR,
            }
        }
        metadata_path = os.path.join(experiment_dir, 'experiment_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_metadata, f, ensure_ascii=False, indent=2)
    
    # 创建日志目录
    log_dir = os.path.join(experiment_dir, 'logs')
    safe_makedirs(log_dir)
    logger = setup_logger(log_dir, name='all_train_test')
    for msg in startup_warnings:
        logger.warning(msg)

    logger.info(f"🔧 RNG指纹(seed前): {rng_fp_before_seed}")
    logger.info(f"🔧 RNG指纹(seed后): {rng_fp_after_seed} ({seed_snapshot(config, seed)})")
    
    log_section_header(logger, "🚀 MEDAL-Lite 完整流程: 训练 + 测试")
    logger.info(f"设备: {config.DEVICE}")
    logger.info(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"实验目录: {experiment_dir}")
    logger.info(f"实验标签: {run_tag}")
    
    # 输出配置摘要
    logger.info("")
    logger.info("📋 实验配置:")
    logger.info(f"  训练集: {config.BENIGN_TRAIN}, {config.MALICIOUS_TRAIN}")
    logger.info(f"  测试集: {config.BENIGN_TEST}, {config.MALICIOUS_TEST}")
    logger.info(f"  标签噪声率: {config.LABEL_NOISE_RATE*100:.0f}%")
    logger.info("")
    
    if not getattr(args, 'experiment_dir', None):
        logger.info(f"✓ 实验元数据已保存: {metadata_path}")
        logger.info("")
    
    start_stage = getattr(args, 'start_stage', '1')
    
    # 检查预处理数据
    if PREPROCESS_AVAILABLE:
        log_section_header(logger, "📦 检查预处理数据")
        train_exists = check_preprocessed_exists('train')
        test_exists = check_preprocessed_exists('test')
        
        logger.info(f"训练集预处理: {'✓ 存在' if train_exists else '⚠️ 不存在'}")
        logger.info(f"测试集预处理: {'✓ 存在' if test_exists else '⚠️ 不存在'}")
        logger.info("")
    
    # 训练阶段
    if start_stage != "test":
        log_section_header(logger, "📚 PHASE 1: 训练阶段")
        
        try:
            classifier = train_main(args)
            logger.info("✓ 训练阶段完成!")
        except Exception as e:
            logger.error(f"❌ 训练阶段失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        logger.info("⏭️ 跳过训练阶段")
    
    # 测试阶段
    log_section_header(logger, "🧪 PHASE 2: 测试阶段")
    
    try:
        # 确保测试时使用实验目录中的模型
        # 如果backbone_path未指定，尝试从实验目录加载
        if not getattr(args, 'backbone_path', None):
            # 尝试从分类器元数据中获取backbone路径
            classifier_metadata_path = os.path.join(config.CLASSIFICATION_DIR, 'models', 'model_metadata.json')
            if os.path.exists(classifier_metadata_path):
                with open(classifier_metadata_path, 'r', encoding='utf-8') as f:
                    classifier_metadata = json.load(f)
                backbone_path_from_meta = classifier_metadata.get('backbone_path')
                if backbone_path_from_meta and os.path.exists(backbone_path_from_meta):
                    args.backbone_path = backbone_path_from_meta
                    logger.info(f"✓ 从分类器元数据获取骨干网络路径: {backbone_path_from_meta}")
                else:
                    # 尝试使用实验目录中的backbone
                    experiment_backbone = os.path.join(config.FEATURE_EXTRACTION_DIR, 'models', 'backbone_pretrained.pth')
                    if os.path.exists(experiment_backbone):
                        args.backbone_path = experiment_backbone
                        logger.info(f"✓ 使用实验目录中的骨干网络: {experiment_backbone}")
        
        metrics = test_main(args)
        logger.info("✓ 测试阶段完成!")
    except Exception as e:
        logger.error(f"❌ 测试阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 最终总结
    log_final_summary(logger, "完整流程完成", {
        "准确率": metrics['accuracy'],
        "精确率": metrics['precision_pos'],
        "召回率": metrics['recall_pos'],
        "F1分数": metrics['f1_pos'],
        "AUC": metrics.get('auc', 'N/A')
    }, {
        "实验目录": experiment_dir,
        "骨干网络": os.path.join(config.FEATURE_EXTRACTION_DIR, 'models'),
        "标签矫正": os.path.join(config.LABEL_CORRECTION_DIR, 'models'),
        "数据增强": os.path.join(config.DATA_AUGMENTATION_DIR, 'models'),
        "分类器": os.path.join(config.CLASSIFICATION_DIR, 'models'),
        "测试结果": config.RESULT_DIR,
        "训练日志": log_dir
    })
    
    # 恢复原始输出目录（如果需要）
    # 注意：这里不恢复，因为后续可能还需要使用实验目录
    # config.FEATURE_EXTRACTION_DIR = original_feature_extraction_dir
    # config.LABEL_CORRECTION_DIR = original_label_correction_dir
    # config.DATA_AUGMENTATION_DIR = original_data_augmentation_dir
    # config.CLASSIFICATION_DIR = original_classification_dir
    # config.RESULT_DIR = original_result_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEDAL-Lite 完整训练+测试")
    
    parser.add_argument("--noise_rate", type=float, default=None, help="标签噪声率（默认使用config.LABEL_NOISE_RATE）")
    parser.add_argument("--start_stage", type=str, default="1", 
                       choices=["1", "2", "3", "test"], help="起始阶段")
    parser.add_argument("--backbone_path", type=str, default=None, help="骨干网络路径")
    add_finetune_backbone_cli_args(
        parser,
        enable_help="启用骨干微调（训练过程自动适配）",
        disable_help="禁用骨干微调",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖config.SEED）")
    parser.add_argument("--run_tag", type=str, default=None, help="实验标签（默认使用时间戳）")
    parser.add_argument("--experiment_dir", type=str, default=None, help="实验目录路径（用于测试时指定）")
    
    args = parser.parse_args()
    if args.noise_rate is not None:
        config.LABEL_NOISE_RATE = args.noise_rate
    
    apply_finetune_backbone_override(args, config)
    
    main(args)
