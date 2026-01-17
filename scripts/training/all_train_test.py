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
import hashlib
import random

from MoudleCode.utils.config import config
from MoudleCode.utils.helpers import set_seed, setup_logger
from MoudleCode.utils.logging_utils import (
    log_section_header, log_data_stats, log_final_summary
)

import numpy as np
import torch

from scripts.training.train import main as train_main
from scripts.testing.test import main as test_main

try:
    from scripts.utils.preprocess import check_preprocessed_exists
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False


def main(args):
    """主函数：运行训练和测试"""

    def _rng_fingerprint_short() -> str:
        h = hashlib.sha256()
        try:
            h.update(repr(random.getstate()).encode('utf-8'))
        except Exception:
            h.update(b'py_random_error')
        try:
            ns = np.random.get_state()
            h.update(str(ns[0]).encode('utf-8'))
            h.update(np.asarray(ns[1], dtype=np.uint32).tobytes())
            h.update(str(ns[2]).encode('utf-8'))
            h.update(str(ns[3]).encode('utf-8'))
            h.update(str(ns[4]).encode('utf-8'))
        except Exception:
            h.update(b'numpy_random_error')
        try:
            h.update(torch.get_rng_state().detach().cpu().numpy().tobytes())
        except Exception:
            h.update(b'torch_cpu_rng_error')
        try:
            if torch.cuda.is_available():
                for s in torch.cuda.get_rng_state_all():
                    h.update(s.detach().cpu().numpy().tobytes())
            else:
                h.update(b'no_cuda')
        except Exception:
            h.update(b'torch_cuda_rng_error')
        return h.hexdigest()[:16]

    seed = getattr(args, 'seed', None) or getattr(config, 'SEED', None) or 42
    config.SEED = seed
    rng_fp_before_seed = _rng_fingerprint_short()
    set_seed(seed)
    rng_fp_after_seed = _rng_fingerprint_short()
    config.create_dirs()
    logger = setup_logger(os.path.join(config.OUTPUT_ROOT, "logs"), name='all_train_test')

    logger.info(f"🔧 RNG指纹(seed前): {rng_fp_before_seed}")
    logger.info(f"🔧 RNG指纹(seed后): {rng_fp_after_seed} (args.seed={seed} | config.SEED={int(getattr(config, 'SEED', -1))} | torch.initial_seed={int(torch.initial_seed()) if hasattr(torch, 'initial_seed') else 'N/A'})")
    
    log_section_header(logger, "🚀 MEDAL-Lite 完整流程: 训练 + 测试")
    logger.info(f"设备: {config.DEVICE}")
    logger.info(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 输出配置摘要
    logger.info("")
    logger.info("📋 实验配置:")
    logger.info(f"  训练集: {config.BENIGN_TRAIN}, {config.MALICIOUS_TRAIN}")
    logger.info(f"  测试集: {config.BENIGN_TEST}, {config.MALICIOUS_TEST}")
    logger.info(f"  标签噪声率: {config.LABEL_NOISE_RATE*100:.0f}%")
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
        "模型文件": config.CHECKPOINT_DIR,
        "测试结果": config.RESULT_DIR,
        "训练日志": os.path.join(config.OUTPUT_ROOT, 'logs')
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEDAL-Lite 完整训练+测试")
    
    parser.add_argument("--noise_rate", type=float, default=None, help="标签噪声率（默认使用config.LABEL_NOISE_RATE）")
    parser.add_argument("--start_stage", type=str, default="1", 
                       choices=["1", "2", "3", "test"], help="起始阶段")
    parser.add_argument("--backbone_path", type=str, default=None, help="骨干网络路径")
    parser.add_argument("--finetune_backbone", action="store_true", help="启用骨干微调")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖config.SEED）")
    
    args = parser.parse_args()
    if args.noise_rate is not None:
        config.LABEL_NOISE_RATE = args.noise_rate
    
    if args.finetune_backbone:
        config.FINETUNE_BACKBONE = True
    
    main(args)
