"""
MEDAL-Lite 日志工具 (重构版)
============================
提供统一的日志格式和阶段性配置输出
"""

import logging
import os
from typing import Optional
from datetime import datetime


_DEFAULT_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
_DEFAULT_DATEFMT = '%H:%M:%S'


def _make_formatter():
    fmt = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
    setattr(fmt, '_medal_formatter', True)
    return fmt


def configure_root_logger(level: int = logging.INFO) -> None:
    """配置根日志记录器"""
    root = logging.getLogger()
    if getattr(root, '_medal_configured', False):
        if root.level != level:
            root.setLevel(level)
        return

    root.setLevel(level)
    formatter = _make_formatter()

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        for h in root.handlers:
            try:
                h.setLevel(level)
            except Exception:
                pass
            try:
                h.setFormatter(formatter)
            except Exception:
                pass

    setattr(root, '_medal_configured', True)


def setup_logger(log_dir: Optional[str] = None, name: str = 'medal', level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    configure_root_logger(level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    if logger.handlers:
        logger.handlers.clear()
    return logger


def log_section_header(logger, title: str, char: str = "=", width: int = 70):
    """输出分节标题"""
    logger.info("")
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)


def log_subsection_header(logger, title: str, char: str = "-", width: int = 70):
    """输出子分节标题"""
    logger.info("")
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)


def log_key_value(logger, key: str, value, indent: int = 2):
    """输出键值对"""
    prefix = " " * indent
    logger.info(f"{prefix}- {key}: {value}")


def log_param_group(logger, title: str, params: dict, indent: int = 2):
    """输出参数组"""
    logger.info(f"\n{title}:")
    for key, value in params.items():
        log_key_value(logger, key, value, indent)


def log_stage_start(logger, stage_name: str, description: str = ""):
    """输出阶段开始标记"""
    log_section_header(logger, f"🚀 {stage_name}")
    if description:
        logger.info(f"目标: {description}")
    logger.info("")


def log_stage_end(logger, stage_name: str, summary: dict = None):
    """输出阶段结束标记"""
    logger.info("")
    log_subsection_header(logger, f"✅ {stage_name} 完成")
    if summary:
        for key, value in summary.items():
            log_key_value(logger, key, value)
    logger.info("")


def log_input_paths(logger, paths: dict):
    """输出输入路径"""
    logger.info("📥 输入数据路径:")
    for name, path in paths.items():
        logger.info(f"  ✓ {name}: {path}")
    logger.info("")


def log_output_paths(logger, paths: dict):
    """输出输出路径"""
    logger.info("📁 输出文件路径:")
    for name, path in paths.items():
        logger.info(f"  ✓ {name}: {path}")
    logger.info("")


def log_training_config(logger, config, stage: str):
    """输出训练配置（委托给config的方法）"""
    if hasattr(config, 'log_stage_config'):
        config.log_stage_config(logger, stage)
    else:
        logger.warning(f"⚠️ config 对象没有 log_stage_config 方法")


def log_data_stats(logger, stats: dict, title: str = "数据统计"):
    """输出数据统计"""
    logger.info(f"📊 {title}:")
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    - {k}: {v}")
        else:
            logger.info(f"  - {key}: {value}")
    logger.info("")


def log_model_info(logger, model_name: str, params: dict):
    """输出模型信息"""
    logger.info(f"🔧 {model_name}:")
    for key, value in params.items():
        logger.info(f"  - {key}: {value}")
    logger.info("")


def log_progress(logger, current: int, total: int, prefix: str = "进度", extra: str = ""):
    """输出进度信息"""
    pct = current / total * 100 if total > 0 else 0
    msg = f"{prefix}: {current}/{total} ({pct:.1f}%)"
    if extra:
        msg += f" | {extra}"
    logger.info(msg)


def log_epoch_metrics(logger, epoch: int, total_epochs: int, metrics: dict, prefix: str = ""):
    """输出epoch指标"""
    pct = (epoch + 1) / total_epochs * 100
    parts = [f"[{prefix}] Epoch [{epoch+1}/{total_epochs}] ({pct:.1f}%)"]
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    logger.info(" | ".join(parts))


def log_early_stopping(logger, epoch: int, best_epoch: int, best_metric: float, 
                       current_metric: float, patience_count: int, patience: int):
    """输出早停信息"""
    logger.info("")
    log_section_header(logger, "🛑 早停触发 (Early Stopping)")
    logger.info(f"  当前轮次: Epoch {epoch}")
    logger.info(f"  最佳指标: {best_metric:.4f} (Epoch {best_epoch})")
    logger.info(f"  当前指标: {current_metric:.4f}")
    logger.info(f"  连续 {patience_count} 轮未改善 (耐心值: {patience})")
    logger.info("")


def log_final_summary(logger, title: str, metrics: dict, paths: dict = None):
    """输出最终总结"""
    log_section_header(logger, f"🎉 {title}")
    logger.info("")
    logger.info("📊 最终性能:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  ✓ {key}: {value:.4f} ({value*100:.2f}%)")
        else:
            logger.info(f"  ✓ {key}: {value}")
    
    if paths:
        logger.info("")
        log_output_paths(logger, paths)
    
    logger.info("=" * 70)
