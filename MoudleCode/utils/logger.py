"""
日志记录工具
"""

import logging
import os
from datetime import datetime

from MoudleCode.utils.logging_utils import setup_logger as _setup_logger


def setup_logger(name, log_file=None, level=logging.INFO):
    """设置日志记录器"""
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    return _setup_logger(log_dir=os.path.dirname(log_file) if log_file else None, name=name, level=level)


def get_timestamp():
    """获取当前时间戳字符串"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

