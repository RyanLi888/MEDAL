"""
训练数据加载工具。
统一处理：
- 优先读取预处理数据
- 回退到PCAP解析
- 可选的特征归一化
"""
from typing import Tuple

import numpy as np

from MoudleCode.preprocessing.pcap_parser import load_dataset

try:
    from scripts.utils.preprocess import (
        check_preprocessed_exists,
        load_preprocessed,
        normalize_burstsize_inplace,
    )

    PREPROCESS_AVAILABLE = True
except Exception:
    PREPROCESS_AVAILABLE = False

    def normalize_burstsize_inplace(x: np.ndarray) -> np.ndarray:
        return x


def load_train_dataset(config, prefer_preprocessed: bool = True, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, str]:
    source = "pcap"
    if prefer_preprocessed and PREPROCESS_AVAILABLE and check_preprocessed_exists("train"):
        x_train, y_train, _ = load_preprocessed("train")
        source = "preprocessed"
    else:
        x_train, y_train, _ = load_dataset(
            benign_dir=config.BENIGN_TRAIN,
            malicious_dir=config.MALICIOUS_TRAIN,
            sequence_length=config.SEQUENCE_LENGTH,
        )

    if normalize and x_train is not None:
        x_train = normalize_burstsize_inplace(x_train)

    return x_train, y_train, source
