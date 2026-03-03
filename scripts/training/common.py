"""
训练脚本公共工具：
- CLI 中骨干微调开关
- RNG 指纹与种子快照
- Stage4 原始序列加载
- 通用目录创建
"""
import argparse
import hashlib
import os
import random
from typing import Any, Optional, Tuple

import numpy as np
import torch


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rng_fingerprint_short() -> str:
    h = hashlib.sha256()
    try:
        h.update(repr(random.getstate()).encode("utf-8"))
    except Exception:
        h.update(b"py_random_error")
    try:
        ns = np.random.get_state()
        h.update(str(ns[0]).encode("utf-8"))
        h.update(np.asarray(ns[1], dtype=np.uint32).tobytes())
        h.update(str(ns[2]).encode("utf-8"))
        h.update(str(ns[3]).encode("utf-8"))
        h.update(str(ns[4]).encode("utf-8"))
    except Exception:
        h.update(b"numpy_random_error")
    try:
        h.update(torch.get_rng_state().detach().cpu().numpy().tobytes())
    except Exception:
        h.update(b"torch_cpu_rng_error")
    try:
        if torch.cuda.is_available():
            for s in torch.cuda.get_rng_state_all():
                h.update(s.detach().cpu().numpy().tobytes())
        else:
            h.update(b"no_cuda")
    except Exception:
        h.update(b"torch_cuda_rng_error")
    return h.hexdigest()[:16]


def seed_snapshot(config_obj: Any, args_seed: Optional[int] = None) -> str:
    torch_seed = None
    try:
        torch_seed = int(torch.initial_seed())
    except Exception:
        torch_seed = None

    parts = []
    if args_seed is not None:
        try:
            parts.append(f"args.seed={int(args_seed)}")
        except Exception:
            parts.append(f"args.seed={args_seed}")
    parts.append(f"config.SEED={int(getattr(config_obj, 'SEED', -1))}")
    parts.append(f"torch.initial_seed={torch_seed}")
    return " | ".join(parts)


def add_finetune_backbone_cli_args(
    parser: argparse.ArgumentParser,
    enable_help: str = "启用骨干微调（需原始序列参与）",
    disable_help: str = "禁用骨干微调",
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--finetune_backbone",
        dest="finetune_backbone",
        action="store_true",
        help=enable_help,
    )
    group.add_argument(
        "--no_finetune_backbone",
        dest="finetune_backbone",
        action="store_false",
        help=disable_help,
    )
    parser.set_defaults(finetune_backbone=None)


def apply_finetune_backbone_override(args: Any, config_obj: Any) -> None:
    value = getattr(args, "finetune_backbone", None)
    if value is not None:
        config_obj.FINETUNE_BACKBONE = bool(value)


def load_stage4_real_sequences(
    data_augmentation_dir: str,
    logger: Optional[Any] = None,
) -> Tuple[Optional[np.ndarray], str]:
    real_kept_path = os.path.join(data_augmentation_dir, "models", "real_kept_data.npz")
    if not os.path.exists(real_kept_path):
        return None, real_kept_path

    try:
        real_pack = np.load(real_kept_path)
        x_real = real_pack.get("X_real", None)
        if x_real is None and logger is not None:
            logger.warning(f"⚠ real_kept_data中缺少X_real字段: {real_kept_path}")
        return x_real, real_kept_path
    except Exception as exc:
        if logger is not None:
            logger.warning(f"⚠ 读取原始序列失败: {real_kept_path} ({exc})")
        return None, real_kept_path
