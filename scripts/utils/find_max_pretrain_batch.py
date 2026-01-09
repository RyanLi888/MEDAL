"""
Find the maximum Stage-1 pretraining batch size that fits on the current GPU.

This runs a single forward+backward+step for the same compute graph used in Stage 1
(SimMTM++ reconstruction + instance contrastive loss), and reports the largest
batch size that does not OOM.

Usage examples:
  MEDAL_GPU_ID=0 python scripts/utils/find_max_pretrain_batch.py
  MEDAL_GPU_ID=0 MEDAL_CONTRASTIVE_METHOD=nnclr python scripts/utils/find_max_pretrain_batch.py --max 256
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))


def _try_step(batch_size: int, device: str) -> tuple[bool, str]:
    import torch
    import torch.optim as optim

    from MoudleCode.utils.config import config
    from MoudleCode.feature_extraction.backbone import SimMTMLoss, build_backbone
    from MoudleCode.feature_extraction.traffic_augmentation import DualViewAugmentation
    from MoudleCode.feature_extraction.instance_contrastive import (
        InstanceContrastiveLearning,
        HybridPretrainingLoss,
    )

    torch.cuda.reset_peak_memory_stats()

    backbone = build_backbone(config, logger=None).to(device)
    backbone.train()

    use_instance_contrastive = bool(getattr(config, "USE_INSTANCE_CONTRASTIVE", True))
    if not use_instance_contrastive:
        return False, "USE_INSTANCE_CONTRASTIVE is disabled in config; enable it to test the full Stage-1 graph."

    augmentation = DualViewAugmentation(config)
    instance_contrastive = InstanceContrastiveLearning(backbone, config).to(device)
    simmtm = SimMTMLoss(
        config,
        mask_rate=float(getattr(config, "SIMMTM_MASK_RATE", 0.5)),
        noise_std=float(getattr(config, "PRETRAIN_NOISE_STD", 0.05)),
    )
    hybrid = HybridPretrainingLoss(
        simmtm_loss=simmtm,
        instance_contrastive=instance_contrastive,
        lambda_infonce=float(getattr(config, "INFONCE_LAMBDA", 0.3)),
    )

    optimizer = optim.AdamW(
        list(backbone.parameters()) + list(instance_contrastive.projection_head.parameters()),
        lr=float(getattr(config, "PRETRAIN_LR", 1e-3)),
        weight_decay=float(getattr(config, "PRETRAIN_WEIGHT_DECAY", 1e-4)),
    )

    L = int(getattr(config, "SEQUENCE_LENGTH", 1024))
    D = int(getattr(config, "INPUT_FEATURE_DIM", 4))
    x = torch.zeros((batch_size, L, D), device=device, dtype=torch.float32)

    # Set valid_mask to 1 for all tokens so we test the worst-case compute.
    vm_idx = getattr(config, "VALID_MASK_INDEX", None)
    if vm_idx is not None and 0 <= int(vm_idx) < D:
        x[:, :, int(vm_idx)] = 1.0

    try:
        optimizer.zero_grad(set_to_none=True)
        x_view1, x_view2 = augmentation(x)
        loss, _loss_dict = hybrid(backbone=backbone, x_original=x, x_view1=x_view1, x_view2=x_view2, epoch=0)
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated() / (1024**2))
        return True, f"ok (peak_alloc={peak} MiB)"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, "OOM"
        return False, f"RuntimeError: {e}"
    finally:
        del optimizer, hybrid, simmtm, instance_contrastive, augmentation, backbone, x
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.empty_cache()


def main() -> int:
    _ensure_project_root_on_path()

    parser = argparse.ArgumentParser(description="Find max pretrain batch size that fits GPU memory.")
    parser.add_argument("--min", type=int, default=16)
    parser.add_argument("--max", type=int, default=256)
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0 or cpu")
    args = parser.parse_args()

    from MoudleCode.utils.config import config

    device = args.device or str(getattr(config, "DEVICE", "cpu"))
    if "cuda" in device and (os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == ""):
        # Best effort reminder; config may remap devices if CUDA_VISIBLE_DEVICES is set.
        pass

    bs_values = list(range(int(args.min), int(args.max) + 1, int(args.step)))
    best = None
    print(f"Device: {device}")
    print(f"Testing batch sizes: {bs_values}")

    for bs in bs_values:
        ok, msg = _try_step(bs, device)
        print(f"  bs={bs:4d}: {msg}")
        if ok:
            best = bs
        else:
            # Once we hit OOM, higher batch sizes will very likely OOM too.
            if msg == "OOM":
                break

    if best is None:
        print("No batch size in range succeeded.")
        return 1

    print(f"\nMax batch size (in this test): {best}")
    print("Tip: If you want more negatives without increasing peak activation memory, keep batch fixed and set:")
    print("  MEDAL_PRETRAIN_GRADIENT_ACCUMULATION_STEPS=2/4 and/or MEDAL_CONTRASTIVE_METHOD=nnclr with MEDAL_NNCLR_QUEUE_SIZE=8192+")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

