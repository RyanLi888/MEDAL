import logging
from typing import Any, Dict, Tuple

import torch


def load_state_dict_shape_safe(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
    logger: logging.Logger,
    prefix: str = "model",
) -> Tuple[list, list]:
    model_sd = model.state_dict()
    filtered: Dict[str, Any] = {}
    skipped_missing = []
    skipped_shape = []

    for k, v in state_dict.items():
        if k not in model_sd:
            skipped_missing.append(k)
            continue
        try:
            if tuple(getattr(v, 'shape', ())) != tuple(getattr(model_sd[k], 'shape', ())):
                skipped_shape.append((k, tuple(getattr(v, 'shape', None)), tuple(getattr(model_sd[k], 'shape', None))))
                continue
        except Exception:
            skipped_shape.append((k, None, None))
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if skipped_shape:
        logger.warning(
            f"⚠ {prefix} checkpoint contains shape-mismatched keys; they were skipped (showing up to 20):"
        )
        for k, src, dst in skipped_shape[:20]:
            logger.warning(f"  - {k}: ckpt={src} model={dst}")
    if skipped_missing:
        logger.warning(f"⚠ {prefix} checkpoint contains unknown keys; they were skipped (showing up to 20):")
        for k in skipped_missing[:20]:
            logger.warning(f"  - {k}")
    if missing:
        logger.warning(f"⚠ {prefix} missing_keys after loading (showing up to 20): {missing[:20]}")
    if unexpected:
        logger.warning(f"⚠ {prefix} unexpected_keys after loading (showing up to 20): {unexpected[:20]}")

    logger.info(
        f"✓ {prefix} state_dict loaded (matched={len(filtered)}/{len(state_dict)}; "
        f"skipped_shape={len(skipped_shape)} skipped_unknown={len(skipped_missing)})"
    )

    return missing, unexpected
