"""
模型加载辅助函数
"""
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_backbone_safely(
    backbone_path: str,
    config,
    device: str = 'cuda',
    logger: Optional[logging.Logger] = None
):
    """
    安全加载骨干网络
    
    Args:
        backbone_path: 模型路径
        config: 配置对象
        device: 设备
        logger: 日志记录器（可选）
        
    Returns:
        backbone: 加载后的模型
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    from MoudleCode.feature_extraction.backbone import build_backbone
    
    # 先加载检查点
    load_location = 'cpu' if device == 'cpu' else 'cpu'
    try:
        checkpoint = torch.load(backbone_path, map_location=load_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(backbone_path, map_location=load_location)
    
    # 获取状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'backbone_state_dict' in checkpoint:
        state_dict = checkpoint['backbone_state_dict']
    else:
        state_dict = checkpoint
    
    # 构建模型
    backbone = build_backbone(config, logger)
    
    # 加载权重
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"[模型加载] 缺失的键（前10个）: {missing_keys[:10]}")
    if unexpected_keys:
        logger.info(f"[模型加载] 意外的键（前10个）: {unexpected_keys[:10]}")
    
    # 移动到设备
    backbone.to(device)
    backbone.eval()
    
    return backbone
