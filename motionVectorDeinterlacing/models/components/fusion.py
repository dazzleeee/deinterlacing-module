import torch
import torch.nn as nn
from ..registry import COMPONENT_REGISTRY

@COMPONENT_REGISTRY.register()
class AdaptiveFusion(nn.Module):
    """
    动静自适应融合模块
    """
    def __init__(self, c_in=64):
        super().__init__()
        # 如果需要可学习参数，可以在这里定义
        # self.weight = nn.Parameter(...) 
        pass

    def forward(self, curr_feat, warped_feat, mask):
        """
        Args:
            curr_feat: 当前帧特征
            warped_feat: 对齐后的参考帧特征
            mask: 运动掩码 (1=动, 0=静)
        """
        # 静态区域策略：Weave 模式 (直接平均)
        static_feat = (curr_feat + warped_feat) * 0.5
        
        # 融合公式
        # mask 越接近 1 (动)，越信任 warped_feat (Refiner 对齐过的)
        # mask 越接近 0 (静)，越信任 static_feat (平均值)
        return mask * warped_feat + (1.0 - mask) * static_feat