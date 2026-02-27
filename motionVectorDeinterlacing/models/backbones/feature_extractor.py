import torch
import torch.nn as nn
from motionVectorDeinterlacing.models.registry import COMPONENT_REGISTRY
from motionVectorDeinterlacing.models.components.blocks import ChannelAttentionBlock
# ==========================================
# 方案 A: 极简卷积版 (Vanilla Convolution)
# 特点：计算量极小，适合低性能设备
# ==========================================
@COMPONENT_REGISTRY.register('SimpleFeatureExtractor')
class SimpleFeatureExtractor(nn.Module):
    # 既然写死了，__init__ 里的 act_cfg 参数也可以删掉了，保持接口清爽
    def __init__(self, in_channels=3, nf=32):
        super().__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, 1),
            # 直接实例化类，斜率设为底层视觉常用的 0.1
            # inplace=True 可以直接覆盖原内存，这在你的实时流处理中能省下不少显存
            nn.LeakyReLU(0.1, inplace=True), 
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.body(x)

# ==========================================
# 方案 B: 5个残差块版 (Deep Residual RepConv)
# 特点：感受野大，特征提取能力强，支持重参数化白嫖性能
# ==========================================
@COMPONENT_REGISTRY.register('ResidualFeatureExtractor')
class ResidualFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, nf=32, num_blocks=5, act_cfg="PReLU", deploy=False):
        super().__init__()
        # 1. 第一层将 RGB 转为特征通道
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        
        # 2. 串联 5 个带有通道注意力的残差块
        # 内部使用 RepConv，训练时多分支，部署时融合成单层
        self.body = nn.Sequential(*[
            ChannelAttentionBlock(nf=nf, act_cfg=act_cfg, res_scale=0.1, deploy=deploy)
            for _ in range(num_blocks)
        ])
        
        self.conv_last = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        res = self.body(feat)
        out = self.conv_last(res)
        return out + feat # 全局残差连接，保证深层网络信号稳定

