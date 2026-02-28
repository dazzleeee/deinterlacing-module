import torch
import torch.nn as nn
from motionVectorDeinterlacing.models.registry import COMPONENT_REGISTRY
from motionVectorDeinterlacing.models.components.activation import build_activation  

#There are three stage which need fusion:
# 1. fusion of 1-order and 2-order motion features (MultiOrderFusion)
# 2. fusion of current frame feature and warped reference feature (MotionAdaptiveFusion)
# 3. fusion of forward and backward propagated features (BasicConcatFusion)
# ==========================================
# 1. 高级版：动静自适应融合 (用于时域传播) 
# /fusion of current frame feature and h_prop
# ==========================================
@COMPONENT_REGISTRY.register('MotionAdaptiveFusion')
class MotionAdaptiveFusion(nn.Module):
    """
    带“大脑”的动静自适应融合模块 (Soft Bob/Weave)
    """
    def __init__(self, c_in=64, act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super().__init__()
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(c_in * 2, c_in, 3, 1, 1),
            build_activation(act_cfg), 
            nn.Conv2d(c_in, 1, 3, 1, 1),
            nn.Sigmoid() 
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c_in * 2, c_in, 3, 1, 1),
            build_activation(act_cfg)  
        )

    def _init_special_weights(self):
        """专属覆盖逻辑：只修改最后一层的输出分布，保护 Sigmoid/Softmax"""
        nn.init.normal_(self.mask_predictor[-2].weight, mean=0, std=0.01)
        nn.init.constant_(self.mask_predictor[-2].bias, 0)

    def forward(self, curr_feat, h_prop):
        cat_feat = torch.cat([curr_feat, h_prop], dim=1)
        mask = self.mask_predictor(cat_feat)
        filtered_h_prop = h_prop * mask
        out_feat = self.fusion_conv(torch.cat([curr_feat, filtered_h_prop], dim=1))
        return out_feat, mask 


# ==========================================
# 2. 经典版：基础拼接融合 (用于双向末端合并)
# /fusion of forward and backward propagated features/h_prop and current feature fusion
# ==========================================
@COMPONENT_REGISTRY.register('BasicConcatFusion')
class BasicConcatFusion(nn.Module):
    """
    老代码中的 1x1 拼接融合，简单粗暴且有效
    """
    def __init__(self, c_in=64):
        super().__init__()
        self.fusion = nn.Conv2d(c_in * 2, c_in, 1, 1)

    def forward(self, fwd_feat, bwd_feat):
        return self.fusion(torch.cat([fwd_feat, bwd_feat], dim=1))


# ==========================================
# 3. SoftGate 动静融合 (用于时域传播)/fusion of current frame feature and h_prop
# ==========================================
@COMPONENT_REGISTRY.register('SoftGateAdaptiveFusion')
class SoftGateAdaptiveFusion(nn.Module):
    """
    轻量级动静自适应融合模块 (基于简化版 ConvGRU 更新门机制)
    """
    def __init__(self, mid=64):
        super().__init__()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 3, 1, 1),
            nn.Sigmoid() 
        )

    def _init_special_weights(self):
        """专属覆盖逻辑：只修改最后一层的输出分布，保护 Sigmoid/Softmax"""
        nn.init.normal_(self.mask_conv[-2].weight, mean=0, std=0.01)
        nn.init.constant_(self.mask_conv[-2].bias, 0)

    def forward(self, curr_feat, h_prop):
        cat_feat = torch.cat([curr_feat, h_prop], dim=1)
        mask = self.mask_conv(cat_feat)
        fused_feat = mask * h_prop + (1.0 - mask) * curr_feat
        return fused_feat, mask
    

# ==========================================
# 4. Multi-Order Fusion (用于一阶和二阶传播融合) /fusion of 1-order and 2-order motion features
# ==========================================

@COMPONENT_REGISTRY.register('DeinterlacingMultiOrderFusion')
class DeinterlacingMultiOrderFusion(nn.Module):
    def __init__(self, mid_channels=64): 
        super().__init__()
        # 接收三个特征：curr_feat(标准答案), h1_warped(一阶草稿), h2_warped(二阶草稿)
        self.attention_net = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            # 输出 3 个权重通道！
            nn.Conv2d(mid_channels, 3, 3, 1, 1) 
        )
    
    def _init_special_weights(self):
        """专属覆盖逻辑：只修改最后一层的输出分布，保护 Sigmoid/Softmax"""
        nn.init.normal_(self.attention_net[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.attention_net[-1].bias, 0)

    def forward(self, curr_feat, h1_warped, h2_warped):
        # 1. 把标准答案和两份草稿摆在一起对质
        cat_feat = torch.cat([curr_feat, h1_warped, h2_warped], dim=1)
        
        # 2. 算出 3 份权重
        weight_logits = self.attention_net(cat_feat)
        weights = torch.softmax(weight_logits, dim=1)
        
        w_curr = weights[:, 0:1, :, :] 
        w_h1   = weights[:, 1:2, :, :] 
        w_h2   = weights[:, 2:3, :, :] 
        
        # 3. 终极融合：带“止损机制”的 Alpha Blending
        h_prop = (curr_feat * w_curr) + (h1_warped * w_h1) + (h2_warped * w_h2)
        
        return h_prop, w_curr, w_h1, w_h2


# ==========================================
# 5. Baseline: 简单平均一阶二阶融合 (消融实验用) /fusion of 1-order and 2-order motion features
# ==========================================
@COMPONENT_REGISTRY.register('SimpleAverageFusion')
class SimpleAverageFusion(nn.Module):
    """
    最原始的做法：无视特征质量，强行把一阶和二阶相加后除以 2
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, curr_feat, h1_warped, h2_warped):
        h_prop = (h1_warped + h2_warped) * 0.5
        # 返回虚拟权重对齐接口
        B, _, H, W = h1_warped.shape
        w1 = torch.full((B, 1, H, W), 0.5, device=h1_warped.device)
        w2 = torch.full((B, 1, H, W), 0.5, device=h1_warped.device)
        return h_prop, w1, w2


