import torch
import torch.nn as nn
import torch.nn.functional as F
from motionVectorDeinterlacing.models.registry import COMPONENT_REGISTRY
from .blocks import ResidualBlock, build_activation
from motionVectorDeinterlacing.utils.ops import mv_warp # 1. 修正导入

# --- 0. 极简零初始化版 (Vanilla) ---
@COMPONENT_REGISTRY.register('VanillaMVRefiner')
class VanillaMVRefiner(nn.Module):
    def __init__(self, in_channels=2, mid_channels=32, act_cfg="ReLU"):
        super().__init__()
        # 工业化打包，极简的 ReLU 激活
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            build_activation(act_cfg), 
            nn.Conv2d(mid_channels, 2, 3, 1, 1)
        )
        
        # 零初始化
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, mv):
        # 极其清爽的残差相加
        return mv + self.body(mv)

# --- 1. 基础版：轻量级残差修正 ---
@COMPONENT_REGISTRY.register('BasicMVRefiner')
class BasicMVRefiner(nn.Module):
    def __init__(self, in_channels=2, feat_channels=64, mid_channels=32, act_cfg="PReLU"):
        super().__init__()
        # 工业化改造：打包成 body
        self.body = nn.Sequential(
            nn.Conv2d(in_channels + feat_channels, mid_channels, 3, 1, 1),
            build_activation(act_cfg),
            nn.Conv2d(mid_channels, 2, 3, 1, 1)
        )

        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, mv, feat):
        cat_in = torch.cat([mv, feat], dim=1)
        return mv + self.body(cat_in)

# --- 2. 凸上采样版 (Convex Upsampling)：针对 16x16 宏块的 1/4 优化版 ---
@COMPONENT_REGISTRY.register('ConvexUpsamplingRefiner')
class ConvexUpsamplingRefiner(nn.Module):
    def __init__(self, feat_channels=64, up_factor=4, act_cfg="PReLU"):
        super().__init__()
        self.up_factor = up_factor
        self.kernel_size = 3
        # 预测 4x4 区域内每个像素的 9 个权重
        out_channels = (up_factor ** 2) * (self.kernel_size ** 2)
        
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(feat_channels, 64, 3, 1, 1),
            build_activation(act_cfg),
            nn.Conv2d(64, out_channels, 1)
        )

        # 在 ConvexUpsamplingRefiner 的 __init__ 最后加上：
        nn.init.zeros_(self.mask_predictor[-1].weight)
        nn.init.zeros_(self.mask_predictor[-1].bias)

    def forward(self, mv, feat):
        """
        mv: (B, 2, H, W) 
        feat: (B, C, H/4, W/4) 
        """
        B, C, H, W = mv.shape
        H_lr, W_lr = H // self.up_factor, W // self.up_factor

        lr_mv = F.avg_pool2d(mv, kernel_size=self.up_factor)
        
        mask = self.mask_predictor(feat)
        mask = mask.view(B, 1, self.up_factor**2, self.kernel_size**2, H_lr, W_lr)
        mask = F.softmax(mask, dim=3)
        
        mv_padded = F.pad(lr_mv, (1, 1, 1, 1), mode='replicate')
        mv_unfolded = F.unfold(mv_padded, kernel_size=3).view(B, 2, 1, 9, H_lr, W_lr)
        
        out = torch.sum(mask * mv_unfolded, dim=3) # (B, 2, 16, H_lr, W_lr)
        out = out.view(B, 2, self.up_factor, self.up_factor, H_lr, W_lr)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
        return out.view(B, 2, H, W)

# --- 3. 门控残差版 (Gated Refiner) ---
@COMPONENT_REGISTRY.register('GatedMVRefiner')
class GatedMVRefiner(nn.Module):
    def __init__(self, feat_channels=64, mid_channels=64, act_cfg="PReLU"):
        super().__init__()
        self.mv_res_conv = nn.Sequential(
            nn.Conv2d(feat_channels + 2, mid_channels, 3, 1, 1),
            build_activation(act_cfg),
            nn.Conv2d(mid_channels, 2, 3, 1, 1)
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(feat_channels + 2, mid_channels, 3, 1, 1),
            build_activation(act_cfg),
            nn.Conv2d(mid_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.mv_res_conv[-1].weight)
        nn.init.zeros_(self.mv_res_conv[-1].bias)

    def forward(self, mv, feat):
        cat_in = torch.cat([mv, feat], dim=1)
        return mv + self.gate_conv(cat_in) * self.mv_res_conv(cat_in)

# --- 4. 图像引导版 (ImageGuided - 接口统一版) ---

@COMPONENT_REGISTRY.register('ImageGuidedMVRefiner')
class ImageGuidedMVRefiner(nn.Module):
    def __init__(self, mid=64, act_cfg="PReLU"):
        super().__init__()
     
        self.fusion = nn.Conv2d(2 + mid * 2, mid, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlock(mid, act_cfg=act_cfg) for _ in range(4)]
        )
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 3, 1, 1),
            build_activation(act_cfg),
            nn.Conv2d(mid, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(mid, 2, 3, 1, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, raw_mv, curr_feat, ref_feat):
        warped_ref_feat = mv_warp(ref_feat, raw_mv)
        diff_feat = curr_feat - warped_ref_feat
        occ_mask = self.mask_predictor(torch.cat([curr_feat, warped_ref_feat], dim=1)) # 输出 Sigmoid
        masked_diff = diff_feat * occ_mask
        inp = torch.cat([raw_mv, curr_feat, masked_diff], dim=1)
        
        x = self.body(self.fusion(inp))
        delta_mv = self.out_conv(x) * 0.1 
        return raw_mv + delta_mv

# --- 5. 瘦身实时版 (LiteImageGuided - 步长卷积/接口统一版) ---
@COMPONENT_REGISTRY.register('LiteImageGuidedMVRefiner')
class LiteImageGuidedMVRefiner(nn.Module):
    def __init__(self, mid=64, up_factor=4, act_cfg="PReLU"):
        super().__init__()
        self.up_factor = up_factor
 
        
        # 使用步长卷积进行下采样，相比 Pooling 更有学习能力
        self.downsample_mv = nn.Conv2d(2, 2, kernel_size=up_factor, stride=up_factor, padding=0)
        
        self.fusion = nn.Conv2d(2 + mid * 2, mid, 3, 1, 1)
        self.body = nn.Sequential(
            ResidualBlock(mid, act_cfg=act_cfg),
            ResidualBlock(mid, act_cfg=act_cfg)
        )
        # 接口统一：只输出 2 通道
        self.out_conv = nn.Conv2d(mid, 2, 3, 1, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, raw_mv, curr_feat, ref_feat):
        # --- 新增：对输入的高维特征进行下采样，对齐低分辨率空间 ---
        lr_curr_feat = F.avg_pool2d(curr_feat, self.up_factor)
        lr_ref_feat = F.avg_pool2d(ref_feat, self.up_factor)

        # 1. 步长卷积下采样 MV
        lr_mv = self.downsample_mv(raw_mv)
        
        # 2. 低分辨率对齐与误差计算 (全部使用 lr_ 级别的张量)
        warped_ref_feat = mv_warp(lr_ref_feat, lr_mv)
        diff_feat = lr_curr_feat - warped_ref_feat
        
        # 3. 融合与残差预测 (拼接 lr_ 级别的张量)
        inp = torch.cat([lr_mv, lr_curr_feat, diff_feat], dim=1)
        x = self.body(self.fusion(inp))
        
        # 4. 残差缩放并放大
        lr_delta = self.out_conv(x) * 0.1
        delta_mv = F.interpolate(lr_delta, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        
        return raw_mv + delta_mv
    
# --- 6. 终极实时版 (Lite + Convex Upsampling 融合版) ---
@COMPONENT_REGISTRY.register('LiteConvexMVRefiner')
class LiteConvexMVRefiner(nn.Module):
    def __init__(self, mid=64, up_factor=4, act_cfg="PReLU"):
        super().__init__()
        self.up_factor = up_factor
        self.kernel_size = 3
      
        
        # 1. 步长卷积下采样 MV (代替 pooling)
        self.downsample_mv = nn.Conv2d(2, 2, kernel_size=up_factor, stride=up_factor)
        
        # 2. 轻量级特征与误差融合 (1/4 分辨率下运行)
        self.fusion = nn.Conv2d(2 + mid * 2, mid, 3, 1, 1)
        self.body = nn.Sequential(
            ResidualBlock(mid, act_cfg=act_cfg),
            ResidualBlock(mid, act_cfg=act_cfg)
        )
        
        # 3. 计算低分辨率的 MV 修正量
        self.res_conv = nn.Conv2d(mid, 2, 3, 1, 1)
        nn.init.zeros_(self.res_conv.weight)
        nn.init.zeros_(self.res_conv.bias)

        # 4. 凸上采样权重预测器 (Convex Mask Predictor)
        # 注意：这里直接吃 body 提好的 mid 通道特征，避免重复计算！
        out_channels = (up_factor ** 2) * (self.kernel_size ** 2)
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(mid, 64, 3, 1, 1),
            build_activation(act_cfg),
            nn.Conv2d(64, out_channels, 1)
        )
        nn.init.zeros_(self.mask_predictor[-1].weight)
        nn.init.zeros_(self.mask_predictor[-1].bias)

    def forward(self, raw_mv, curr_feat, ref_feat):
        B, _, H, W = raw_mv.shape
        H_lr, W_lr = H // self.up_factor, W // self.up_factor
        lr_curr_feat = F.avg_pool2d(curr_feat, self.up_factor)
        lr_ref_feat = F.avg_pool2d(ref_feat, self.up_factor)

        # --- 第一阶段：低分辨率极速计算 (耗时极低) ---
        lr_mv = self.downsample_mv(raw_mv)
        
        # Warp & 计算误差图
        warped_ref_feat = mv_warp(lr_ref_feat, lr_mv)
        diff_feat = lr_curr_feat - warped_ref_feat
        
        # 提取融合特征 x
        inp = torch.cat([lr_mv, lr_curr_feat, diff_feat], dim=1)
        x = self.body(self.fusion(inp))
        
        # 算出低清修正量，并得到修正后的低清 MV
        lr_delta = self.res_conv(x) * 0.1
        refined_lr_mv = lr_mv + lr_delta

        # --- 第二阶段：凸组合上采样恢复至全尺寸 ---
        # 1. 用特征 x 预测 9 个邻居的放大权重
        mask = self.mask_predictor(x)
        mask = mask.view(B, 1, self.up_factor**2, self.kernel_size**2, H_lr, W_lr)
        mask = F.softmax(mask, dim=3)
        
        # 2. 提取 3x3 邻居
        mv_padded = F.pad(refined_lr_mv, (1, 1, 1, 1), mode='replicate')
        mv_unfolded = F.unfold(mv_padded, kernel_size=3).view(B, 2, 1, 9, H_lr, W_lr)
        
        # 3. 加权求和并用 Pixel Shuffle 思想铺平
        out = torch.sum(mask * mv_unfolded, dim=3)
        out = out.view(B, 2, self.up_factor, self.up_factor, H_lr, W_lr)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous()
        
        return out.view(B, 2, H, W)