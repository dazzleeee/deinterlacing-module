import torch
import torch.nn as nn
from typing import Dict, Optional

from ..registry import ARCH_REGISTRY, COMPONENT_REGISTRY, build_from_cfg
from ...utils.ops import MVWarp, DeintShufflePack, ResidualBlock
from ...utils.utils_flow import quarterPixelMV_to_pixelMV, rescale_mv_temporal


@ARCH_REGISTRY.register()
class mvd_net(nn.Module):
    def __init__(
        self, 
        mid_channels: int, 
        num_blocks: int,
        # control second-order flow usage
        use_second_order: bool,
        # control MV preprocessing of 1-order MV
        mv_correct_cfg: dict,
        # control MV refiner
        mv_refiner_cfg: Optional[Dict] = {'type': 'MVRefiner'},
        # Control switch for enabling/disabling motion-static adaptive fusion
        fusion_cfg: Optional[Dict] = {'type': 'AdaptiveFusion'}, 
   
    ):
   
        super().__init__()
        
        # 保存开关
        self.mid = mid
        self.mv_mode = mv_mode
        self.correct_offset = correct_offset
        self.use_second_order = use_second_order
        self.use_adaptive_fusion = use_adaptive_fusion

        # 1. 零件工厂：不再写死，而是从注册表里“点菜”
        self.mvwarp = MVWarp()
        self.mv_refiner = build_from_cfg(refiner_cfg, COMPONENT_REGISTRY)

        # 2. 基础结构
        # 浅层特征提取
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid, mid, 3, 1, 1), nn.LeakyReLU(0.1, True)
        )
        
        # 前后向传播块（解包 * 生成 15 个 Block）
        self.backward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        self.forward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        
        # 融合与细化
        self.fusion = nn.Conv2d(mid * 2, mid, 1, 1)
        self.refinement = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1), nn.LeakyReLU(0.1, True)
        )
        
        # 去隔行上采样层
        self.deint_up = DeintShufflePack(mid, mid, scale_factor=2)
        self.last_conv = nn.Conv2d(mid, 3, 3, 1, 1)
    

    def compute_flow(self, raw_mv, curr_feat, ref_feat, src_oe, tgt_oe):
        # 1. 物理修正 (归一化/缩放/0.5像素偏移)
        mv_init = self.pre_process_mv(raw_mv, src_oe, tgt_oe)
        
        # 2. 调用零件 (根据配置可能是普通版或 Adaptive 版)
        # 如果是 Adaptive 版，会多返回一个 motion_mask
        res = self.mv_refiner(mv_init, curr_feat, ref_feat)
        refined_mv, conf, motion_mask = res if len(res)==3 else (res[0], res[1], None)

        # 3. 【核心开关】动静分离融合
        warped_feat = self.mvwarp(ref_feat, refined_mv)
        
        if self.use_adaptive_fusion and motion_mask is not None:
            # 静止区域特征：直接取平均（类似 Weave 模式，保留最高清晰度）
            # 动态区域特征：使用对齐后的特征
            # 融合公式: $Feat_{final} = Mask \cdot Feat_{warped} + (1-Mask) \cdot \frac{Curr + Warped}{2}$
            static_feat = (curr_feat + warped_feat) * 0.5
            final_feat = motion_mask * warped_feat + (1.0 - motion_mask) * static_feat
        else:
            final_feat = warped_feat
            
        return refined_mv, final_feat, conf

    def forward_sequence(self, imgs, mv_fwd, field_ids):
        """训练模式：一次处理整段 [B, T, C, H, W]"""
        B, T, C, H, W = imgs.shape
        # 特征提取
        feats = self.feat_extract(imgs.view(-1, C, H, W)).view(B, T, -1, H, W)
        
        # --- 后向传播 ---
        bwd_features = [None] * T
        h_bwd, h_bwd_old = torch.zeros_like(feats[:, 0]), torch.zeros_like(feats[:, 0])
        for t in range(T - 1, -1, -1):
            h_prop = torch.zeros_like(h_bwd)
            if t < T - 1:
                h1_prop = self.compute_flow(-mv_fwd[:, t+1], feats[:, t], feats[:, t+1], h_bwd, field_ids[:, t+1], field_ids[:, t], scale=0.5)
                if self.use_second_order and (t < T - 2):
                    # 二阶：直接使用直达地图 (t -> t+2)
                    h2_prop = self.compute_flow(-mv_fwd[:, t+2], feats[:, t], feats[:, t+2], h_bwd_old, field_ids[:, t+2], field_ids[:, t], scale=1.0)
                    h_prop = (h1_prop + h2_prop) * 0.5  
                else:
                    h_prop = h1_prop
            h_bwd_old, h_bwd = h_bwd.clone(), self.backward_resblocks(h_prop + feats[:, t])
            bwd_features[t] = h_bwd

        # --- 前向传播与输出 ---
        outs = []
        h_fwd, h_fwd_old = torch.zeros_like(feats[:, 0]), torch.zeros_like(feats[:, 0])
        for t in range(T):
            h_prop = torch.zeros_like(h_fwd)
            if t > 0:
                h1_prop = self.compute_flow(mv_fwd[:, t], feats[:, t], feats[:, t-1], h_fwd, field_ids[:, t-1], field_ids[:, t], scale=0.5)
                if self.use_second_order and (t > 1):
                    # 二阶：直接使用直达地图 (t -> t-2)
                    h2_prop = self.compute_flow(mv_fwd[:, t], feats[:, t], feats[:, t-2], h_fwd_old, field_ids[:, t-2], field_ids[:, t], scale=1.0)
                    h_prop = (h1_prop + h2_prop) * 0.5
                else:
                    h_prop = h1_prop
            h_fwd_old, h_fwd = h_fwd.clone(), self.forward_resblocks(h_prop + feats[:, t])
            
            fused = self.fusion(torch.cat([h_fwd, bwd_features[t]], dim=1))
            up = self.deint_up(self.refinement(fused), o_e=field_ids[:, t])
            outs.append(self.last_conv(up))
        return torch.stack(outs, dim=1)

    def forward_recurrent(self, x_curr, mv_curr, fid_curr, fid_prev, h_prev):
        """实时模式：逐帧处理 [B, C, H, W]"""
        feat_curr = self.feat_extract(x_curr)
        flow = self.compute_flow(mv_curr, fid_prev, fid_curr)
        h_warped = self.mvwarp(h_prev, flow)
        h_curr = self.forward_resblocks(h_warped + feat_curr)
        up = self.deint_up(self.refinement(h_curr), o_e=fid_curr)
        return self.last_conv(up), h_curr

    def forward(self, *args, mode='sequence', **kwargs):
        """中转站，根据 mode 决定跑哪个逻辑"""
        if mode == 'sequence':
            return self.forward_sequence(*args, **kwargs)
        else:
            return self.forward_recurrent(*args, **kwargs)
    
    def forward_sliding_window(self, x_window, mv_window, fid_window, h_fwd_prev):
        """
        x_window: [B, window_size, C, H, W]  (例如 window_size=3 或 5)
        h_fwd_prev: [B, mid, H, W] (从历史中累积下来的前向记忆)
        """
        B, T, C, H, W = x_window.shape
        # 1. 提取窗口内所有帧的特征
        feats = self.feat_extract(x_window.view(-1, C, H, W)).view(B, T, -1, H, W)
        
        # 2. 局部后向传播 (只看窗口内的未来)
        # 我们只关心窗口第一帧 (t=0) 的后向特征
        h_bwd = torch.zeros_like(feats[:, 0])
        h_bwd_old = torch.zeros_like(feats[:, 0])
        
        # 倒着循环：从最后一帧到第一帧
        for t in range(T - 1, -1, -1):
            h_prop = torch.zeros_like(h_bwd)
            if t < T - 1:
                flow1 = self.compute_flow(-mv_window[:, t+1], fid_window[:, t+1], fid_window[:, t])
                h1_warped = self.mvwarp(h_bwd, flow1)
                if self.use_second_order and (t < T - 2):
                    # 1. 一阶流：t -> t-1 (利用直达地图的一半)
                    flow1 = self.compute_flow(mv_bwd[:, t] * 0.5, fid_prev, fid_curr) 
                    h1_warped = self.mvwarp(h_bwd, flow1)
                    
                    # 2. 二阶流：t -> t-2 (直接利用这张“直达地图”)
                    # 注意：这里不再需要 warp(flow2, flow1) 的链式合成！
                    flow2_direct = self.compute_flow(mv_bwd[:, t], fid_prev2, fid_curr) # 这里的 scale 为 1.0
                    h2_warped = self.mvwarp(h_bwd_old, flow2_direct)
                    
                    h_prop = (h1_warped + h2_warped) * 0.5
                else:
                    h_prop = h1_warped
            h_bwd_old, h_bwd = h_bwd.clone(), self.backward_resblocks(h_prop + feats[:, t])
        
        # 此时得到的 h_bwd 就是融合了窗口内未来信息的特征
        
        # 3. 前向累积 (只有窗口第一帧参与)
        # 将上一时刻的全局记忆 h_fwd_prev 对齐到当前帧
        flow_fwd = self.compute_flow(mv_window[:, 0], fid_window[:, 0], fid_window[:, 0]) # 简化示意
        h_fwd_warped = self.mvwarp(h_fwd_prev, flow_fwd)
        h_fwd_curr = self.forward_resblocks(h_fwd_warped + feats[:, 0])
        
        # 4. 融合输出
        fused = self.fusion(torch.cat([h_fwd_curr, h_bwd], dim=1))
        up = self.deint_up(self.refinement(fused), o_e=fid_window[:, 0])
        out = self.last_conv(up)
        
        return out, h_fwd_curr # 返回输出和更新后的全局记忆