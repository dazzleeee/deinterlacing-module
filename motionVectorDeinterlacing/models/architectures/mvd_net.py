import torch # tensor manipulation, equipment management, math operations 
import torch.nn as nn # neural network layers, activiation function, sequential containers
# import registry
import torch.nn.functional as F # functional API for activations, pooling, etc.

from motionVectorDeinterlacing.models.registry import (
    ARCH_REGISTRY, 
    COMPONENT_REGISTRY,
    build_from_cfg,
)
# 基础组件与算子 (合并重复的 ops 导入)
from motionVectorDeinterlacing.utils.ops import mv_warp, default_init_weights 
import motionVectorDeinterlacing.models.modules.gmc
from motionVectorDeinterlacing.utils.utils_flow import (
    quarterPixelMV_to_pixelMV, 
    rescale_mv_temporal
)
from config.config_schema import MVDNetConfig # for pydantic type checking
from motionVectorDeinterlacing.models import backbones, components    


@ARCH_REGISTRY.register('RealTimeMVDnet')
class RealTimeMVDnet(nn.Module):    
    def __init__(self, cfg: MVDNetConfig):
        super().__init__()
        self.mid = cfg.mid
        self.lookahead = cfg.lookahead 
        self.prop_order = cfg.propagation_order
        self.feat_extract = build_from_cfg(cfg.feature_extractor_cfg, COMPONENT_REGISTRY)
        self.mv_refiner = build_from_cfg(cfg.mv_refiner_cfg, COMPONENT_REGISTRY)
        self.gmc = build_from_cfg(cfg.gmc_cfg, COMPONENT_REGISTRY) # 全局运动补偿模块

        # 双向传播模块
        backward_blocks = [
            build_from_cfg(cfg.residual_block_cfg, COMPONENT_REGISTRY)
            for _ in range(cfg.num_blocks)
        ]
        self.backward_resblocks = nn.Sequential(*backward_blocks)

        forward_blocks = [
            build_from_cfg(cfg.residual_block_cfg, COMPONENT_REGISTRY)
            for _ in range(cfg.num_blocks)
        ]
        self.forward_resblocks = nn.Sequential(*forward_blocks)
        
        # 融合与重建

        self.foward_backward_fusion = build_from_cfg(
            cfg.foward_backward_fusion_cfg, COMPONENT_REGISTRY,
        ) #fusion of forward and backward features
        
        self.h_prop_current_feat_fusion = build_from_cfg(
            cfg.h_prop_current_feat_fusion_cfg, COMPONENT_REGISTRY,
        ) #fusion of current frame feature and h_prop 
        self.first_2nd_order_fusion = build_from_cfg(
            cfg.first_2nd_order_fusion_cfg, COMPONENT_REGISTRY,
        ) #fusion of 1-order and 2-order motion features
        

        self.refinement = nn.Sequential(
            nn.Conv2d(self.mid, self.mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )
        self.deint_up = build_from_cfg(cfg.deint_up_cfg, COMPONENT_REGISTRY)
        self.last_conv = nn.Conv2d(self.mid, 3, 3, 1, 1)

        # 核心护城河：全局前向隐状态缓存 (仅在推理模式下有效)
        self.register_buffer('h_fwd_cache', None)

        self.init_weights()

    def init_weights(self):
        default_init_weights(self, scale=1.0) 
        # 【第二轮特殊定制】：遍历全网，如果哪个子模块有专属要求，让它自己覆盖！
        # self.modules() 会递归查找所有的子组件、孙子组件
        for name, module in self.modules():
            # 动态检查这个组件有没有定义特殊的初始化方法
            if hasattr(module, '_init_special_weights'):
                module._init_special_weights()

    def reset_state(self):
        """流式处理开始前，或切换视频流通道时，必须调用此函数清空历史记忆"""
        # 缓存结构: (h_tm1, h_tm2, feat_tm1, feat_tm2)
        self.h_fwd_cache = None

    def _refine_mv_with_gmc(self, raw_pixel_mv, feat_target, feat_ref):
        """
        统一处理 MV 的 解耦(GMC) -> 局部精调(Refiner) -> 重组 流程
        """
        B, C, H, W = feat_target.shape
        
        # 1. 压缩回宏块级别给 CPU 算 RANSAC，极大提速
        block_size = 16 
        raw_mv_small = F.avg_pool2d(raw_pixel_mv, kernel_size=block_size, stride=block_size)
        
        # 2. 动态判断插值模式 (基于当前实例化的 refiner 类型)
        # 如果名字里带有 'Convex'，说明是阵营 A，用 nearest 保留马赛克边缘
        refiner_name = self.mv_refiner.__class__.__name__
        interp_mode = 'nearest' if 'Convex' in refiner_name else 'bilinear'
        
        # 3. GMC 解耦：计算局部物体运动和全局背景光流
        obj_motion, global_flow = self.gmc(
            raw_mv_small, 
            H, 
            W, 
            interpolation_mode=interp_mode
        )
        
        # 4. Refiner 精调 (你的 refiner 接口完全统一，直接调即可)
        # 此时 obj_motion 是剥离了背景运动的 [B, 2, H, W] 张量
        if refiner_name in ['VanillaMVRefiner']:
            refined_obj_motion = self.mv_refiner(obj_motion)
        elif refiner_name in ['BasicMVRefiner', 'GatedMVRefiner', 'ConvexUpsamplingRefiner']:
            refined_obj_motion = self.mv_refiner(obj_motion, feat_target)
        else:
            # 适配 ImageGuided, LiteImageGuided, LiteConvex
            refined_obj_motion = self.mv_refiner(obj_motion, feat_target, feat_ref)
        
        # 5. 终极运动：将平滑的全局背景 与 精调后的局部细节 重新合并
        final_mv = global_flow + refined_obj_motion
        
        return final_mv

    def forward(self, imgs, mv_fwd, field_ids):
        """
        根据 self.training 自动切换 BPTT 展开 或 单步状态机推流
        imgs:      [B, T, C, H, W]
        flows_fwd: [B, T, 2, H, W]  # H.264 提取的前向参考 MV (本质是后向映射)
        field_ids: [B, T]           # 顶/底场标识符
        """
        B, T, C, H, W = imgs.shape
        
        flags = field_ids.view(B, T, 1, 1, 1).expand(B, T, 1, H, W).float() 
        inp = torch.cat([imgs, flags], dim=2)
        feats = self.feat_extract(inp.view(-1, C + 1, H, W)).view(B, T, self.mid, H, W)

        # =====================================================================
        # 模式 A: 训练模式 (BPTT 展开计算 Loss) 
        # =====================================================================
        if self.training:
            bwd_features = [None] * T
            refined_mvs = [None] * T  
            
            # --- 1. 局部反向传播 ---
            for t in range(T):
                has_t1 = (t + 1 < T)
                has_t2 = (t + 2 < T)

                if not has_t1:
                    bwd_features[t] = feats[:, t]
                    continue
                
                if has_t2:
                    # ✅ 物理偏移修改 1: 传入 t+2 和 t+1 的极性
                    raw_mv_t1_ref_t2 = rescale_mv_temporal(
                        quarterPixelMV_to_pixelMV(-mv_fwd[:, t+2]),
                        src_oe=field_ids[:, t+2],
                        tgt_oe=field_ids[:, t+1]
                    )
                    mv_t1_ref_t2 = self._refine_mv_with_gmc(raw_mv_t1_ref_t2, feats[:, t+1], feats[:, t+2])
                    t1_warped_from_t2 = mv_warp(feats[:, t+2], mv_t1_ref_t2)
                    hidden_state_t1 = self.h_prop_current_feat_fusion(t1_warped_from_t2, feats[:, t+1])
                    hidden_state_t1 = self.backward_resblocks(hidden_state_t1) 
                else:
                    hidden_state_t1 = feats[:, t+1]
                
                # ✅ 物理偏移修改 2: 传入 t+1 和 t 的极性
                raw_mv_t_ref_t1 = rescale_mv_temporal(
                    quarterPixelMV_to_pixelMV(-mv_fwd[:, t+1]),
                    src_oe=field_ids[:, t+1],
                    tgt_oe=field_ids[:, t]
                )
                mv_t_ref_t1 = self._refine_mv_with_gmc(raw_mv_t_ref_t1, feats[:, t], feats[:, t+1])
                t_order1_warped_bwd = mv_warp(hidden_state_t1, mv_t_ref_t1)
                
                if has_t2:
                    # 二阶跳跃不需要 rescale 缩放时间，因此也不需要垂直修正
                    raw_mv_t_ref_t2 = quarterPixelMV_to_pixelMV(-mv_fwd[:, t+2])
                    mv_t_ref_t2 = self._refine_mv_with_gmc(raw_mv_t_ref_t2, feats[:, t], feats[:, t+2])
                    t_order2_warped_bwd = mv_warp(feats[:, t+2], mv_t_ref_t2)
                else:
                    t_order2_warped_bwd = torch.zeros_like(t_order1_warped_bwd) 

                fusion_order1and2_bwd = self.first_2nd_order_fusion(t_order1_warped_bwd, t_order2_warped_bwd)
                h_bwd = self.h_prop_current_feat_fusion(fusion_order1and2_bwd, feats[:, t])
                h_bwd_in = torch.cat([h_bwd, feats[:, t]], dim=1)
                bwd_features[t] = self.backward_resblocks(h_bwd_in)

            # --- 2. 前向传播与融合 ---
            fwd_features = [None] * T
            outs = [None] * T 
            
            for t in range(T):
                has_tm1 = (t - 1 >= 0)
                has_tm2 = (t - 2 >= 0)
                
                if has_tm1:
                    # ✅ 物理偏移修改 3: 传入 t-1 和 t 的极性
                    raw_mv_t_ref_tm1 = rescale_mv_temporal(
                        quarterPixelMV_to_pixelMV(mv_fwd[:, t]),
                        src_oe=field_ids[:, t-1],
                        tgt_oe=field_ids[:, t]
                    ) 
                    mv_t_ref_tm1 = self._refine_mv_with_gmc(raw_mv_t_ref_tm1, feats[:, t], feats[:, t-1])
                    t_order1_warped_fwd = mv_warp(fwd_features[t-1], mv_t_ref_tm1)
                    
                    refined_mvs[t] = mv_t_ref_tm1
                else:
                    t_order1_warped_fwd = torch.zeros_like(feats[:, t])
                    refined_mvs[t] = torch.zeros(B, 2, H, W, device=imgs.device)
                    
                if has_tm2:
                    raw_mv_t_ref_tm2 = quarterPixelMV_to_pixelMV(mv_fwd[:, t]) 
                    mv_t_ref_tm2 = self._refine_mv_with_gmc(raw_mv_t_ref_tm2, feats[:, t], feats[:, t-2])
                    t_order2_warped_fwd = mv_warp(fwd_features[t-2], mv_t_ref_tm2)
                else:
                    t_order2_warped_fwd = torch.zeros_like(feats[:, t])

                if has_tm1 or has_tm2:
                    fusion_order1and2_fwd = self.first_2nd_order_fusion(t_order1_warped_fwd, t_order2_warped_fwd)
                    h_fwd = self.h_prop_current_feat_fusion(fusion_order1and2_fwd, feats[:, t])
                else:
                    h_fwd = feats[:, t]

                bidirect_feat = torch.cat([h_fwd, bwd_features[t], feats[:, t]], dim=1)
                h_fwd = self.forward_resblocks(bidirect_feat)
                fwd_features[t] = h_fwd

            # --- 3. 晚期融合与重建 ---
            for t in range(T):
                final_feat = fwd_features[t]           
                current_o_e = field_ids[:, t] if field_ids is not None else t % 2
                up_feat = self.deint_up(final_feat, o_e=current_o_e)
                hr_feat = self.refinement(up_feat)
                out_residual = self.last_conv(hr_feat)
                curr_field = imgs[:, t] 
                base_frame = F.interpolate(curr_field, scale_factor=(2, 1), mode='bilinear', align_corners=False)
                
                outs[t] = out_residual + base_frame

            sr_out = torch.stack(outs, dim=1) 
            all_mvs = torch.stack(refined_mvs, dim=1)
            
            return {
                'sr': sr_out,
                'flows': all_mvs,   
                'lr_imgs': imgs     
            }

        # =====================================================================
        # 模式 B: 推理模式 (实时推流状态机)
        # =====================================================================
        else:
            with torch.no_grad():
                bwd_features = [None] * T
                fwd_features = [None] * T
                outs = [None] * T
                
                if self.h_fwd_cache is None:
                    h_tm1, h_tm2, feat_tm1, feat_tm2 = None, None, None, None
                else:
                    h_tm1, h_tm2, feat_tm1, feat_tm2 = self.h_fwd_cache

                # --- 1. 推理态反向 Lookahead ---
                for t in range(T):
                    has_t1 = (t + 1 < T)
                    has_t2 = (t + 2 < T)
                    
                    if not has_t1:
                        bwd_features[t] = feats[:, t]
                        continue
                        
                    if has_t2:
                        # ✅ 推理态物理偏移修改 1: 传入 t+2 和 t+1 的极性
                        raw_mv_t1_ref_t2 = rescale_mv_temporal(
                            quarterPixelMV_to_pixelMV(-mv_fwd[:, t+2]),
                            src_oe=field_ids[:, t+2] if field_ids is not None else (t+2)%2,
                            tgt_oe=field_ids[:, t+1] if field_ids is not None else (t+1)%2
                        )
                        mv_t1_ref_t2 = self._refine_mv_with_gmc(raw_mv_t1_ref_t2, feats[:, t+1], feats[:, t+2])
                        t1_warped_from_t2 = mv_warp(feats[:, t+2], mv_t1_ref_t2)
                        hidden_state_t1 = self.h_prop_current_feat_fusion(t1_warped_from_t2, feats[:, t+1])
                        hidden_state_t1 = self.backward_resblocks(hidden_state_t1) 
                    else:
                        hidden_state_t1 = feats[:, t+1]
                    
                    # ✅ 推理态物理偏移修改 2: 传入 t+1 和 t 的极性
                    raw_mv_t_ref_t1 = rescale_mv_temporal(
                        quarterPixelMV_to_pixelMV(-mv_fwd[:, t+1]),
                        src_oe=field_ids[:, t+1] if field_ids is not None else (t+1)%2,
                        tgt_oe=field_ids[:, t] if field_ids is not None else t%2
                    )
                    mv_t_ref_t1 = self._refine_mv_with_gmc(raw_mv_t_ref_t1, feats[:, t], feats[:, t+1])
                    t_order1_warped_bwd = mv_warp(hidden_state_t1, mv_t_ref_t1)
                    
                    if has_t2:
                        raw_mv_t_ref_t2 = quarterPixelMV_to_pixelMV(-mv_fwd[:, t+2])
                        mv_t_ref_t2 = self._refine_mv_with_gmc(raw_mv_t_ref_t2, feats[:, t], feats[:, t+2])
                        t_order2_warped_bwd = mv_warp(feats[:, t+2], mv_t_ref_t2)
                    else:
                        t_order2_warped_bwd = torch.zeros_like(t_order1_warped_bwd) 

                    fusion_order1and2_bwd = self.first_2nd_order_fusion(t_order1_warped_bwd, t_order2_warped_bwd)
                    h_bwd = self.h_prop_current_feat_fusion(fusion_order1and2_bwd, feats[:, t])
                    h_bwd_in = torch.cat([h_bwd, feats[:, t]], dim=1)
                    bwd_features[t] = self.backward_resblocks(h_bwd_in)

                # --- 2. 推理态前向传播与缓存更新 ---
                for t in range(T):
                    ref_feat_1 = feats[:, t-1] if t > 0 else feat_tm1
                    ref_feat_2 = feats[:, t-2] if t > 1 else (feat_tm1 if t == 1 else feat_tm2)
                    
                    if h_tm1 is not None and ref_feat_1 is not None:
                        # ✅ 推理态物理偏移修改 3: 结合历史状态的安全极性推断
                        current_oe = field_ids[:, t] if field_ids is not None else t%2
                        prev_oe = field_ids[:, t-1] if (t-1 >= 0 and field_ids is not None) else (t-1)%2
                        
                        raw_mv_t_ref_tm1 = rescale_mv_temporal(
                            quarterPixelMV_to_pixelMV(mv_fwd[:, t]),
                            src_oe=prev_oe,
                            tgt_oe=current_oe
                        ) 
                        mv_t_ref_tm1 = self._refine_mv_with_gmc(raw_mv_t_ref_tm1, feats[:, t], ref_feat_1)
                        t_order1_warped_fwd = mv_warp(h_tm1, mv_t_ref_tm1)
                    else:
                        t_order1_warped_fwd = torch.zeros_like(feats[:, t])
                        
                    if h_tm2 is not None and ref_feat_2 is not None:
                        raw_mv_t_ref_tm2 = quarterPixelMV_to_pixelMV(mv_fwd[:, t]) 
                        mv_t_ref_tm2 = self._refine_mv_with_gmc(raw_mv_t_ref_tm2, feats[:, t], ref_feat_2)
                        t_order2_warped_fwd = mv_warp(h_tm2, mv_t_ref_tm2)
                    else:
                        t_order2_warped_fwd = torch.zeros_like(feats[:, t])

                    if h_tm1 is not None or h_tm2 is not None:
                        fusion_order1and2_fwd = self.first_2nd_order_fusion(t_order1_warped_fwd, t_order2_warped_fwd)
                        h_fwd = self.h_prop_current_feat_fusion(fusion_order1and2_fwd, feats[:, t])
                    else:
                        h_fwd = feats[:, t]

                    bidirect_feat = torch.cat([h_fwd, bwd_features[t], feats[:, t]], dim=1)
                    h_fwd = self.forward_resblocks(bidirect_feat)
                    fwd_features[t] = h_fwd

                    h_tm2 = h_tm1
                    h_tm1 = h_fwd.detach()
                    feat_tm2 = feat_tm1
                    feat_tm1 = feats[:, t].detach()

                self.h_fwd_cache = (h_tm1, h_tm2, feat_tm1, feat_tm2)

                # --- 3. 推理态重建 ---
                for t in range(T):
                    final_feat = fwd_features[t]           
                    current_o_e = field_ids[:, t] if field_ids is not None else t % 2
                    up_feat = self.deint_up(final_feat, o_e=current_o_e)
                    hr_feat = self.refinement(up_feat)
                    out_residual = self.last_conv(hr_feat)
                    curr_field = imgs[:, t] 
                    base_frame = F.interpolate(curr_field, scale_factor=(2, 1), mode='bilinear', align_corners=False)
                    
                    outs[t] = out_residual + base_frame
                    
                return torch.stack(outs, dim=1)