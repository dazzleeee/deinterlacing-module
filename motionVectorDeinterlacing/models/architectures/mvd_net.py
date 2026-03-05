import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from motionVectorDeinterlacing.models.registry import (
    ARCH_REGISTRY, 
    COMPONENT_REGISTRY,
    build_from_cfg,
)
from motionVectorDeinterlacing.utils.ops import mv_warp, default_init_weights 
import motionVectorDeinterlacing.models.modules.gmc
from motionVectorDeinterlacing.utils.utils_flow import (
    quarterPixelMV_to_pixelMV, 
    rescale_mv_temporal
)
from config.config_schema import MVDNetConfig 
from motionVectorDeinterlacing.models import backbones, components    
from motionVectorDeinterlacing.models.components.fusion import MemoryEfficientTemporalFusion


@ARCH_REGISTRY.register('RealTimeMVDnet')
class RealTimeMVDnet(nn.Module):    
    def __init__(self, cfg: MVDNetConfig):
        super().__init__()
        self.mid = cfg.mid
        self.lookahead = cfg.lookahead 
        self.prop_order = cfg.propagation_order
        self.feat_extract = build_from_cfg(cfg.feature_extractor_cfg, COMPONENT_REGISTRY)
        
        def inject_dim(cfg_dict):
            d = dict(cfg_dict)
            t = d['type']
            if 'GMC' in t: d['in_channels'] = self.mid * 2
            elif 'MultiOrder' in t: d['mid_channels'] = self.mid
            elif 'SoftGate' in t or 'Lite' in t or 'ImageGuided' in t: d['mid'] = self.mid
            elif 'MotionAdaptive' in t or 'BasicConcat' in t: d['c_in'] = self.mid
            return d

        # 接下来，所有带有隐式维度的组件，都必须套上 inject_dim()
        self.mv_refiner = build_from_cfg(inject_dim(cfg.mv_refiner_cfg), COMPONENT_REGISTRY)
        self.gmc = build_from_cfg(inject_dim(cfg.gmc_cfg), COMPONENT_REGISTRY) 

        res_cfg = dict(cfg.residual_block_cfg)
        if 'nf' not in res_cfg:
            res_cfg['nf'] = self.mid

        # 双向传播残差块
        backward_blocks = [
            build_from_cfg(res_cfg, COMPONENT_REGISTRY)
            for _ in range(cfg.num_blocks)
        ]
        self.backward_resblocks = nn.Sequential(*backward_blocks)

        forward_blocks = [
            build_from_cfg(res_cfg, COMPONENT_REGISTRY)
            for _ in range(cfg.num_blocks)
        ]
        self.forward_resblocks = nn.Sequential(*forward_blocks)
        
        # 融合与重建
        self.foward_backward_fusion = build_from_cfg(inject_dim(cfg.foward_backward_fusion_cfg), COMPONENT_REGISTRY) 
        self.h_prop_current_feat_fusion = build_from_cfg(inject_dim(cfg.h_prop_current_feat_fusion_cfg), COMPONENT_REGISTRY)  
        self.first_2nd_order_fusion = build_from_cfg(inject_dim(cfg.first_2nd_order_fusion_cfg), COMPONENT_REGISTRY) 
        # ==========================================================
       
       
        
        # ==========================================================
        # ✅ 新增：用于兼容 Dense Connection 的 1x1 降维卷积
        # ==========================================================
        self.bwd_channel_reduce = nn.Conv2d(self.mid * 2, self.mid, kernel_size=1, stride=1)
        # 删掉 self.fwd_channel_reduce，换成下面这个：
       
        self.memory_efficient_fusion = MemoryEfficientTemporalFusion(mid_channels=self.mid)

        self.refinement = nn.Sequential(
            nn.Conv2d(self.mid, self.mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )
        
        # --- 自动注入 in_channels 和 out_channels ---
        deint_cfg = dict(cfg.deint_up_cfg)
        if 'in_channels' not in deint_cfg:
            deint_cfg['in_channels'] = self.mid
        if 'out_channels' not in deint_cfg:
            deint_cfg['out_channels'] = self.mid
            
        self.deint_up = build_from_cfg(deint_cfg, COMPONENT_REGISTRY)
        
        self.last_conv = nn.Conv2d(self.mid, 3, 3, 1, 1)

        # 核心护城河：全局前向隐状态缓存
        self.h_fwd_cache = None

        self.init_weights()

    def init_weights(self):
        default_init_weights(self, scale=1.0) 
        for module in self.modules():
            if hasattr(module, '_init_special_weights'):
                module._init_special_weights()

    def reset_state(self):
        self.h_fwd_cache = None
        self.stream_cache = None
        

    def _refine_mv_with_gmc(self, raw_pixel_mv, feat_target, feat_ref):
        B, C, H, W = feat_target.shape
        block_size = 16 
        raw_mv_small = F.avg_pool2d(raw_pixel_mv, kernel_size=block_size, stride=block_size)
        
        refiner_name = self.mv_refiner.__class__.__name__
        interp_mode = 'nearest' if 'Convex' in refiner_name else 'bilinear'
        
     
        
# ======== 唯一需要修改的地方：把特征传给 gmc ========
        obj_motion, global_flow = self.gmc(
            raw_mv_small, H, W, interpolation_mode=interp_mode,
            feat_curr=feat_target, feat_ref=feat_ref  # <--- 加了这一行！
        )

        if refiner_name in ['VanillaMVRefiner']:
            refined_obj_motion = self.mv_refiner(obj_motion)
        elif refiner_name in ['BasicMVRefiner', 'GatedMVRefiner', 'ConvexUpsamplingRefiner']:
            refined_obj_motion = self.mv_refiner(obj_motion, feat_target)
        else:
            refined_obj_motion = self.mv_refiner(obj_motion, feat_target, feat_ref)
        
        final_mv = global_flow + refined_obj_motion
        return final_mv
    
    @torch.no_grad()
    def forward_stream_step(self, imgs_win, mvs_win, fids_win):
        """
        真实的流式单步推理。每次严格接收一个长度为 3 的滑动窗口 [t, t+1, t+2]。
        输出第 t 帧的高清结果，并更新内部状态。
        """
        B, T, C, H, W = imgs_win.shape
        assert T == 3, "Stream step requires exactly a 3-frame window [t, t+1, t+2]"

        # 1. 提取窗口内三帧的特征
        flags = fids_win.view(B, T, 1, 1, 1).expand(B, T, 1, H, W).float()
        inp = torch.cat([imgs_win, flags], dim=2)
        feats = self.feat_extract(inp.view(-1, C + 1, H, W)).view(B, T, self.mid, H, W)

        feat_t, feat_t1, feat_t2 = feats[:, 0], feats[:, 1], feats[:, 2]
        mv_t, mv_t1, mv_t2 = mvs_win[:, 0], mvs_win[:, 1], mvs_win[:, 2]
        fid_t, fid_t1, fid_t2 = fids_win[:, 0], fids_win[:, 1], fids_win[:, 2]

        # ==========================================
        # 2. 计算当前帧 t 的后向传播特征 (利用 t+1 和 t+2)
        # ==========================================
        # 2.1 将 t+2 warp 到 t+1
        raw_mv_t1_ref_t2 = rescale_mv_temporal(
            quarterPixelMV_to_pixelMV(-mv_t2), src_oe=fid_t2, tgt_oe=fid_t1
        )
        mv_t1_ref_t2 = self._refine_mv_with_gmc(raw_mv_t1_ref_t2, feat_t1, feat_t2)
        t1_warped_from_t2 = mv_warp(feat_t2, mv_t1_ref_t2)
        hidden_state_t1, _ = self.h_prop_current_feat_fusion(t1_warped_from_t2, feat_t1)
        hidden_state_t1 = self.backward_resblocks(hidden_state_t1)

        # 2.2 将上面算出的隐状态 warp 到 t
        raw_mv_t_ref_t1 = rescale_mv_temporal(
            quarterPixelMV_to_pixelMV(-mv_t1), src_oe=fid_t1, tgt_oe=fid_t
        )
        mv_t_ref_t1 = self._refine_mv_with_gmc(raw_mv_t_ref_t1, feat_t, feat_t1)
        t_order1_warped_bwd = mv_warp(hidden_state_t1, mv_t_ref_t1)

        # 2.3 将 t+2 直接 warp 到 t (二阶融合)
        raw_mv_t_ref_t2 = quarterPixelMV_to_pixelMV(-mv_t2)
        mv_t_ref_t2 = self._refine_mv_with_gmc(raw_mv_t_ref_t2, feat_t, feat_t2)
        t_order2_warped_bwd = mv_warp(feat_t2, mv_t_ref_t2)

        fusion_order1and2_bwd, *_ = self.first_2nd_order_fusion(feat_t, t_order1_warped_bwd, t_order2_warped_bwd)
        h_bwd, _ = self.h_prop_current_feat_fusion(fusion_order1and2_bwd, feat_t)
        
        h_bwd_in = torch.cat([h_bwd, feat_t], dim=1)
        h_bwd_in = self.bwd_channel_reduce(h_bwd_in)
        bwd_feature_t = self.backward_resblocks(h_bwd_in)

        # ==========================================
        # 3. 计算当前帧 t 的前向传播特征 (读取流式 Cache)
        # ==========================================
        if self.stream_cache is None:
            h_tm1, h_tm2, feat_tm1, feat_tm2, fid_tm1 = None, None, None, None, None
        else:
            h_tm1, h_tm2, feat_tm1, feat_tm2, fid_tm1 = self.stream_cache

        if h_tm1 is not None and feat_tm1 is not None:
            raw_mv_t_ref_tm1 = rescale_mv_temporal(
                quarterPixelMV_to_pixelMV(mv_t), src_oe=fid_tm1, tgt_oe=fid_t
            )
            mv_t_ref_tm1 = self._refine_mv_with_gmc(raw_mv_t_ref_tm1, feat_t, feat_tm1)
            t_order1_warped_fwd = mv_warp(h_tm1, mv_t_ref_tm1)
        else:
            t_order1_warped_fwd = torch.zeros_like(feat_t)

        if h_tm2 is not None and feat_tm2 is not None:
            raw_mv_t_ref_tm2 = quarterPixelMV_to_pixelMV(mv_t)
            mv_t_ref_tm2 = self._refine_mv_with_gmc(raw_mv_t_ref_tm2, feat_t, feat_tm2)
            t_order2_warped_fwd = mv_warp(h_tm2, mv_t_ref_tm2)
        else:
            t_order2_warped_fwd = torch.zeros_like(feat_t)

        if h_tm1 is not None or h_tm2 is not None:
            fusion_order1and2_fwd, *_ = self.first_2nd_order_fusion(feat_t, t_order1_warped_fwd, t_order2_warped_fwd)
            h_fwd, _ = self.h_prop_current_feat_fusion(fusion_order1and2_fwd, feat_t)
        else:
            h_fwd = feat_t

        bidirect_feat = self.memory_efficient_fusion(h_fwd, bwd_feature_t, feat_t) # 始终维持在 64 维
        h_fwd = self.forward_resblocks(bidirect_feat)

        # ==========================================
        # 4. 更新流式 Cache (为下一帧的到来做准备)
        # ==========================================
        self.stream_cache = (h_fwd.detach(), h_tm1, feat_t.detach(), feat_tm1, fid_t)

        # ==========================================
        # 5. 生成高分辨率图像
        # ==========================================
        up_feat = self.deint_up(h_fwd, o_e=fid_t)
        hr_feat = self.refinement(up_feat)
        out_residual = self.last_conv(hr_feat)
        base_frame = F.interpolate(imgs_win[:, 0], scale_factor=(2, 1), mode='bilinear', align_corners=False)
        
        return out_residual + base_frame

    def forward(self, imgs, mv_fwd, field_ids):
        B, T, C, H, W = imgs.shape
        
        flags = field_ids.view(B, T, 1, 1, 1).expand(B, T, 1, H, W).float() 
        inp = torch.cat([imgs, flags], dim=2)
        feats = self.feat_extract(inp.view(-1, C + 1, H, W)).view(B, T, self.mid, H, W)

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
                    raw_mv_t1_ref_t2 = rescale_mv_temporal(
                        quarterPixelMV_to_pixelMV(-mv_fwd[:, t+2]),
                        src_oe=field_ids[:, t+2],
                        tgt_oe=field_ids[:, t+1]
                    )
                    mv_t1_ref_t2 = self._refine_mv_with_gmc(raw_mv_t1_ref_t2, feats[:, t+1], feats[:, t+2])
                    t1_warped_from_t2 = mv_warp(feats[:, t+2], mv_t1_ref_t2)
                    hidden_state_t1, _ = self.h_prop_current_feat_fusion(t1_warped_from_t2, feats[:, t+1])
                    hidden_state_t1 = self.backward_resblocks(hidden_state_t1)
                else:
                    hidden_state_t1 = feats[:, t+1]
                
                raw_mv_t_ref_t1 = rescale_mv_temporal(
                    quarterPixelMV_to_pixelMV(-mv_fwd[:, t+1]),
                    src_oe=field_ids[:, t+1],
                    tgt_oe=field_ids[:, t]
                )
                mv_t_ref_t1 = self._refine_mv_with_gmc(raw_mv_t_ref_t1, feats[:, t], feats[:, t+1])
                t_order1_warped_bwd = mv_warp(hidden_state_t1, mv_t_ref_t1)
                
                if has_t2:
                    raw_mv_t_ref_t2 = quarterPixelMV_to_pixelMV(-mv_fwd[:, t+2])
                    mv_t_ref_t2 = self._refine_mv_with_gmc(raw_mv_t_ref_t2, feats[:, t], feats[:, t+2])
                    t_order2_warped_bwd = mv_warp(feats[:, t+2], mv_t_ref_t2)
                else:
                    t_order2_warped_bwd = torch.zeros_like(t_order1_warped_bwd) 

                fusion_order1and2_bwd, *_= self.first_2nd_order_fusion(feats[:, t], t_order1_warped_bwd, t_order2_warped_bwd)
                h_bwd, _ = self.h_prop_current_feat_fusion(fusion_order1and2_bwd, feats[:, t])
                
                h_bwd_in = torch.cat([h_bwd, feats[:, t]], dim=1) 
                # ✅ 降维：128 -> 64
                h_bwd_in = self.bwd_channel_reduce(h_bwd_in)
                bwd_features[t] = self.backward_resblocks(h_bwd_in)

            # --- 2. 前向传播与融合 ---
            fwd_features = [None] * T
            outs = [None] * T 
            
            for t in range(T):
                has_tm1 = (t - 1 >= 0)
                has_tm2 = (t - 2 >= 0)
                
                if has_tm1:
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
                    fusion_order1and2_fwd, *_= self.first_2nd_order_fusion(feats[:, t], t_order1_warped_fwd, t_order2_warped_fwd)
                    h_fwd, _ = self.h_prop_current_feat_fusion(fusion_order1and2_fwd, feats[:, t])
                else:
                    h_fwd = feats[:, t]

                # ✅ 修复：使用无峰值的 Memory Efficient Fusion，训练和推理逻辑彻底对齐
                bidirect_feat = self.memory_efficient_fusion(h_fwd, bwd_features[t], feats[:, t])
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

        else:
            with torch.no_grad():
                bwd_features = [None] * T
                fwd_features = [None] * T
                outs = [None] * T
                
                if self.h_fwd_cache is None:
                    h_tm1, h_tm2, feat_tm1, feat_tm2 = None, None, None, None
                else:
                    h_tm1, h_tm2, feat_tm1, feat_tm2 = self.h_fwd_cache

                for t in range(T):
                    has_t1 = (t + 1 < T)
                    has_t2 = (t + 2 < T)
                    
                    if not has_t1:
                        bwd_features[t] = feats[:, t]
                        continue
                        
                    if has_t2:
                        raw_mv_t1_ref_t2 = rescale_mv_temporal(
                            quarterPixelMV_to_pixelMV(-mv_fwd[:, t+2]),
                            src_oe=field_ids[:, t+2] if field_ids is not None else (t+2)%2,
                            tgt_oe=field_ids[:, t+1] if field_ids is not None else (t+1)%2
                        )
                        mv_t1_ref_t2 = self._refine_mv_with_gmc(raw_mv_t1_ref_t2, feats[:, t+1], feats[:, t+2])
                        t1_warped_from_t2 = mv_warp(feats[:, t+2], mv_t1_ref_t2)
                        hidden_state_t1, _ = self.h_prop_current_feat_fusion(t1_warped_from_t2, feats[:, t+1])
                        hidden_state_t1 = self.backward_resblocks(hidden_state_t1) 
                    else:
                        hidden_state_t1 = feats[:, t+1]
                    
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

                    fusion_order1and2_bwd, *_= self.first_2nd_order_fusion(feats[:, t], t_order1_warped_bwd, t_order2_warped_bwd)
                    h_bwd, _ = self.h_prop_current_feat_fusion(fusion_order1and2_bwd, feats[:, t])
                    
                    h_bwd_in = torch.cat([h_bwd, feats[:, t]], dim=1)
                    # ✅ 降维：128 -> 64
                    h_bwd_in = self.bwd_channel_reduce(h_bwd_in)
                    bwd_features[t] = self.backward_resblocks(h_bwd_in)

                for t in range(T):
                    ref_feat_1 = feats[:, t-1] if t > 0 else feat_tm1
                    ref_feat_2 = feats[:, t-2] if t > 1 else (feat_tm1 if t == 1 else feat_tm2)
                    
                    if h_tm1 is not None and ref_feat_1 is not None:
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
                        fusion_order1and2_fwd, *_= self.first_2nd_order_fusion(feats[:, t], t_order1_warped_fwd, t_order2_warped_fwd)
                        h_fwd, _ = self.h_prop_current_feat_fusion(fusion_order1and2_fwd, feats[:, t])
                    else:
                        h_fwd = feats[:, t]

                   
                    # ✅ 修复：验证模式下也统一使用显存优化融合
                    bidirect_feat = self.memory_efficient_fusion(h_fwd, bwd_features[t], feats[:, t])
                    h_fwd = self.forward_resblocks(bidirect_feat)
                    fwd_features[t] = h_fwd

                    h_tm2 = h_tm1
                    h_tm1 = h_fwd.detach()
                    feat_tm2 = feat_tm1
                    feat_tm1 = feats[:, t].detach()

                self.h_fwd_cache = (h_tm1, h_tm2, feat_tm1, feat_tm2)

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