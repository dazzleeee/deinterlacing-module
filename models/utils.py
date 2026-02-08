import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MVWarp, ResidualBlock, DeintShufflePack

# Refiner 保持不变 (它非常完美，不需要动)
class ImageGuidedMVRefiner(nn.Module):
    def __init__(self, mid=64):
        super().__init__()
        self.mvwarp = MVWarp()
        self.fusion = nn.Conv2d(2 + mid * 2, mid, 3, 1, 1)
        self.body = nn.Sequential(
            ResidualBlock(mid), ResidualBlock(mid), ResidualBlock(mid), ResidualBlock(mid)
        )
        # Channel 3: Feature Confidence
        self.out_conv = nn.Conv2d(mid, 4, 3, 1, 1) 
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.out_conv.bias.data[2] = 2.0  
        self.out_conv.bias.data[3] = 2.0 

    def forward(self, raw_mv, curr_feat, ref_feat):
        warped_ref_feat = self.mvwarp(ref_feat, raw_mv)
        diff_feat = curr_feat - warped_ref_feat
        inp = torch.cat([raw_mv, curr_feat, diff_feat], dim=1)
        x = self.fusion(inp)
        x = self.body(x)
        out = self.out_conv(x)
        raw_delta = out[:, :2, :, :]
        delta_mv = torch.tanh(raw_delta) * 5.0 
        mv_gate = torch.sigmoid(out[:, 2:3, :, :]) 
        feat_conf = torch.sigmoid(out[:, 3:4, :, :]) 
        refined_mv = (raw_mv * mv_gate) + delta_mv
        return refined_mv, feat_conf

# --- 🔥 重写 MVSR: 步步为营版 ---
class MVSR(nn.Module):
    def __init__(self, mid=64, blocks=15, scale=2):
        super().__init__()
        self.mid = mid
        self.scale = scale
        self.mvwarp = MVWarp()

        # 只需要一个特征提取器了 (不再分 Main/Aux)
        self.feat_extract = nn.Sequential(
            nn.Conv2d(4, mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )

        self.mv_refiner = ImageGuidedMVRefiner(mid)
        
        # 两个方向的 ResBlocks
        self.backward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        self.forward_resblocks = nn.Sequential(*[ResidualBlock(mid) for _ in range(blocks)])
        
        self.refinement = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )
        self.fusion = nn.Conv2d(mid * 2, mid, 1, 1)
        self.deint_up = DeintShufflePack(mid, mid, scale, 3, mode=1)
        self.last_conv = nn.Conv2d(mid, 3, 3, 1, 1)

    def compute_flow(self, mv, curr_feat, ref_feat):
        return self.mv_refiner(mv / 4.0, curr_feat, ref_feat)

    def forward(self, imgs, mv_fwd, mv_bwd, field_ids=None):
        # imgs: [B, T, C, H, W] (这里的 T 是连续的所有帧)
        # mv_fwd: [B, T, 2, H, W] (H.264 MV, 通常 t->t-2)
        
        B, T, C, H, W = imgs.shape

        if field_ids is None:
            field_ids = torch.zeros(B, T, device=imgs.device) 
        
        # 1. 统一提取所有帧的特征
        # Input: Image + Field_ID
        flags = field_ids.view(B, T, 1, 1, 1).expand(B, T, 1, H, W).float()
        inp = torch.cat([imgs, flags], dim=2)
        feats = self.feat_extract(inp.view(-1, C + 1, H, W)).view(B, T, self.mid, H, W)
        
        # 收集输出
        refined_flows_list = [] 
        conf_masks_list = []

        # ==========================================
        # Backward Branch (Future T-1 -> Past 0)
        # ==========================================
        bwd_features = [None] * T
        h_bwd = torch.zeros_like(feats[:, 0])      
        
        for t in range(T - 1, -1, -1):
            curr_feat = feats[:, t]
            
            if t < T - 1:
                ref_feat = feats[:, t+1] # 下一帧 (物理时间 t+1)
                
                # 关键逻辑：
                # 我们只有 t+1 -> t-1 的 MV (mv_fwd[:, t+1])
                # 我们需要 t+1 -> t 的流
                # 假设：运动是线性的，取反(变成 backward) 并折半
                # Raw MV: -(t+1 -> t-1) * 0.5 ≈ (t+1 -> t)
                raw_mv_step = -mv_fwd[:, t+1] * 0.5
                
                # Refine: 让网络看着 t+1 和 t 的图，把这个折半的 MV 修准
                flow_step, conf_step = self.compute_flow(raw_mv_step, curr_feat, ref_feat)
                
                # Warp 隐状态
                h_bwd_warped = self.mvwarp(h_bwd, flow_step)
                
                # Masking: 烟雾区域切断历史
                h_bwd = h_bwd_warped * conf_step
            else:
                # 最后一帧，没未来
                h_bwd = torch.zeros_like(curr_feat)
            
            # 融合当前帧信息 + 历史信息
            h_bwd = h_bwd + curr_feat
            h_bwd = self.backward_resblocks(h_bwd)
            bwd_features[t] = h_bwd

        # ==========================================
        # Forward Branch (Past 0 -> Future T-1)
        # ==========================================
        fwd_features = [None] * T
        h_fwd = torch.zeros_like(feats[:, 0])
        
        for t in range(T):
            curr_feat = feats[:, t]
            
            if t > 0:
                ref_feat = feats[:, t-1] # 上一帧
                
                # 关键逻辑：
                # 我们有 t -> t-2 的 MV (mv_fwd[:, t])
                # 我们需要 t -> t-1 的流
                # 假设：取一半
                raw_mv_step = mv_fwd[:, t] * 0.5
                
                # Refine: 精修这个 0.5 MV
                flow_step, conf_step = self.compute_flow(raw_mv_step, curr_feat, ref_feat)
                
                # 存下来算 Loss
                refined_flows_list.append(flow_step)
                conf_masks_list.append(conf_step)
                
                # Warp 隐状态
                h_fwd_warped = self.mvwarp(h_fwd, flow_step)
                
                # Masking
                h_fwd = h_fwd_warped * conf_step
            else:
                # 第一帧，没过去
                # 占位
                refined_flows_list.append(torch.zeros_like(mv_fwd[:, 0]))
                conf_masks_list.append(torch.ones_like(mv_fwd[:, 0, 0:1]))
                h_fwd = torch.zeros_like(curr_feat)

            # 融合
            h_fwd = h_fwd + curr_feat
            h_fwd = self.forward_resblocks(h_fwd)
            fwd_features[t] = h_fwd

        # ==========================================
        # Reconstruction (每帧都重建！)
        # ==========================================
        outs = []
        for t in range(T):
            # 拼接双向特征
            fused = self.fusion(torch.cat([fwd_features[t], bwd_features[t]], dim=1))            
            refined = self.refinement(fused)
            
            # 每一帧都根据自己的 Field ID 进行上采样
            current_o_e = field_ids[:, t] 
            up_feat = self.deint_up(refined, o_e=current_o_e)
            out = self.last_conv(up_feat)   
            outs.append(out)

        sr_imgs = torch.stack(outs, dim=1)
        refined_flows = torch.stack(refined_flows_list, dim=1)
        conf_masks = torch.stack(conf_masks_list, dim=1)
        
        # 返回所有帧的结果
        return sr_imgs, refined_flows, conf_masks
    
@register_component
class MVRefiner_Adaptive(nn.Module):
    def __init__(self, mid=64):
        super().__init__()
        self.mvwarp = MVWarp()
        self.fusion = nn.Conv2d(2 + mid * 2, mid, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(mid) for _ in range(4)])
        
        # 4通道：0-1(Delta), 2(Gate), 3(Confidence)
        self.out_conv = nn.Conv2d(mid, 4, 3, 1, 1)
        
        # 【新增】运动检测头：专门从 diff 中提取运动得分
        self.motion_detector = nn.Sequential(
            nn.Conv2d(mid, 16, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid()
        )
        # 初始化
        nn.init.zeros_(self.out_conv.weight); self.out_conv.bias.data[2:4] = 2.0

    def forward(self, raw_mv, curr_feat, ref_feat):
        # 1. 计算对齐残差 (互信息最大化的负向指标)
        warped_ref = self.mvwarp(ref_feat, raw_mv)
        diff = curr_feat - warped_ref
        
        # 2. 生成运动掩码 (1=动, 0=静)
        # 标清视频噪声多，通过 detector 的小卷积核进行空间平滑，防止噪点被误判为运动
        motion_mask = self.motion_detector(torch.abs(diff))
        
        # 3. 深度推理：计算 Delta 和 Gate
        x = self.body(self.fusion(torch.cat([raw_mv, curr_feat, diff], dim=1)))
        out = self.out_conv(x)
        
        refined_mv = (raw_mv * torch.sigmoid(out[:, 2:3])) + torch.tanh(out[:, :2]) * 5.0
        conf = torch.sigmoid(out[:, 3:4])
        
        # 返回精修后的 MV、特征置信度，以及运动掩码
        return refined_mv, conf, motion_mask
    
