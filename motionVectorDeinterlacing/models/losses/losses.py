# models/losses.py
from motionVectorDeinterlacing.models.registry import COMPONENT_REGISTRY, LOSS_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F

from motionVectorDeinterlacing.utils.ops import mv_warp # 1. 修正导入

# ==========================================
# 1. 基础图像重建 Loss (Charbonnier)
# ==========================================
@LOSS_REGISTRY.register('CharbonnierLoss')
class CharbonnierLoss(nn.Module):
    """
    比 L1 更平滑，比 L2 更鲁棒。SR 任务的标准配置。
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


# ==========================================
# 2. 场感知时域对齐 Loss (Field-Aware Temporal Loss)
# ==========================================
@LOSS_REGISTRY.register('FieldAwareMaskedTemporalLoss')
class FieldAwareMaskedTemporalLoss(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, sr_imgs, flows, masks=None):
        # sr_imgs: [B, T, 3, 2H, W] (拼接好的逐行全高清图)
        # flows: [B, T, 2, H, W] (原始隔行大小的流)
        B, T, _, H2, W = sr_imgs.shape 
        H = H2 // 2
        
        loss_temporal = 0.0
        for t in range(1, T):
            curr_sr = sr_imgs[:, t]      # 当前高清帧
            prev_sr = sr_imgs[:, t-1]    # 前一高清帧
            
            # 将物理场光流拉伸到高清尺寸
            flow_t = flows[:, t]
            flow_hr = F.interpolate(flow_t, size=(H2, W), mode='bilinear', align_corners=False).clone()
            flow_hr[:, 1] *= 2.0 # 高度拉伸了两倍，Y方向光流幅度也要乘2
            
            mask_hr = 1.0
            if masks is not None:
                mask_hr = F.interpolate(masks[:, t].unsqueeze(1), size=(H2, W), mode='bilinear', align_corners=False)
            
            warped_prev = mv_warp(prev_sr, flow_hr) 
            diff = torch.abs(curr_sr - warped_prev)
            loss_temporal += (diff * mask_hr).mean()

        return loss_temporal / (T - 1)





# ==========================================
# 3. 光流约束 Loss (FlowLoss)
# ==========================================
@LOSS_REGISTRY.register('FlowLoss')
class FlowLoss(nn.Module):
    """
    【自监督约束】包含光度一致性和图像指导的光流平滑。
    必须传入纯粹的“场”图像，禁止传入交错帧，防止梯度爆炸。
    """
    def __init__(self):
        super().__init__()
      

    def gradient(self, data):
        D_dy = data[:, :, 1:] - data[:, :, :-1]
        D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
        return D_dx, D_dy

    def smooth_loss(self, flow, img):
        flow_dx, flow_dy = self.gradient(flow)
        img_dx, img_dy = self.gradient(img)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True))
        return (torch.abs(flow_dx) * weights_x).mean() + (torch.abs(flow_dy) * weights_y).mean()

    def forward(self, refined_flows, imgs, conf_masks=None):
        B, T, _, H, W = refined_flows.shape
        loss_smooth, loss_photo = 0.0, 0.0
        
        for t in range(T):
            flow = refined_flows[:, t]
            curr_img = imgs[:, t]
            loss_smooth += self.smooth_loss(flow, curr_img)
            
            if t > 0: 
                prev_img = imgs[:, t-1] 
                warped_prev = mv_warp(prev_img, flow)
                diff = torch.sqrt((curr_img - warped_prev)**2 + 1e-6)
                
                if conf_masks is not None:
                    mask = conf_masks[:, t] 
                    if mask.shape[-2:] != diff.shape[-2:]:
                        mask = F.interpolate(mask, size=diff.shape[-2:], mode='bilinear')
                    diff = diff * mask
                    
                loss_photo += diff.mean()
            
        return loss_photo / (T - 1), loss_smooth / T


# ==========================================
# 4. 跳帧静态背景稳定性 Loss (Static Background)
# ==========================================
@LOSS_REGISTRY.register('StaticBackgroundLoss')
class StaticBackgroundLoss(nn.Module):
    """
    【抗闪烁约束】强制远距离帧（t-2, t+2）在静态背景下保持一致。
    """
    def __init__(self, motion_sensitivity=0.5): 
        super().__init__()
        self.motion_sensitivity = motion_sensitivity

    def forward(self, sr_imgs, ref_flows):
        loss = 0.0
        B, T, C, H, W = sr_imgs.shape
        
        for t in range(2, T - 2):
            curr = sr_imgs[:, t]
            # 运动强度指示器（用单场光流做近似判断即可）
            flow_mag = torch.sum(ref_flows[:, t]**2, dim=1, keepdim=True)
            flow_mag_hr = F.interpolate(flow_mag, size=(H, W), mode='bilinear')
            static_weight = torch.exp(-self.motion_sensitivity * flow_mag_hr)
            
            loss += (torch.abs(curr - sr_imgs[:, t-2]) * static_weight).mean()
            loss += (torch.abs(curr - sr_imgs[:, t+2]) * static_weight).mean()
            
        return loss / (T - 4 + 1e-8)


# ==========================================
# 5. MVDNet 损失总线 (Loss Aggregator)
# ==========================================
@LOSS_REGISTRY.register('MVDNetLossAggregator')
class MVDNetLossAggregator(nn.Module):
    """
    完全由 Config 驱动的 Loss 总包工头，实现课程学习、动态权重调节与 Top/Bot 场分离解耦。
    """
    def __init__(self, loss_cfg):
        super().__init__()
        
        # 1. 提取配置
        char_cfg = loss_cfg.get('charbonnier', {})
        flow_cfg = loss_cfg.get('flow', {})
        temp_cfg = loss_cfg.get('temporal', {})
        pp_cfg = loss_cfg.get('pingpong', {})

        # 2. 读取权重参数
        self.w_char = char_cfg.get('weight', 1.0)
        self.w_flow = flow_cfg.get('weight', 0.05)
        self.w_temp = temp_cfg.get('weight', 0.1) 
        self.w_pp = pp_cfg.get('weight', 0.5)

        # 3. 读取开启轮次
        self.start_flow = flow_cfg.get('start_epoch', 50)
        self.start_temp = temp_cfg.get('start_epoch', 150)
        self.start_pp = pp_cfg.get('start_epoch', 300)
        
        motion_sens = pp_cfg.get('motion_sensitivity', 0.5)

        # 4. 实例化裁判员
        self.char_loss = CharbonnierLoss()
        self.flow_loss = FlowLoss()
        self.temp_loss = FieldAwareMaskedTemporalLoss() 
        self.pp_loss = StaticBackgroundLoss(motion_sensitivity=motion_sens)

    def forward(self, outputs, targets, epoch):
        loss_dict = {}
        total_loss = 0.0
        
        l_char = self.char_loss(outputs['sr'], targets['hr'])
        loss_dict['l_char'] = l_char * self.w_char
        total_loss += loss_dict['l_char']

        flows = outputs['flows']
        lr_imgs = outputs['lr_imgs']

        if epoch >= self.start_flow:
            l_photo, l_smooth = self.flow_loss(flows, lr_imgs)
            loss_dict['l_flow'] = (l_photo + l_smooth) * self.w_flow
            total_loss += loss_dict['l_flow']

        if epoch >= self.start_temp and self.w_temp > 0.0:
            l_temp = self.temp_loss(outputs['sr'], flows, outputs.get('masks'))
            loss_dict['l_temp'] = l_temp * self.w_temp
            total_loss += loss_dict['l_temp']
            
        if epoch >= self.start_pp:
            l_pp = self.pp_loss(outputs['sr'], flows)
            loss_dict['l_pp'] = l_pp * self.w_pp
            total_loss += loss_dict['l_pp']

        loss_dict['total_loss'] = total_loss
        return loss_dict