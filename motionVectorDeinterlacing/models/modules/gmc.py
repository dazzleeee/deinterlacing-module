# models/modules/gmc.py
from ..registry import COMPONENT_REGISTRY  # 导入组件注册表
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2  # 工业级图像处理通常依赖 cv2 的 RANSAC，速度最快
from motionVectorDeinterlacing.utils.ops import mv_warp
import logging # 在文件开头导入

@COMPONENT_REGISTRY.register('GMC')  # <--- 加上这一行！
class GlobalMotionCompensator(nn.Module):
    def __init__(self, min_inliers=10, max_scale_err=0.2, max_trans_err=0.2):
        """
        Args:
            min_inliers (int): 最少需要多少个一致点，少于这个数直接放弃 GMC。
            max_scale_err (float): 允许的最大缩放误差 (比如 0.2 代表只允许 0.8~1.2 倍缩放)。
            max_trans_err (float): 允许的最大平移误差 (相对图像宽高的比例)。
        """
        super().__init__()
        self.min_inliers = min_inliers
        self.max_scale_err = max_scale_err
        self.max_trans_err = max_trans_err

    def _tensor_to_numpy_pts(self, flow):
        """
        把稀疏的光流图转换为点集 (Src -> Dst)
        flow: [2, H, W]
        Return: src_pts (N, 2), dst_pts (N, 2)
        """
        # 1. 找到所有非零向量的坐标（简单过滤静止背景和无效区域）
        # 这里假设 flow 绝对值大于 0.1 的才算有效运动，去除噪点
        mag = torch.sum(flow**2, dim=0).sqrt()
        valid_mask = mag > 0.1
        
        y_idxs, x_idxs = torch.where(valid_mask)
        
        if len(y_idxs) < self.min_inliers:
            return None, None

        # Src: 原始坐标 (x, y)
        src_pts = torch.stack([x_idxs, y_idxs], dim=1).float()
        
        # Dst: 目标坐标 = 原始坐标 + 光流
        # 注意 flow 的顺序通常是 [x_flow, y_flow] 或 [y, x]，要根据你项目确认！
        # 假设 flow 是 [dx, dy]
        deltas = flow[:, y_idxs, x_idxs].permute(1, 0) # (N, 2)
        dst_pts = src_pts + deltas

        return src_pts.cpu().numpy(), dst_pts.cpu().numpy()

    def _calculate_affine_robust(self, flow, H, W):
        """核心：带熔断机制的矩阵计算"""
        src, dst = self._tensor_to_numpy_pts(flow)
        
        # === 熔断 1: 点太少 ===
        if src is None or len(src) < self.min_inliers:
            return torch.eye(2, 3)

        # 使用 OpenCV 的 RANSAC 估算仿射矩阵 (2x3)
        # cv2.estimateAffinePartial2D 比 estimateAffine2D 更稳，它限制了剪切(Shear)
        # 只允许：平移 + 旋转 + 统一缩放 (最符合摄像机运动)
        try:
            matrix, _ = cv2.estimateAffinePartial2D(
                src, dst, 
                method=cv2.RANSAC, 
                ransacReprojThreshold=3.0
            )
        except Exception as e:
            # 真正使用 logging 记录警告！
            # 用 warning 级别，这样在查阅训练/推理日志时能轻易过滤出来
            logging.warning(
                f"[GMC] OpenCV estimateAffinePartial2D failed. "
                f"Error: {e} | src shape: {src.shape}, dst shape: {dst.shape}. "
                f"Fallback to Identity matrix."
            )
            
            # 依然执行熔断，返回单位矩阵保命
            return torch.eye(2, 3)

        # === 熔断 2: 算失败了 ===
        if matrix is None:
            return torch.eye(2, 3)

        # === 熔断 3: 检查矩阵是否“疯了” (Sanity Check) ===
        # 矩阵结构: [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
        sx = np.sqrt(matrix[0, 0]**2 + matrix[0, 1]**2)
        sy = np.sqrt(matrix[1, 0]**2 + matrix[1, 1]**2)
        
        # 检查缩放是否过大/过小 (摄像机一般不会突然变焦两倍)
        if abs(sx - 1) > self.max_scale_err or abs(sy - 1) > self.max_scale_err:
            return torch.eye(2, 3)

        # 检查平移是否飞出屏幕 (比如平移量超过了图像的 20%)
        tx, ty = matrix[0, 2], matrix[1, 2]
        if abs(tx) > W * self.max_trans_err or abs(ty) > H * self.max_trans_err:
            return torch.eye(2, 3)

        return torch.from_numpy(matrix).float()

    def get_global_flow_map(self, affine_matrix, H, W, device):
        """把 2x3 矩阵还原成 Dense Flow Map"""
        # 如果是单位矩阵，直接返回 0
        if torch.equal(affine_matrix, torch.eye(2, 3).to(affine_matrix)):
            return torch.zeros((1, 2, H, W), device=device)
            
        # 生成网格
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        # Homogeneous coordinates: [x, y, 1]
        grid = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().to(device) # (H, W, 3)
        grid = grid.unsqueeze(0) # (1, H, W, 3)
        
        # Apply matrix: (1, H, W, 3) @ (2, 3).T -> (1, H, W, 2)
        # grid points (new)
        new_grid = grid @ affine_matrix.T.to(device)
        
        # Global Flow = New Pos - Old Pos
        # grid[..., :2] 取出 x, y
        global_flow = new_grid - grid[..., :2]
        
        # Permute to (B, 2, H, W)
        return global_flow.permute(0, 3, 1, 2)

    # ... 前面的 __init__ 和矩阵计算方法完全保持不变 ...

    def forward(self, raw_mv_blocks, H, W, interpolation_mode='bilinear'):
        """
        Args:
            raw_mv_blocks: [B, 2, h_small, w_small] 稀疏MV
            H, W: 目标分辨率
            interpolation_mode: 'bilinear' (适配普通 Refiner) 或 'nearest' (适配 Convex)
        Returns:
            object_motion: [B, 2, H, W] 去除背景后的物体运动
            global_flow: [B, 2, H, W] 背景运动
        """
        B, _, h_small, w_small = raw_mv_blocks.shape
        device = raw_mv_blocks.device
        
        global_flow_list = []
        
        # 1. 计算仿射矩阵并转为 Dense Global Flow
        for i in range(B):
            matrix = self._calculate_affine_robust(raw_mv_blocks[i], H, W)
            g_flow = self.get_global_flow_map(matrix, H, W, device)
            global_flow_list.append(g_flow)
            
        global_flow = torch.cat(global_flow_list, dim=0) # [B, 2, H, W] 全分辨率摄像机运动
        
        # ==========================================
        # 2. 根据不同的 Refiner 需求，执行不同的剥离策略
        # ==========================================
        if interpolation_mode == 'bilinear':
            # 【适配普通 Refiner】(如 ImageGuided, Gated, Vanilla)
            # 直接把原始 MV 平滑放大，在全分辨率下做减法
            dense_total_mv = F.interpolate(
                raw_mv_blocks, size=(H, W), mode='bilinear', align_corners=False
            )
            object_motion = dense_total_mv - global_flow
            
        elif interpolation_mode == 'nearest':
            # a. 直接用 area 插值将全分辨率 global_flow 降采样到目标 MV 的长宽
            global_flow_lr = F.interpolate(
                global_flow, size=(h_small, w_small), mode='area'
            )
            
            # b. 在同分辨率下减去全局运动
            object_motion_lr = raw_mv_blocks - global_flow_lr
            
            # c. 再将局部运动向量放大回原图尺寸
            object_motion = F.interpolate(
                object_motion_lr, size=(H, W), mode='nearest'
            )
            
        else:
            raise ValueError(f"Unsupported interpolation_mode: {interpolation_mode}")
        
        return object_motion, global_flow
    
@COMPONENT_REGISTRY.register('IdentityGMC')
class IdentityGMC(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, raw_mv_blocks, H, W, interpolation_mode='bilinear'):
        # 1. 直接把原始稀疏 MV 放大到全分辨率
        # 即使不做背景补偿，也要支持不同的插值方式（bilinear 或 nearest）
        # 这样才能配合不同的 Refiner (比如 ConvexUpsamplingRefiner)
        dense_total_mv = F.interpolate(
            raw_mv_blocks, size=(H, W), mode=interpolation_mode
        )
        
        # 2. 模拟解耦接口：物体运动 = 总运动，背景运动 = 0
        global_flow = torch.zeros_like(dense_total_mv)
        
        return dense_total_mv, global_flow

class GPUAffinePredictor(nn.Module):
    """纯 GPU 的极速轻量级全局运动（仿射矩阵）预测器"""
    def __init__(self, in_channels=128): # 改变：吃 64+64 通道的特征
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1) 
        )
        self.fc = nn.Linear(64, 6)
        nn.init.zeros_(self.fc.weight)
        self.fc.bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, feat_curr, feat_ref):
        B = feat_curr.size(0)
        x = torch.cat([feat_curr, feat_ref], dim=1) 
        feat = self.net(x).view(B, -1)
        theta = self.fc(feat).view(B, 2, 3)
        return theta

@COMPONENT_REGISTRY.register('FastGPUGMC')
class FastGPUGMC(nn.Module):
    """
    无缝替代原来 OpenCV RANSAC GMC 的纯 GPU 版本。
    完美继承原有接口，直接预测矩阵并转换为密集全局光流。
    """
    def __init__(self, in_channels=128):
        super().__init__()
        self.affine_predictor = GPUAffinePredictor(in_channels=in_channels)

    def get_global_flow_map(self, affine_matrix, H, W, device):
        B = affine_matrix.size(0)
        # 1. 生成基础网格
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # [H, W, 3]
        
        # ================== 🚀 核心修复部分 ==================
        # 2. 将空间维度 H 和 W 拉平，方便与仿射矩阵进行 Batch 矩阵乘法
        # grid 原本是 [H, W, 3]，拉平为 [H*W, 3]，再扩展到 Batch 维度变成 [B, H*W, 3]
        grid_flat = grid.view(-1, 3).unsqueeze(0).expand(B, -1, -1)
        
        # 3. 矩阵乘法: [B, H*W, 3] @ [B, 3, 2] -> [B, H*W, 2]
        new_grid_flat = torch.matmul(grid_flat, affine_matrix.transpose(1, 2))
        
        # 4. 算完之后，把展平的维度重新变回图像的 H 和 W
        new_grid = new_grid_flat.view(B, H, W, 2)
        grid_expanded = grid.unsqueeze(0).expand(B, -1, -1, -1) # 原始 grid 也扩展到 batch 用于相减
        # =====================================================
        
        # 5. 计算全局光流差值
        global_flow = new_grid - grid_expanded[..., :2]
        return global_flow.permute(0, 3, 1, 2) # [B, 2, H, W]

    def forward(self, raw_mv_blocks, H, W, interpolation_mode='bilinear', feat_curr=None, feat_ref=None):
        B = raw_mv_blocks.size(0)
        device = raw_mv_blocks.device
        
        # 1. 极速预测全局仿射矩阵
        theta = self.affine_predictor(feat_curr, feat_ref)
        
        # 2. 转换为全分辨率全局光流 (代替了原来的 RANSAC)
        global_flow = self.get_global_flow_map(theta, H, W, device)
        
        # 3. 剥离局部运动 (和老代码逻辑完美一致，适配凸上采样)
        if interpolation_mode == 'bilinear':
            dense_total_mv = F.interpolate(raw_mv_blocks, size=(H, W), mode='bilinear', align_corners=False)
            object_motion = dense_total_mv - global_flow
        elif interpolation_mode == 'nearest':
            _, _, h_small, w_small = raw_mv_blocks.shape
            global_flow_lr = F.interpolate(global_flow, size=(h_small, w_small), mode='area')
            object_motion_lr = raw_mv_blocks - global_flow_lr
            object_motion = F.interpolate(object_motion_lr, size=(H, W), mode='nearest')
            
        return object_motion, global_flow