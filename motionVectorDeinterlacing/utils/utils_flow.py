# process motion vector for warping: 
# 1.from H.264 raw（1/4 pel） to pixel, 
# 2.and apply vertical physical corrections

import torch

def quarterPixelMV_to_pixelMV(raw_mv):
    """
    [原子操作 1]
    将 H.264 原始运动矢量 (1/4 像素单位) 转换为实际像素单位。
    
    Args:
        raw_mv (Tensor): [B, 2, H, W]
    Returns:
        Tensor: 像素单位的 MV
    """
    # 这里的 .float() 是为了保险，防止传入的是 int 类型导致无法除出小数
    return raw_mv.float() / 4.0


def process_flow_physics(mv, src_oe, tgt_oe, scale=0.5, correct_offset=True):
    """
    [原子操作 2]
    处理运动矢量的物理特性：时间缩放 + 隔行扫描垂直偏移修正。
    
    Args:
        mv (Tensor): [B, 2, H, W] 像素单位的 MV
        src_oe (Tensor): 参考场极性 (0=Top, 1=Bot)
        tgt_oe (Tensor): 目标场极性
        scale (float): 时间步长缩放 (0.5 或 1.0)
        correct_offset (bool): 开关，是否执行物理修正
    Returns:
        Tensor: 修正后的 MV
    """
    # 1. 时间步长缩放 (Scale)
    # 比如从 t->t-2 的 MV 缩放到 t->t-1，就需要 * 0.5
    mv = mv * scale
    
    # 2. 垂直偏移修正 (Vertical Offset)
    if correct_offset:
        # 计算 Mask：只有异场 (Top->Bot 或 Bot->Top) 才需要修正
        # view(-1, 1, 1, 1) 是为了广播到 [B, 2, H, W]
        mask = (src_oe != tgt_oe).view(-1, 1, 1, 1).float()
        
        # 计算 Offset：
        # 如果目标是 Bot(1)，说明空间位置偏下，需要 +0.5 去找它
        # 如果目标是 Top(0)，说明空间位置偏上，需要 -0.5 去找它
        offset = torch.where(tgt_oe.view(-1, 1, 1, 1) == 1, 0.5, -0.5)
        
        # 为了不破坏原来的 Tensor 计算图，建议 clone 一下
        # 且只修正 Y 轴 (index 1)
        mv_corrected = mv.clone()
        mv_corrected[:, 1:2, :, :] += (mask * offset)
        return mv_corrected
        
    return mv