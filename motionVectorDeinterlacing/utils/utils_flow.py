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


# 替换 utils_flow.py 中的函数定义：
def rescale_mv_temporal(mv, src_oe=None, tgt_oe=None, scale=0.5, correct_offset=True):
    mv = mv * scale
    
    
        
    return mv