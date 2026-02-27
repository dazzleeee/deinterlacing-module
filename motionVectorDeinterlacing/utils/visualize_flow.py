import cv2
import numpy as np

def flow_to_color(flow_tensor, max_val=None):
    """
    将 [2, H, W] 的光流 Tensor 转为 HWC 的 RGB 图像用于保存
    """
    flow = flow_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    h, w = flow.shape[:2]
    
    # 转换为极坐标 (角度代表颜色，长度代表亮度)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2 # 色相 (0-180)
    hsv[..., 1] = 255                   # 饱和度最大
    
    if max_val is None:
        max_val = np.max(mag) + 1e-6
    hsv[..., 2] = np.clip(mag * 255 / max_val, 0, 255) # 亮度反映运动大小
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb