import numpy as np
import cv2

def flow_to_image(flow):
    """
    将光流/MV 转换为可视化图像 (RGB)。
    
    Args:
        flow: numpy array of shape [H, W, 2]. 
              flow[:,:,0] 是水平位移 (x), flow[:,:,1] 是垂直位移 (y).
    Returns:
        vis_img: numpy array of shape [H, W, 3], range [0, 255], uint8.
    """
    h, w = flow.shape[:2]
    
    # 1. 创建 HSV 画布
    # H (Hue): 色相，代表方向
    # S (Saturation): 饱和度，代表大小
    # V (Value): 亮度，设为最大 255
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    # 2. 将直角坐标 (x, y) 转换为极坐标 (magnitude, angle)
    # mag: 向量长度 (速度)，ang: 向量角度 (方向)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 3. 映射角度到色相 (Hue)
    # OpenCV 的 Hue 范围是 [0, 179]，所以角度 / 2
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # 4. 映射大小到亮度 (Value)
    # 归一化：让最大的运动变成最亮 (255)
    # 这里的 cv2.normalize 是为了让可视化更明显，把运动放大到 0-255 范围
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # 5. HSV 转 RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return rgb

# --- 测试代码 ---
if __name__ == '__main__':
    # 造一个伪造的 MV: H=256, W=256, 2通道
    dummy_flow = np.zeros((256, 256, 2), dtype=np.float32)
    
    # 让左半部分向右动 (x=5)，右半部分向下动 (y=5)
    dummy_flow[:, :128, 0] = 5.0
    dummy_flow[:, 128:, 1] = 5.0
    
    # 可视化
    vis_img = flow_to_image(dummy_flow)
    
    # 保存
    cv2.imwrite('test_flow_vis.png', vis_img)
    print("Flow visualization saved to test_flow_vis.png")