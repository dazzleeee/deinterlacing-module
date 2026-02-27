import torch
from thop import profile
from models.mvd_net import RealTimeMVDnet # 替换为你的路径
from config.config_schema import MVDNetConfig

def measure_flops_and_params(model, input_shape, device='cuda'):
    # 构造假数据
    B, T, C, H, W = input_shape
    imgs = torch.randn(B, T, C, H, W).to(device)
    mvs = torch.randn(B, T, 2, H, W).to(device)
    fids = torch.zeros(B, T).long().to(device)
    
    # 使用 thop 计算
    flops, params = profile(model, inputs=(imgs, mvs, fids), verbose=False)
    
    print(f"Total Params: {params / 1e6:.2f} M")
    print(f"Total FLOPs : {flops / 1e9:.2f} G MACs (For T={T} frames)")

# 运行: measure_flops_and_params(model.eval(), (1, 5, 3, 540, 960))