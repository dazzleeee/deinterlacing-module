import torch
import torch.nn as nn
import torch.nn.functional as F


def mv_warp(x, flow, padding_mode='border', align_corners=True):
    """
    这就相当于你那个 MVWarp 的纯函数版本
    """
    B, C, H, W = x.shape
    
    # 1. 生成基础网格
    ys = torch.linspace(-1, 1, H, device=x.device)
    xs = torch.linspace(-1, 1, W, device=x.device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0) # (1, H, W, 2)
    
    # 2. 加上归一化后的光流
    vgrid = grid + torch.stack([
        flow[:, 0] * (2.0 / (W - 1)), 
        flow[:, 1] * (2.0 / (H - 1))
    ], dim=-1)
    
    # 3. 采样返回
    return F.grid_sample(x, vgrid, mode='bilinear', 
                         padding_mode=padding_mode, 
                         align_corners=align_corners)

def default_init_weights(module, scale=1, act_cfg=None):
    """
    根据 act_cfg 自动匹配 Kaiming 初始化参数
    act_cfg 可以是字符串 "ReLU" 或 字典 {"type": "LeakyReLU", "negative_slope": 0.1}
    """
    # 1. 解析配置，提取初始化参数
    nonlinearity = 'relu'
    a = 0
    
    if act_cfg is not None:
        if isinstance(act_cfg, str):
            act_type = act_cfg
        else:
            act_type = act_cfg.get('type', 'ReLU')
            a = act_cfg.get('negative_slope', 0) # 针对 LeakyReLU
            # 如果是 PReLU，虽然 a 是学习出来的，但初始化通常参考 0.25 或 0
            if act_type == 'PReLU':
                a = act_cfg.get('init', 0.25) 
        
        # 映射到 PyTorch init 的命名规范
        mapping = {
            'ReLU': 'relu',
            'LeakyReLU': 'leaky_relu',
            'PReLU': 'leaky_relu', # PReLU 在初始化数学上接近 leaky_relu
            'Tanh': 'tanh',
            'Sigmoid': 'sigmoid'
        }
        nonlinearity = mapping.get(act_type, 'relu')

    # 2. 遍历并初始化
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, 
                a=a, 
                mode='fan_in', 
                nonlinearity=nonlinearity
            )
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()