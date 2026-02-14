import torch
from thop import profile
from models import build_model  # 假设你有一个构建模型的接口，或者直接 import 你的类
# 如果没有 build_model，可以直接：from models.architectures.mvd_net import MVDNet

def get_model_complexity(model, input_shape):
    """
    计算模型的 FLOPs 和 参数量。
    
    Args:
        model (nn.Module): 你的模型实例
        input_shape (tuple): 输入张量的形状，例如 (1, 3, 256, 256) 对于单帧
                             如果是视频序列，可能是 (1, 5, 3, 256, 256) [B, T, C, H, W]
    """
    # 确保模型在 CPU 上，避免显存干扰
    model.cpu()
    model.eval()

    # 创建一个伪造的输入 Tensor
    input_tensor = torch.randn(input_shape)

    # 计算 FLOPs 和 Params
    # 注意：custom_ops 参数用于处理一些 thop 无法自动识别的自定义层
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    return flops, params

if __name__ == '__main__':
    # 1. 配置你的模型参数
    # 假设你的 MVDNet 需要 mid_channels=64
    from models.architectures.mvd_net import MVDNet
    
    model = MVDNet(mid_channels=64, num_blocks=15)
    
    # 2. 定义输入大小
    # 假设输入是：Batch=1, Frames=5, Channels=3, Height=256, Width=256
    # 请根据你的 forward_sequence 函数的实际参数调整 inputs
    # 如果 forward 需要多个参数 (imgs, flow, etc.)，profile 的 inputs 需要传 tuple
    # 这里以最简单的单 Tensor 输入为例，实际情况可能需要修改
    input_shape = (1, 5, 3, 256, 256) 
    
    # 注意：thop 默认只支持单个输入。如果你的 forward 接受多个参数，
    # 你可能需要写一个 Wrapper (包装器) 类来把多个参数打包成一个。
    
    print(f"Testing model: {model.__class__.__name__}")
    
    # 简单估算（仅针对标准卷积层）
    # 如果报错，说明 thop 不支持你的自定义层，或者输入格式不对
    try:
        # 创建 Dummy Inputs
        imgs = torch.randn(1, 5, 3, 64, 64) # 使用小尺寸以快速测试
        mvs = torch.randn(1, 5, 2, 64, 64)
        fids = torch.zeros(1, 5)
        
        # 使用 thop.profile
        flops, params = profile(model, inputs=(imgs, mvs, fids), verbose=False)
        
        print(f"FLOPs: {flops / 1e9:.2f} G (Giga-FLOPs)")
        print(f"Params: {params / 1e6:.2f} M (Million-Params)")
        
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        print("Tip: 确保输入参数与 forward 函数完全匹配。")