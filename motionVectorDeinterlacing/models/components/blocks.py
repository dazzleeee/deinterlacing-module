import torch
import torch.nn as nn
from ..registry import COMPONENT_REGISTRY
# 必须导入你定义的助手函数
from .activation import build_activation 
from motionVectorDeinterlacing.utils.ops import default_init_weights
import torch.nn.functional as F

# ==========================================
# 选项 A: 标准残差块 (StandardResidualBlock)
# 特点：快，PC GPU 兼容性最好
# ==========================================
@COMPONENT_REGISTRY.register('StandardResidualBlock')
class StandardResidualBlock(nn.Module):
    def __init__(self, nf, act_cfg="PReLU", res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            build_activation(act_cfg),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
        nn.init.zeros_(self.body[2].weight)
        nn.init.zeros_(self.body[2].bias)
    def forward(self, x):
        # 使用残差缩放 λ=0.1
        return x + self.body(x) * self.res_scale

# ==========================================
# 选项 B: 通道注意力残差块 (RCAB)
# 特点：画质上限更高，能自动学会哪些通道对去交错更重要
# ==========================================
@COMPONENT_REGISTRY.register('ChannelAttentionBlock')
class ChannelAttentionBlock(nn.Module):
    def __init__(self, nf, act_cfg="PReLU", res_scale=0.1, reduction=16, deploy=False):
        super().__init__()
        self.res_scale = res_scale
        self.conv_layers = nn.Sequential(
            RepConv(nf, nf, deploy=deploy),
            build_activation(act_cfg),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
        
        # 极轻量的通道注意力 (Squeeze-and-Excitation)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf, nf // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, nf, 1, padding=0),
            nn.Sigmoid()
        )
        # 1. 主分支零初始化 (保证初始输出为 0)
        nn.init.zeros_(self.conv_layers[-1].weight)
        nn.init.zeros_(self.conv_layers[-1].bias)
        
        # 2. 注意力分支零初始化 (保证初始权重为 Sigmoid(0) = 0.5)
        nn.init.zeros_(self.ca[-2].weight)
        nn.init.zeros_(self.ca[-2].bias)

    def forward(self, x):
        res = self.conv_layers(x)
        attn = self.ca(res)
        return x + (res * attn) * self.res_scale


class RepConv(nn.Module):
    """底层视觉专用的无 BN 重参数化卷积"""
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- 部署模式 (极简) ---
        if self.deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        # --- 训练模式 (多分支) ---
        else:
            # 分支 1：标准 3x3 卷积 (带偏置)
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
            # 分支 2：1x1 卷积 (不带偏置，因为分支 1 已经有了)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            # 分支 3：恒等映射 (仅当输入输出通道一致时存在)
            self.rbr_identity = nn.Identity() if out_channels == in_channels else None

    def forward(self, x):
        if self.deploy:
            return self.rbr_reparam(x) # 部署时，只有一层 3x3，极速！
        
        # 训练时，三路特征相加，感受野无敌
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        return out

    def switch_to_deploy(self):
        """核心魔法：训练完成后，调用此函数将多分支融合成一个 3x3 卷积"""
        if self.deploy:
            return
        
        # 1. 获取 3x3 的权重和偏置
        weight_3x3 = self.rbr_dense.weight.data
        bias_3x3 = self.rbr_dense.bias.data
        
        # 2. 获取 1x1 的权重，并用 0 填充成 3x3 的形状
        weight_1x1 = self.rbr_1x1.weight.data
        weight_1x1_padded = F.pad(weight_1x1, [1, 1, 1, 1])
        
        # 3. 构造恒等映射的 3x3 权重 (中心点为 1，其余为 0)
        weight_id = torch.zeros_like(weight_3x3)
        if self.rbr_identity is not None:
            for i in range(self.in_channels):
                weight_id[i, i, 1, 1] = 1.0

        # 4. 矩阵加法：融合成最终的单层权重！
        fused_weight = weight_3x3 + weight_1x1_padded + weight_id
        fused_bias = bias_3x3 # 1x1和id分支没有偏置
        
        # 5. 替换为部署用的卷积层
        self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, bias=True)
        self.rbr_reparam.weight.data = fused_weight
        self.rbr_reparam.bias.data = fused_bias
        
        # 6. 删掉训练时的臃肿分支，释放内存
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        self.__delattr__('rbr_identity')
        self.deploy = True

import torch
import torch.nn as nn
# 假设你在这个文件里引入了 default_init_weights 和注册表
# from ..utils.ops import default_init_weights 
# from ..registry import COMPONENT_REGISTRY

@COMPONENT_REGISTRY.register('DeintShufflePack') # 如果你需要注册它
class DeintShufflePack(nn.Module):
    """Pixel Shuffle upsample layer with Deinterlacing support (Optimized Version)."""
 
    def __init__(self, in_channels, out_channels, scale_factor=2,
                 upsample_kernel=3, mode: int=1):
        super().__init__()
        # 加入防呆设计：Deinterlacing 只能做垂直 2 倍放大
        assert scale_factor == 2, "DeintShufflePack only supports scale_factor=2!"
        
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel

        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        
        self.init_weights()
 
    def init_weights(self):
        # 绝不忘掉我们之前讨论过的初始化铁律！
        default_init_weights(self, 1)
 
    def forward(self, x, o_e):
        x = self.upsample_conv(x)
 
        B, C, H, W = x.shape
        C2 = C // 2
        H2 = H * 2 

        # 1. 极其关键：将 mask 转为 bool 型！布尔索引在 CUDA 底层速度最快
        if isinstance(o_e, torch.Tensor):
            mask_odd = o_e.view(B, 1, 1, 1).bool()
        else:
            mask_odd = torch.tensor(o_e, device=x.device, dtype=torch.bool).view(B, 1, 1, 1)

        if self.mode == 1:
            # 拿到两把拉链齿（不增加任何显存，这只是 View）
            x1 = x[:, :C2, :, :]
            x2 = x[:, C2:, :, :]
            
            # 2. 神级优化：使用 torch.where 玩转“零和博弈”
            # torch.where(condition, true_tensor, false_tensor)
            # 如果是奇数场 (mask_odd=True)：x2 在上，x1 在下
            # 如果是偶数场 (mask_odd=False)：x1 在上，x2 在下
            top_half = torch.where(mask_odd, x2, x1)
            bot_half = torch.where(mask_odd, x1, x2)
            
            # 3. 完美咬合！只创建一次最终的输出张量
            total = torch.stack((top_half, bot_half), dim=3).view(B, C2, H2, W)

        return total