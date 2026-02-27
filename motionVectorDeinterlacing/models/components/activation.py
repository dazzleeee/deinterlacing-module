import torch.nn as nn
from ..registry import ACTIVATION_REGISTRY, build_from_cfg

# --- 1. 注册 PyTorch 原生激活函数 ---
# 这样你就能在 YAML 里直接写 "ReLU" 或 "LeakyReLU" 了
ACTIVATION_REGISTRY.register('ReLU')(nn.ReLU)
ACTIVATION_REGISTRY.register('LeakyReLU')(nn.LeakyReLU)
ACTIVATION_REGISTRY.register('PReLU')(nn.PReLU)
ACTIVATION_REGISTRY.register('Sigmoid')(nn.Sigmoid)
ACTIVATION_REGISTRY.register('Tanh')(nn.Tanh)
ACTIVATION_REGISTRY.register('Identity')(nn.Identity)

# --- 2. 编写全局构建助手函数 ---
def build_activation(cfg):
    """
    一个更加健壮的构建函数，适应你的 Pydantic 配置逻辑
    cfg 可能是: "PReLU" 或 {"type": "LeakyReLU", "negative_slope": 0.1}
    """
    if cfg is None:
        return nn.Identity()
    
    # 利用你之前写好的 build_from_cfg
    return build_from_cfg(cfg, ACTIVATION_REGISTRY)