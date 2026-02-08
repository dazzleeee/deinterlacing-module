# models/__init__.py

# 1. 导入注册中心的功能，方便外部直接 models.build_model
from .registry import build_model, ARCH_REGISTRY

# 2. 【核心】手动导入所有的模型文件
# 只有在这里 import 了，mvsr.py 里的 @register_model 装饰器才会运行
from .mvsr import MVSR

# 3. 如果以后你有其他模型，比如 basicvsr.py，也在这里写一行：
# from .basicvsr import BasicVSR

# 定义外部可见的接口
__all__ = ['build_model', 'ARCH_REGISTRY']