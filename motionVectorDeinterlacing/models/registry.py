# motionVectorDeinterlacing/registry.py

class Registry:
    def __init__(self, name):
        """
        初始化注册表
        Args:
            name (str): 注册表的名字，比如 'arch', 'backbone'，用于报错时提示
        """
        self._name = name
        self._module_dict = {}

    def register(self, name=None):
        """
        装饰器工厂。
        Args:
            name (str, optional): 指定别名。如果不填，默认用类名。
        """
        # --- 第一层：接收别名 ---
        def _register(cls):
            # --- 第二层：接收类 ---
            
            # 1. 确定 Key（名字）
            key = name if name is not None else cls.__name__
            
            # 2. 安全检查：防止你不小心注册了两个重名的类
            if key in self._module_dict:
                raise KeyError(f"{key} 已经在 {self._name} 注册表中存在了！请检查是否重名。")
            
            # 3. 登记
            self._module_dict[key] = cls
            
            # 4. 必须原样返回类，否则类就无法实例化了
            return cls

        return _register

    def get(self, key):
        """查表方法"""
        cls = self._module_dict.get(key)
        if cls is None:
            raise KeyError(f"在 {self._name} 注册表中找不到 '{key}'。请检查拼写或是否 import 了该文件。")
        return cls

# =========================================
# 实例化five个注册表对象
# =========================================
ARCH_REGISTRY = Registry('arch')
COMPONENT_REGISTRY = Registry('component')
ACTIVATION_REGISTRY = Registry('activation')
LOSS_REGISTRY = Registry('loss')
DATASET_REGISTRY = Registry('dataset')

def build_from_cfg(cfg, registry):
    """
    Args:
        cfg (dict | BaseModel | str): 配置字典/Pydantic模型/字符串别名
        registry (Registry): 注册表对象
    """
    # 1. 兼容 Pydantic (V2 版本用 model_dump, V1 版本用 dict)
    if hasattr(cfg, 'model_dump'):
        cfg = cfg.model_dump()
    elif hasattr(cfg, 'dict'):
        cfg = cfg.dict()

    # 2. 如果传来的是个字符串（比如 cfg='ReLU'），自动包装成字典
    if isinstance(cfg, str):
        cfg = {'type': cfg}

    if not isinstance(cfg, dict):
        raise TypeError(f"cfg 必须是 字典/Pydantic模型/字符串，但收到了 {type(cfg)}")
    
    if 'type' not in cfg:
        raise KeyError(f"配置中缺少 'type' 字段 (用于查表导航): {cfg}")
    
    args = cfg.copy()
    obj_type = args.pop('type')
    obj_cls = registry.get(obj_type)
    
    return obj_cls(**args)