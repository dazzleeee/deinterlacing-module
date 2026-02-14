# models/registry.py

# 两个档案柜
ARCH_REGISTRY = {}
COMPONENT_REGISTRY = {}

def register_model(cls):
    ARCH_REGISTRY[cls.__name__] = cls
    return cls

def register_component(cls):
    COMPONENT_REGISTRY[cls.__name__] = cls
    return cls

def build_from_cfg(cfg, registry):
    """通用的构建函数"""
    name = cfg['type']
    args = cfg.get('args', {})
    return registry[name](**args)