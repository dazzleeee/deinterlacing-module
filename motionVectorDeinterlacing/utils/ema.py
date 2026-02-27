import torch
import copy

class ModelEMA:
    """
    指数移动平均 (Exponential Moving Average)
    平滑模型参数，极大稳定视频生成的时域连贯性，提升 PSNR。
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # 剥离 DDP 外壳（如果有）
        m_base = model.module if hasattr(model, 'module') else model
        # 深度拷贝一份模型作为 EMA 影子
        self.module = copy.deepcopy(m_base).eval()
        
        # 影子模型不需要计算梯度
        for param in self.module.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        # 取出当前主模型的真实权重
        m_base = model.module if hasattr(model, 'module') else model
        
        # 核心 EMA 平滑更新公式
        for ema_p, m_p in zip(self.module.parameters(), m_base.parameters()):
            ema_p.data.mul_(self.decay).add_(m_p.data, alpha=1 - self.decay)