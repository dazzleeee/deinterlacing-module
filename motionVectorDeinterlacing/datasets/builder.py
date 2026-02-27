import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from motionVectorDeinterlacing.models.registry import DATASET_REGISTRY, build_from_cfg
from .reds_dataset import RedsDeintDataset # 触发注册

DATASET_REGISTRY.register('RedsDeintDataset')(RedsDeintDataset)

def build_dataloader(cfg_dataset, world_size, rank):
    dataset = build_from_cfg(cfg_dataset, DATASET_REGISTRY)
    
    # DDP 核心：让不同显卡吃到不同的数据切片
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(cfg_dataset.get('split') == 'train'))
    else:
        sampler = None
        
    dataloader = DataLoader(
        dataset,
        batch_size=cfg_dataset.get('batch_size', 1),
        shuffle=(sampler is None and cfg_dataset.get('split') == 'train'),
        sampler=sampler,
        num_workers=cfg_dataset.get('num_workers', 4),
        pin_memory=True
    )
    return dataloader