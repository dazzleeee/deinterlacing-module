import os
import torch
import torch.distributed as dist

def init_dist(local_rank=-1, backend='nccl'):
    """初始化 DDP 环境"""
    if dist.is_initialized():
        return
    
    # 适配 torchrun / python -m torch.distributed.launch
    if local_rank == -1:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)

def get_dist_info():
    """获取当前进程的 rank 和总进程数"""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def is_master():
    """判断是否为主进程 (用于打印日志和保存模型)"""
    rank, _ = get_dist_info()
    return rank == 0