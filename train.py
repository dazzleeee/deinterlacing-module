import argparse
import yaml
import torch
import os
import shutil
import time
import collections.abc
from motionVectorDeinterlacing.utils.distributionUtil import init_dist, is_master
from motionVectorDeinterlacing.utils.logger import get_root_logger
from motionVectorDeinterlacing.apis.train_runner import TrainRunner
from config.config_schema import MVDNetConfig

def update_dict(d, u):
    """递归深度合并字典"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_config(file_path):
    """支持 _base_ 继承的 YAML 解析器"""
    with open(file_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 检查是否存在基类配置文件
    if '_base_' in cfg:
        base_files = cfg.pop('_base_')
        if isinstance(base_files, str):
            base_files = [base_files]
            
        base_cfg = {}
        for base_file in base_files:
            # 解析相对路径：以当前 yaml 文件所在目录为基准
            base_file_path = os.path.normpath(os.path.join(os.path.dirname(file_path), base_file))
            # 递归加载并合并基础配置
            base_cfg = update_dict(base_cfg, load_config(base_file_path))
        
        # 用当前文件中的专属配置去覆盖基础配置
        cfg = update_dict(base_cfg, cfg)
        
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train RealTimeMVDnet')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--local_rank', type=int, default=-1, help='For DDP')
    parser.add_argument('-r', '--resume', type=str, default=None, help='Path to checkpoint to resume from') # <-- 新增
    return parser.parse_args()

class AttrDict(dict):
    """Allows nested dictionary access via dot notation (Deep version)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            elif isinstance(value, list):
                # 递归处理列表里的字典
                self[key] = [AttrDict(i) if isinstance(i, dict) else i for i in value]

def main():
    args = parse_args()

    # 1. 唤醒 DDP
    init_dist(args.local_rank)
    
    # --- 实验目录与配置备份初始化 (只在主进程执行) ---
    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_dir = os.path.join('work_dirs', f"{exp_name}_{timestamp}")
    
    if is_master():
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(exp_dir, 'config.yaml'))
    
    # 2. 解析支持继承的配置！
    cfg_dict = load_config(args.config)
    
    logger = get_root_logger('MVDNet', log_file=os.path.join(exp_dir, 'train.log'))
    logger.info(f"Experiment Dir: {exp_dir}")

# --- FIX STARTS HERE ---
    # Wrap the root config to support dot-notation (cfg.train.lr)
    cfg = AttrDict(cfg_dict)
    # Parse specifically the model section into MVDNetConfig
    cfg.model = MVDNetConfig(**cfg_dict.get('model', {})) 
    # --- FIX ENDS HERE ---
    
    # 3. 启动 Runner
    runner = TrainRunner(cfg, exp_dir, resume_path=args.resume) 
    runner.train()

if __name__ == '__main__':
    main()