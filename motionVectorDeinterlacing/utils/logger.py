import os
import time

def get_root_logger(args):
    # 1. 基础实验目录，例如 work_dirs/unified_mvsr_v1_base
    exp_dir = os.path.join('work_dirs', args.exp_name)
    
    # 2. 生成时间戳，例如 20260212_103000
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    # 3. 拼接最终目录
    log_dir = os.path.join(exp_dir, timestamp)
    
    # 4. 自动创建文件夹 (如果有父级目录不存在，会自动创建)
    os.makedirs(log_dir, exist_ok=True)
    
    # 5. 创建 'latest' 软链接 (可选，方便快速找到最新实验)
    # 先删除旧的 link
    symlink_path = os.path.join(exp_dir, 'latest')
    if os.path.islink(symlink_path):
        os.remove(symlink_path)
    # 指向最新的 timestamp 文件夹
    os.symlink(timestamp, symlink_path)
    
    print(f"实验日志将保存在: {log_dir}")
    return log_dir

# 在 train.py 里调用
# args.exp_name 来自命令行参数
log_dir = get_root_logger(args) 
# 然后把 log_dir 传给 Checkpoint 保存函数