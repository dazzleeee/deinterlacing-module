import os
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

def smooth_curve(points, factor=0.8):
    """
    [平滑算法] 使用指数移动平均 (EMA) 让 Loss 曲线看起来更丝滑，
    避免因为 batch 波动导致的锯齿状，适合汇报展示。
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def load_json_log(log_path):
    """
    解析日志文件 (假设是 JSON Lines 格式，每行一个 JSON)
    """
    log_dict = {'iters': [], 'loss': [], 'psnr': [], 'ssim': []}
    
    with open(log_dict_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                
                # 1. 提取 Iteration
                if 'iter' in data:
                    curr_iter = data['iter']
                elif 'epoch' in data:
                    curr_iter = data['epoch']
                else:
                    continue

                # 2. 提取 Loss (通常训练日志里有 loss)
                # 兼容不同的 loss 名字，比如 'l_pix', 'loss_pixel', 'total_loss'
                for key in data.keys():
                    if 'loss' in key:
                        log_dict['iters'].append(curr_iter)
                        log_dict['loss'].append(data[key])
                        break # 只取第一个找到的 loss
                
                # 3. 提取 Validation 指标 (通常验证日志里有 psnr)
                if 'psnr' in data:
                    # 注意：验证的 iter 可能比训练少，需要独立存
                    if 'val_iters' not in log_dict: log_dict['val_iters'] = []
                    log_dict['val_iters'].append(curr_iter)
                    log_dict['psnr'].append(data['psnr'])
                
                if 'ssim' in data:
                    if 'val_iters' not in log_dict: log_dict['val_iters'] = []
                    if len(log_dict['val_iters']) > len(log_dict.get('ssim', [])):
                         log_dict['ssim'].append(data['ssim'])

            except json.JSONDecodeError:
                continue
                
    return log_dict

def plot_log(log_path, out_path, smooth=True):
    # 1. 读取数据
    # 这里假设你的 logger.py 生成的是 json 文件。如果是纯文本，需要用正则解析。
    # 为了演示，我们先模拟读取数据的逻辑
    try:
        data = load_json_log(log_path)
    except:
        print("Json load failed, trying generic parsing or dummy data...")
        return

    iters = data['iters']
    losses = data['loss']
    
    # 创建画布
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 画 Loss (左轴) ---
    color = 'tab:red'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color=color)
    
    if smooth and len(losses) > 100:
        # 画淡色的原始数据
        ax1.plot(iters, losses, color=color, alpha=0.2)
        # 画深色的平滑数据
        smooth_loss = smooth_curve(losses, 0.9)
        ax1.plot(iters, smooth_loss, color=color, linewidth=2, label='Train Loss (Smoothed)')
    else:
        ax1.plot(iters, losses, color=color, label='Train Loss')
        
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 画 PSNR (右轴) ---
    if 'psnr' in data and len(data['psnr']) > 0:
        ax2 = ax1.twinx()  # 共享 x 轴
        color = 'tab:blue'
        ax2.set_ylabel('PSNR (dB)', color=color)
        
        val_iters = data.get('val_iters', iters[:len(data['psnr'])])
        ax2.plot(val_iters, data['psnr'], color=color, linewidth=2, marker='.', label='Val PSNR')
        ax2.tick_params(axis='y', labelcolor=color)

    # 标题和保存
    plt.title(f'Training Curve: {os.path.basename(log_path)}')
    fig.tight_layout()
    
    print(f"Saving curve to {out_path}")
    plt.savefig(out_path, dpi=150)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, help='Path to the json/log file')
    parser.add_argument('--out', type=str, default='train_curve.png', help='Output image path')
    args = parser.parse_args()
    
    if args.log:
        plot_log(args.log, args.out)
    else:
        print("Please provide a log file: python plot_curve.py --log work_dirs/train.log")