import argparse
import cv2
import torch
import numpy as np
import os
from motionVectorDeinterlacing.models.registry import ARCH_REGISTRY
from config.config_schema import MVDNetConfig
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Broadcast Streaming Inference')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('-w', '--weight', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('-i', '--input_video', type=str, required=True, help='Input stream (video file)')
    parser.add_argument('-o', '--output_video', type=str, default='output.mp4', help='Output stream')
    parser.add_argument('-m', '--mv_dir', type=str, required=True, help='Directory containing MV .npz files')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型与权重
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    model_cfg = MVDNetConfig(**cfg_dict.get('model', {}))
    model = ARCH_REGISTRY.get('RealTimeMVDnet')(model_cfg).to(device)
    
    checkpoint = torch.load(args.weight, map_location=device)
    model.load_state_dict(checkpoint.get('ema_state_dict', checkpoint['state_dict']))
    
    model.eval()
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    model.reset_state() # 初始化缓存

    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height * 2))

    sliding_window = [] # 容量固定为 3 的滑动窗口
    frame_idx = 0
    print("Broadcast streaming started. Awaiting signal...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. 接收到新帧，处理数据
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        field_id = frame_idx % 2 
        mv_path = os.path.join(args.mv_dir, f"{frame_idx:08d}_mv_fwd.npz")
        
        mv_fwd = np.zeros((2, height, width), dtype=np.float32)
        if os.path.exists(mv_path):
            f_arr = np.load(mv_path)["flow_fwd"].astype(np.float32)
            if f_arr.ndim == 3 and f_arr.shape[2] == 2: 
                f_arr = np.transpose(f_arr, (2, 0, 1))
            mv_fwd = f_arr
        
        t_img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        t_mv = torch.from_numpy(mv_fwd).float()
        
        # 2. 推入滑动窗口
        sliding_window.append({
            'img': t_img, 'mv': t_mv, 'fid': field_id
        })
        
        # 3. 窗口满了3帧，触发单步流式推理
        if len(sliding_window) == 3:
            b_imgs = torch.stack([x['img'] for x in sliding_window]).unsqueeze(0).to(device)
            b_mvs = torch.stack([x['mv'] for x in sliding_window]).unsqueeze(0).to(device)
            b_fids = torch.tensor([x['fid'] for x in sliding_window]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                sr_out = model.forward_stream_step(b_imgs, b_mvs, b_fids)
                
            # 处理并写出第 t 帧的输出
            out_frame = sr_out[0].permute(1, 2, 0).cpu().numpy()
            out_frame = np.clip(out_frame * 255.0, 0, 255).astype(np.uint8)
            out_writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))
            
            # 弹出队首老帧
            sliding_window.pop(0)
            
        frame_idx += 1

    # 4. 视频流断开（或者结束）时，处理最后两帧（Flush Pipeline）
    if len(sliding_window) > 0:
        print("Flushing the remaining frames in the pipeline...")
        last_item = sliding_window[-1]
        
        # 填充两次 dummy 帧将最后两帧顶出来
        for _ in range(2):
            sliding_window.append({
                'img': last_item['img'], 
                'mv': torch.zeros_like(last_item['mv']), 
                'fid': 1 - sliding_window[-1]['fid']
            })
            
            b_imgs = torch.stack([x['img'] for x in sliding_window]).unsqueeze(0).to(device)
            b_mvs = torch.stack([x['mv'] for x in sliding_window]).unsqueeze(0).to(device)
            b_fids = torch.tensor([x['fid'] for x in sliding_window]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                sr_out = model.forward_stream_step(b_imgs, b_mvs, b_fids)
            
            out_frame = sr_out[0].permute(1, 2, 0).cpu().numpy()
            out_writer.write(cv2.cvtColor(np.clip(out_frame * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            sliding_window.pop(0)

    cap.release()
    out_writer.release()
    print("Broadcast streaming sequence completed successfully!")

if __name__ == '__main__':
    main()