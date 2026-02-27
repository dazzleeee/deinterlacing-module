import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from motionVectorDeinterlacing.models.registry import ARCH_REGISTRY
from config.config_schema import MVDNetConfig
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Real-Time Streaming Inference for MVDNet')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('-w', '--weight', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('-i', '--input_video', type=str, required=True, help='Input interlace video')
    parser.add_argument('-o', '--output_video', type=str, default='output.mp4', help='Output HR video')
    parser.add_argument('--chunk_size', type=int, default=5, help='Sequence length for each forward pass')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Parse only the model configuration
    model_cfg = MVDNetConfig(**cfg_dict.get('model', {}))
    
    # [ A]: Initialize the model properly
    model = ARCH_REGISTRY.get('RealTimeMVDnet')(model_cfg).to(device)

    
    
    # [步骤 B]: 加载权重 (优先加载平滑过的 EMA 权重)
    checkpoint = torch.load(args.weight, map_location=device)
    model.load_state_dict(checkpoint.get('ema_state_dict', checkpoint['state_dict']))
    
    # [步骤 C]: ！！！极其重要：切换到推理模式 ！！！
    model.eval()
    
    # [步骤 D]: 🚀 触发重参数化魔法，融合多分支卷积！(必须在 eval() 之后调用)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
            
    # [步骤 E]: 唤醒状态机，清空历史缓存，准备迎接新视频流
    model.reset_state()

    # 2. 准备视频读写器
    # ... 后面的代码完全保持不变 ...
    # 2. 准备视频读写器
    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 这里的 height 是物理场的高度
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height * 2))

    buffer_imgs = []
    buffer_mvs = []
    buffer_fields = []
    
    print(f"Starting inference... Chunk size: {args.chunk_size}")

    # 3. 核心推流循环 (Streaming Loop)
    # 在不计算梯度的情况下运行，彻底释放显存
    with torch.no_grad():
        # 在 75 行 `with torch.no_grad():` 的下一行加上这几句：
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_time_ms = 0.0
        total_frames = 0
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            lr_field = frame_rgb 
            mv_fwd = np.zeros((2, height, width), dtype=np.float32) 
            field_id = frame_idx % 2 
            
            t_img = torch.from_numpy(lr_field).permute(2, 0, 1).float() / 255.0
            t_mv = torch.from_numpy(mv_fwd).float()
            
            buffer_imgs.append(t_img)
            buffer_mvs.append(t_mv)
            buffer_fields.append(field_id)
            
            if len(buffer_imgs) == args.chunk_size:
                b_imgs = torch.stack(buffer_imgs).unsqueeze(0).to(device)
                b_mvs = torch.stack(buffer_mvs).unsqueeze(0).to(device)
                b_fields = torch.tensor(buffer_fields).unsqueeze(0).to(device)
                start_event.record()
                
                sr_outs = model(b_imgs, b_mvs, b_fields)
                
                end_event.record()
                torch.cuda.synchronize() # 强制 CPU 等待 GPU 算完
                
                chunk_time = start_event.elapsed_time(end_event) # 毫秒
                total_time_ms += chunk_time
                total_frames += args.chunk_size
                
                # 打印当前 Chunk 每帧的平均耗时
                print(f"Inference Time per frame: {chunk_time / args.chunk_size :.2f} ms | FPS: {1000.0 / (chunk_time / args.chunk_size):.1f}")
               
                
                for t in range(args.chunk_size):
                    out_frame = sr_outs[0, t].permute(1, 2, 0).cpu().numpy()
                    out_frame = np.clip(out_frame * 255.0, 0, 255).astype(np.uint8)
                    # ✅ 写回视频前转换回 BGR：RGB -> BGR
                    out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                    out_writer.write(out_frame_bgr)
                
                buffer_imgs.clear()
                buffer_mvs.clear()
                buffer_fields.clear()
            
            frame_idx += 1

    # 处理视频结尾不够一个 Chunk 大小的剩余帧
    if len(buffer_imgs) > 0:
        b_imgs = torch.stack(buffer_imgs).unsqueeze(0).to(device)
        b_mvs = torch.stack(buffer_mvs).unsqueeze(0).to(device)
        b_fields = torch.tensor(buffer_fields).unsqueeze(0).to(device)
        sr_outs = model(b_imgs, b_mvs, b_fields)
        for t in range(len(buffer_imgs)):
            out_frame = sr_outs[0, t].permute(1, 2, 0).cpu().numpy()
            out_frame = np.clip(out_frame * 255.0, 0, 255).astype(np.uint8)
            out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            out_writer.write(out_frame_bgr) # ✅ 写入 BGR 格式
     
    cap.release()
    out_writer.release()
    print("Inference completed successfully!")

if __name__ == '__main__':
    main()