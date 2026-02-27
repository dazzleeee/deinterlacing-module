# runners/test_runner.py
import torch
import math
from tqdm import tqdm
from motionVectorDeinterlacing.models.registry import ARCH_REGISTRY
from motionVectorDeinterlacing.datasets.builder import build_dataloader

def calculate_psnr(img1, img2):
    """简单的 PSNR 计算，假设输入范围是 [0, 1]"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def test_runner(cfg):
    print("📡 初始化 RealTimeMVDnet 评估流水线...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 构建模型与数据
    model = ARCH_REGISTRY.get('RealTimeMVDnet')(cfg.model).to(device)
    # 加载权重
    checkpoint = torch.load(cfg.test.weight_path, map_location=device)
    model.load_state_dict(checkpoint.get('ema_state_dict', checkpoint['state_dict']))
    
    model.eval() # 开启评估模式 (触发返回单张量，并利用 h_fwd_cache)
    
    # 获取测试集的 DataLoader (必须按视频序列、按时间顺序返回数据)
    # 注意：测试时 batch_size 必须为 1，以保证视频时序的严格对应！
    test_loader = build_dataloader(cfg.dataset.test, world_size=1, rank=0)
    
    total_psnr = 0.0
    total_frames = 0
    
    with torch.no_grad():
        # 这里假设 dataloader 返回的字典中，包含了一个标识视频切换的标志 'is_new_video'
        for data in tqdm(test_loader, desc="Testing"):
            is_new_video = data.get('is_new_video', [False])[0]
            
            # 当切换到新测试视频时，必须清空网络记忆！
            if is_new_video:
                model.reset_state()
            
            # 数据上卡: 形状 [B=1, T, C, H, W]
            imgs = data['lr'].to(device)
            hr_targets = data['hr'].to(device)
            mv_fwd = data['mv_fwd'].to(device)
            field_ids = data['field_ids'].to(device)
            
            # 推理 (直接吞吐整个 Chunk)
            sr_outs = model(imgs, mv_fwd, field_ids)
            
            # 计算 Chunk 内每一帧的 PSNR
            B, T = sr_outs.shape[:2]
            for t in range(T):
                psnr = calculate_psnr(sr_outs[0, t], hr_targets[0, t])
                total_psnr += psnr
                total_frames += 1

    avg_psnr = total_psnr / total_frames if total_frames > 0 else 0
    print(f"✅ 测试完成！共评估 {total_frames} 帧，平均 PSNR: {avg_psnr:.2f} dB")

# 可以在文件最下面写个小入口，方便独立运行
if __name__ == "__main__":
    # 解析参数和 config 的逻辑可以参考 train.py
    pass