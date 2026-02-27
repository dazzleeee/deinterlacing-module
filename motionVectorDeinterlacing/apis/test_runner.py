# runners/test_runner.py
import torch
import math
import argparse
import yaml
from tqdm import tqdm
from skimage.metrics import structural_similarity
from motionVectorDeinterlacing.models.registry import ARCH_REGISTRY
from motionVectorDeinterlacing.datasets.builder import build_dataloader

# ==========================================
# 补丁：支持嵌套字典的“点号”访问
# ==========================================
class AttrDict(dict):
    """Allows nested dictionary access via dot notation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

def calculate_psnr(img1, img2):
    """简单的 PSNR 计算，假设输入范围是 [0, 1]"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    输入 img1, img2 为 [3, H, W] 的 Tensor，范围 [0, 1]
    """
    # 转换为 HWC 格式的 numpy 数组
    im1_np = img1.cpu().numpy().transpose(1, 2, 0)
    im2_np = img2.cpu().numpy().transpose(1, 2, 0)
    # channel_axis=2 表示第3个维度是颜色通道，data_range=1.0 表示数据范围是 0~1
    return structural_similarity(im1_np, im2_np, data_range=1.0, channel_axis=2)

def test_runner(cfg, weight_path):
    print("📡 初始化 RealTimeMVDnet 评估流水线...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 构建模型与数据
    model = ARCH_REGISTRY.get('RealTimeMVDnet')(cfg.model).to(device)
    
    # 加载权重
    print(f"📥 Loading weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint.get('ema_state_dict', checkpoint['state_dict']))
    
    model.eval() # 开启评估模式 (触发返回单张量，并利用 h_fwd_cache)
    
    # 获取验证集的 DataLoader (替换了原先不存在的 dataset.test)
    test_loader = build_dataloader(cfg.dataset.val, world_size=1, rank=0)
    
    total_psnr = 0.0
    total_frames = 0
    total_ssim = 0.0
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            is_new_video = data.get('is_new_video', [False])[0]
            
            # 当切换到新测试视频时，必须清空网络记忆
            if is_new_video:
                model.reset_state()
            
            # 数据上卡
            imgs = data['lr'].to(device)
            hr_targets = data['hr'].to(device)
            mv_fwd = data['mv_fwd'].to(device)
            field_ids = data['field_ids'].to(device)
            
            # 推理
            sr_outs = model(imgs, mv_fwd, field_ids)
            
            B, T = sr_outs.shape[:2]
            for t in range(T):
                # ✅ 关键修复：计算前强制截断到 [0, 1] 范围，防止计算越界报错
                sr_clamped = torch.clamp(sr_outs[0, t], 0.0, 1.0)
                
                psnr = calculate_psnr(sr_clamped, hr_targets[0, t])
                ssim_val = calculate_ssim(sr_clamped, hr_targets[0, t])
                total_psnr += psnr
                total_ssim += ssim_val
                total_frames += 1

    avg_psnr = total_psnr / total_frames if total_frames > 0 else 0
    avg_ssim = total_ssim / total_frames if total_frames > 0 else 0 
    print(f"✅ 测试完成！共评估 {total_frames} 帧 | 平均 PSNR: {avg_psnr:.2f} dB | 平均 SSIM: {avg_ssim:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description='Test RealTimeMVDnet')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('-w', '--weight', type=str, required=True, help='Path to best_model.pth')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 解析 YAML 
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
    
    # 打上 AttrDict 补丁，让后续代码可以用 cfg.model 这种点号语法
    cfg = AttrDict(cfg_dict)
    
    # 启动评估
    test_runner(cfg, args.weight)