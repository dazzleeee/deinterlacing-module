import os, argparse, yaml
import torch
import torch.distributed as dist
import torch.nn.functional as F
import glob
import time
import random  
import numpy as np 
import cv2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import timedelta
import torch.nn as nn
from torchvision import transforms

# 假设这些是你自己的模块
from mv_vsr import MVSR
from reds import RedsDeintDataset 
from utils import MVWarp

# --- [类] SD 降质模拟 ---
class SDDegradation:
    def __init__(self, device):
        self.device = device
        # 色彩抖动：亮度、对比度、饱和度小幅随机
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

    def apply(self, imgs):
        """
        imgs: [B, T, 3, H, W]  Normalized 0~1
        """
        B, T, C, H, W = imgs.shape
        imgs = imgs.view(B * T, C, H, W) # 合并做变换

        # 1. 色彩抖动 (Color Jitter) - 50% 概率
        if random.random() < 0.5:
            imgs = self.color_jitter(imgs)

        # 2. 加入高斯噪声 (Gaussian Noise)
        noise_level = random.uniform(0.01, 0.05)
        noise = torch.randn_like(imgs) * noise_level
        imgs = imgs + noise

        # 3. 随机高斯模糊 (Blur) - 30% 概率
        if random.random() < 0.3:
            kernel = torch.ones(1, 1, 3, 3, device=imgs.device) / 9.0
            kernel = kernel.repeat(C, 1, 1, 1)
            imgs = F.conv2d(imgs, kernel, padding=1, groups=C)

        return imgs.view(B, T, C, H, W).clamp(0, 1)

# --- [Loss] 乒乓时域 Loss ---
class PingPongTemporalLoss(nn.Module):
    def __init__(self, motion_sensitivity=0.5): 
        super().__init__()
        self.mvwarp = MVWarp()
        self.motion_sensitivity = motion_sensitivity

    def forward(self, sr_imgs, refined_flows):
        loss = 0.0
        valid = 0
        B, T, C, H, W = sr_imgs.shape
        
        start_t = 2
        end_t = T - 2
        
        if start_t >= end_t:
             if T >= 5: indices = [2]
             else: indices = []
        else:
             indices = range(start_t, end_t)

        for t in indices:
            curr = sr_imgs[:, t]
            
            # 假设 T 处静止，那么 T-2 和 T+2 也应该是静止的
            flow_mag_lr = torch.sum(refined_flows[:, t]**2, dim=1, keepdim=True)
            flow_mag_hr = F.interpolate(flow_mag_lr, size=(H, W), mode='bilinear', align_corners=False)
            
            # 计算静态权重
            static_weight = torch.exp(-self.motion_sensitivity * flow_mag_hr)
            
            prev = sr_imgs[:, t-2]
            loss += (torch.abs(curr - prev) * static_weight).mean()
            
            next_frame = sr_imgs[:, t+2]
            loss += (torch.abs(curr - next_frame) * static_weight).mean()
            
            valid += 2

        return loss / (valid + 1e-8)

# --- [Loss] 掩码时域 Loss ---
class MaskedTemporalLoss(nn.Module):
    def __init__(self): 
        super().__init__()
        self.mvwarp = MVWarp()

    def forward(self, sr_imgs, refined_flows, conf_masks):
        loss_temporal = 0.0
        valid_frames = 0
        B, T, _, H, W = sr_imgs.shape
        _, _, _, h, w = refined_flows.shape
        
        scale_x = W / w
        scale_y = H / h

        for t in range(1, T):
            curr_sr = sr_imgs[:, t]
            prev_sr = sr_imgs[:, t-1] 
            flow = refined_flows[:, t]
            mask = conf_masks[:, t] 

            flow_full = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
            flow_full = flow_full.clone() 
            flow_full[:, 0, :, :] *= scale_x
            flow_full[:, 1, :, :] *= scale_y

            mask_full = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
            warped_prev = self.mvwarp(prev_sr, flow_full)
            
            diff = torch.abs(curr_sr - warped_prev)
            weighted_diff = diff * mask_full 
            loss_temporal += weighted_diff.mean()
            valid_frames += 1

        if valid_frames > 0:
            return loss_temporal / valid_frames
        else:
            return torch.tensor(0.0, device=sr_imgs.device, requires_grad=True)

# --- [Loss] 光流 Loss ---
class FlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mvwarp = MVWarp()

    def gradient(self, data):
        D_dy = data[:, :, 1:] - data[:, :, :-1]
        D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
        return D_dx, D_dy

    def photometric_loss(self, curr_img, prev_img, flow):
        warped_prev = self.mvwarp(prev_img, flow)
        diff = curr_img - warped_prev
        loss = torch.sqrt(diff * diff + 1e-6)
        return loss.mean()

    def smooth_loss(self, flow, img):
        flow_dx, flow_dy = self.gradient(flow)
        img_dx, img_dy = self.gradient(img)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True))
        loss_x = torch.abs(flow_dx) * weights_x
        loss_y = torch.abs(flow_dy) * weights_y
        return loss_x.mean() + loss_y.mean()

    def forward(self, refined_flows, imgs):
        B, T, _, H, W = refined_flows.shape
        loss_smooth = 0.0
        loss_photo = 0.0
        valid_steps = 0
        for t in range(T):
            flow = refined_flows[:, t]
            curr_img = imgs[:, t]
            loss_smooth += self.smooth_loss(flow, curr_img)
            if t > 0: 
                prev_img = imgs[:, t-1] 
                loss_photo += self.photometric_loss(curr_img, prev_img, flow)
                valid_steps += 1
        return loss_photo / (valid_steps + 1e-8), loss_smooth / T

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# --- 工具函数 ---
def load_network(net, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    is_current_model_ddp = hasattr(net, 'module')
    
    if is_current_model_ddp:
        current_out_conv = net.module.mv_refiner.out_conv
    else:
        current_out_conv = net.mv_refiner.out_conv
    
    target_weight_shape = current_out_conv.weight.shape 
    target_bias_shape = current_out_conv.bias.shape     

    print(f"--- Loading weights from {load_path} ---")

    for k, v in state_dict.items():
        if is_current_model_ddp and not k.startswith('module.'):
            name = 'module.' + k
        elif not is_current_model_ddp and k.startswith('module.'):
            name = k.replace('module.', '')
        else:
            name = k

        if 'mv_refiner.out_conv' in name:
            if 'weight' in name:
                if v.shape[0] == 3 and target_weight_shape[0] == 4:
                    print(f"⚠️ [Adapter] 修复权重 {name}: 3通道 -> 4通道")
                    new_w = torch.zeros(target_weight_shape, device=v.device, dtype=v.dtype)
                    new_w[:3, :, :, :] = v 
                    new_state_dict[name] = new_w
                    continue
            if 'bias' in name:
                if v.shape[0] == 3 and target_bias_shape[0] == 4:
                    print(f"⚠️ [Adapter] 修复偏置 {name}: 3通道 -> 4通道")
                    new_b = torch.zeros(target_bias_shape, device=v.device, dtype=v.dtype)
                    new_b[:3] = v 
                    new_state_dict[name] = new_b
                    continue

        new_state_dict[name] = v

    net.load_state_dict(new_state_dict, strict=False)
    return checkpoint

def psnr_torch_perframe(sr, gt):
    mse = F.mse_loss(sr, gt, reduction='none').mean(dim=[2, 3, 4])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean()

def charbonnier_loss(pred, target, eps=1e-6):
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps)
    return loss.mean()

# --- 分布式验证函数 (仅计算 PSNR，不生成视频) ---
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.float16):
    model.eval()
    local_psnr_sum = 0.0
    local_count = 0

    for imgs, gts, mv_fwd, mv_bwd, fids, _, _ in loader:
        imgs   = imgs.to(device, non_blocking=True)
        gts    = gts.to(device, non_blocking=True)
        mv_fwd = mv_fwd.to(device, non_blocking=True)
        mv_bwd = mv_bwd.to(device, non_blocking=True)
        fids   = fids.to(device, non_blocking=True)
        
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(imgs, mv_fwd, mv_bwd, fids)
            if isinstance(outputs, (tuple, list)):
                sr = outputs[0]
            else:
                sr = outputs
            
            sr = sr.float()
            gts = gts.float()

        batch_size = imgs.size(0)
        local_psnr_sum += float(psnr_torch_perframe(sr, gts)) * batch_size
        local_count += batch_size

    metrics = torch.tensor([local_psnr_sum, local_count], dtype=torch.float64, device=device)
    
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
    total_psnr_sum = metrics[0].item()
    total_count = metrics[1].item()

    model.train()
    
    if total_count == 0: return 0.0
    return total_psnr_sum / total_count

# --- 主函数 ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_rank", type=int, default=-1) 
    ap.add_argument("--cfg", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=8) 
    ap.add_argument("--out_dir",   default="out_mvsr")
    
    # 数据集路径
    ap.add_argument("--reds_root", type=str, default=None) 
    ap.add_argument("--val_reds_root", type=str, default=None)
    ap.add_argument("--flows_root", type=str, default=None)
    ap.add_argument("--val_flows_root", type=str, default=None)
    
    # 训练参数
    ap.add_argument("--scale", type=int, default=2) 
    ap.add_argument("--seq_len", type=int, default=5)
    ap.add_argument("--crop_lr", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ckpt", action="store_true", help="Resume")
    ap.add_argument("--clip_grad", type=float, default=1.0, help="梯度裁剪阈值")
    
    # WandB 参数
    ap.add_argument("--wandb_project", type=str, default="MVSR_Deinterlace")
    ap.add_argument("--exp_name", type=str, default=None)

    args = ap.parse_args()
    
    # --- DDP 初始化 ---
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    is_ddp = args.local_rank != -1
    
    if is_ddp:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cuda')

    # 初始化降质工具
    sd_degrader = SDDegradation(device) 

    if args.cfg:
        with open(args.cfg, "r") as f:
            cfg = yaml.safe_load(f) or {}
            def apply_dict(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        apply_dict(v)
                    else:
                        if hasattr(args, k): setattr(args, k, v)
            apply_dict(cfg)

    if args.local_rank <= 0 and HAS_WANDB:
        run_name = args.exp_name if args.exp_name else f"run_{int(time.time())}"
        wandb.init(project=args.wandb_project, name=run_name, config=args)
        print(f"--- WandB Initialized: {run_name} ---")

    # --- 数据集 ---
    train_ds = RedsDeintDataset(
            reds_root=args.reds_root,
            codec_root=args.flows_root,
            split="train",
            seq_len=args.seq_len,
            crop_lr=args.crop_lr,
            augment=True,
    )
    
    if is_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        shuffle_loader = False 
    else:
        train_sampler = None
        shuffle_loader = True

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=shuffle_loader, 
        sampler=train_sampler, num_workers=args.num_workers, 
        pin_memory=True, drop_last=True, prefetch_factor=2
    )

    val_loader = None
    if args.val_reds_root: 
        val_ds = RedsDeintDataset(
            reds_root=args.val_reds_root,
            codec_root=args.val_flows_root,
            split="val",
            seq_len=args.seq_len,
            crop_lr=None,
            augment=False,
        )
        if is_ddp:
            val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
        else:
            val_sampler = None

        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, sampler=val_sampler, num_workers=4, pin_memory=True
        )

    # --- 模型与优化器 ---
    model = MVSR(mid=64, blocks=15, scale=args.scale).to(device)
    flow_criterion = FlowLoss().to(device)
    temporal_criterion = MaskedTemporalLoss().to(device)
    pingpong_criterion = PingPongTemporalLoss(motion_sensitivity=10.0).to(device)
    
    if is_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler()
    
    start_epoch = 1
   
    # --- 断点续训 ---
    if args.ckpt:
        ckpt_files = sorted(glob.glob(os.path.join(args.out_dir, "epoch_*.pth")))
        if len(ckpt_files) > 0:
            latest_ckpt_path = ckpt_files[-1]
            if args.local_rank <= 0:
                print(f"--- [RESUME] Loading checkpoint from {latest_ckpt_path} ---")
            
            checkpoint = load_network(model, latest_ckpt_path, device)
            # 这里如果不加载 optimizer，建议重置 start_epoch 或者谨慎处理
            # 你的逻辑是不加载 optimizer，所以 start_epoch = checkpoint["epoch"] + 1 是对的
            start_epoch = checkpoint["epoch"] + 1
            print("✅ Model weights loaded. Training with FRESH optimizer.")
        else:
            print("⚠️ No checkpoint found but --ckpt was specified.")
    
    if args.local_rank <= 0:
        os.makedirs(args.out_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.out_dir)
    else:
        writer = None

    best_psnr = -1.0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # --- 训练循环 ---
    for epoch in range(start_epoch, args.epochs + 1):
        if is_ddp:
            train_sampler.set_epoch(epoch)
            
        model.train()
        
        if args.local_rank <= 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = train_loader
            
        for imgs, gts, mv_fwd, mv_bwd, fids, _, _ in pbar:
            imgs   = imgs.to(device, non_blocking=True)
            gts    = gts.to(device, non_blocking=True)
            mv_fwd = mv_fwd.to(device, non_blocking=True)
            mv_bwd = mv_bwd.to(device, non_blocking=True)
            fids   = fids.to(device, non_blocking=True)
            
            # 1. 在线降质 (只影响输入，GTS保持干净)
            with torch.no_grad():
                imgs = sd_degrader.apply(imgs)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                # 2. 前向传播
                sr, refined_flows, conf_masks = model(imgs, mv_fwd, mv_bwd, fids)
                total_loss = 0.0
                loss_dict = {} 

                # 3. 计算 Loss
                l_cfg = cfg.get('loss_config', {})

                if l_cfg.get('use_sr', True):
                    l_sr = charbonnier_loss(sr, gts)
                    total_loss += l_sr * l_cfg.get('w_sr', 1.0)
                    loss_dict['loss/sr'] = l_sr.item()

                if l_cfg.get('use_temp_short', True):
                    l_t_s = temporal_criterion(sr, refined_flows, conf_masks)
                    total_loss += l_t_s * l_cfg.get('w_temp_short', 0.05)
                    loss_dict['loss/temp_short'] = l_t_s.item()

                if l_cfg.get('use_temp_long', True):
                    l_t_l = pingpong_criterion(sr, refined_flows)
                    total_loss += l_t_l * l_cfg.get('w_temp_long', 0.5)
                    loss_dict['loss/temp_long'] = l_t_l.item()

                if l_cfg.get('use_flow', True):
                    l_photo, l_smooth = flow_criterion(refined_flows, imgs)
                    l_f = l_photo * l_cfg.get('w_photo', 0.01) + l_smooth * l_cfg.get('w_smooth', 0.001)
                    total_loss += l_f
                    loss_dict['loss/flow_photo'] = l_photo.item()
                    loss_dict['loss/flow_smooth'] = l_smooth.item()

            # 4. 反向传播与裁剪
            scaler.scale(total_loss).backward()
            
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            
            scaler.step(opt)
            scaler.update()

            # 全局步数 (修正逻辑，防止 WandB 报错)
            current_step = (epoch - 1) * len(train_loader) + (pbar.n if hasattr(pbar, 'n') else 0)

            # 5. 日志记录
            if args.local_rank <= 0 and HAS_WANDB:
                loss_dict['loss/total'] = total_loss.item()
                # 确保在这里记录，step 参数是唯一的
                wandb.log(loss_dict, step=current_step)
                
            if args.local_rank <= 0:
                loss_val = total_loss.item() # <--- 修正变量名错误
                pbar.set_postfix(loss=f"{loss_val:.4f}")
                writer.add_scalar("train/loss", loss_val, current_step)
        
        # --- 8. 验证 (只算 PSNR) ---
        if val_loader:
            val_psnr = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
            
            if args.local_rank <= 0:
                print(f"[Epoch: {epoch}] Validation PSNR: {val_psnr:.3f} dB")
                writer.add_scalar("val/psnr", val_psnr, epoch)
                
                if HAS_WANDB and wandb.run is not None:
                    wandb.log({"val/psnr": val_psnr, "epoch": epoch}, step=current_step)
                
                # 保存模型
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
                    print(f"Best PSNR: ({best_psnr:.3f} dB) saved.")

                ckpt = os.path.join(args.out_dir, f"epoch_{epoch:04d}.pth")
                torch.save({
                    "epoch": epoch, 
                    "model": model.state_dict(), 
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict()
                }, ckpt)
    
        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()
    
    if args.local_rank <= 0 and HAS_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()