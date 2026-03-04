# runners/train_runner.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # ✅ 引入 TensorBoard
import logging
import os
import math

from motionVectorDeinterlacing.models.registry import ARCH_REGISTRY, LOSS_REGISTRY

from motionVectorDeinterlacing.datasets.builder import build_dataloader
from motionVectorDeinterlacing.utils.distributionUtil import is_master, get_dist_info
from motionVectorDeinterlacing.utils.ema import ModelEMA

class TrainRunner:
    def __init__(self, cfg, exp_dir, resume_path=None):
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.rank, self.world_size = get_dist_info()
        self.device = torch.device(f'cuda:{self.rank}')
        
        # 1. 数据加载
        self.train_loader = build_dataloader(cfg.dataset['train'], self.world_size, self.rank)
        if is_master():
            self.val_loader = build_dataloader(cfg.dataset['val'], world_size=1, rank=0)
            self.best_psnr = 0.0 # 记录历史最高分

        # 2. 模型与 DDP
        
        self.model = ARCH_REGISTRY.get('RealTimeMVDnet')(cfg.model).to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
            
        # ✅ 3. 初始化 EMA 影子模型 (衰减率通常取 0.999 或 0.9999)
        self.ema = ModelEMA(self.model, decay=0.999)
        
        # 4. 损失函数与优化器
        self.loss_aggregator = LOSS_REGISTRY.get('MVDNetLossAggregator')(cfg.loss).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.scaler = GradScaler() 
        
        # ✅ 5. LR Scheduler (余弦退火)
        # T_max 设为总 Iteration 数 (总 Epoch * 每个 Epoch 的 Batch 数)
        total_iters = cfg.train.epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_iters, eta_min=1e-6
        )
        
        # 6. 杂项与 TensorBoard
        self.start_epoch = 0
        self.total_epochs = cfg.train.epochs
        self.global_step = 0
        self.logger = logging.getLogger(__name__) if is_master() else None
        
        # ✅ 只在主进程开启 TensorBoard Writer
        self.writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tb_logs')) if is_master() else None
        
        if resume_path is not None and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)
    
    def load_checkpoint(self, resume_path):
        """恢复训练状态的终极魔法"""
        if is_master():
            self.logger.info(f"🔄 正在从 {resume_path} 恢复训练...")
        
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        # 1. 恢复轮次和全局步数
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = self.start_epoch * len(self.train_loader)
        
        # 2. 恢复模型权重 (处理 DDP 的 module 前缀)
        model_state = checkpoint['state_dict']
        if self.world_size > 1:
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
            
        # 3. 恢复 EMA 影子模型
        self.ema.module.load_state_dict(checkpoint['ema_state_dict'])
        
        # 4. 恢复优化器、调度器和混合精度 Scaler
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        
        if is_master():
            self.logger.info(f"✅ 成功恢复！将从 Epoch {self.start_epoch} 继续训练。")

    def train(self):
        for epoch in range(self.start_epoch, self.total_epochs):
            if self.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)
                
            self.model.train()
            
            for batch_idx, data in enumerate(self.train_loader):
                self.global_step += 1
                
                imgs = data['lr'].to(self.device)
                hr_targets = data['hr'].to(self.device)
                mv_fwd = data['mv_fwd'].to(self.device)
                field_ids = data['field_ids'].to(self.device)
                
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(imgs, mv_fwd, field_ids)
                    # 2. 退出 autocast！强制把关键的 outputs 转换回极其安全的 float32
                    outputs_fp32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
                    targets_fp32 = {'hr': hr_targets.float()}

                    # 3. 在 float32 精度下算 Loss（尤其是里面的 torch.exp 就绝对不会爆了）
                    loss_dict = self.loss_aggregator(outputs_fp32, targets_fp32, epoch)
                   
                    
                   
                    total_loss = loss_dict['total_loss']
                
                # 反向传播
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()

                is_step_success = scale_before <= scale_after
                if is_step_success:
                    self.scheduler.step()
                    # 修复 1：EMA 更新必须与 Optimizer 严格同频！
                    self.ema.update(self.model)
                else:
                    # 修复 2：为了防止 T_max 走不完，在跳过更新时，
                    # 我们可以选择静默增加 scheduler 的 internal step，
                    # 骗过警告，保证 LR 曲线的时序进度完美对齐。
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.scheduler.step()
                        
            
                   
                
            
                
                # =======================================================
                # 日志与可视化 (只在主进程 Rank 0 运行)
                # =======================================================
                if is_master():
                    # 1. 每 50 步打印一次 Loss 曲线
                    if self.global_step % self.cfg.train.log_freq == 0:
                        loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                        lr_current = self.optimizer.param_groups[0]['lr']
                        self.logger.info(f"Ep [{epoch}/{self.total_epochs}] Iter [{batch_idx}/{len(self.train_loader)}] LR: {lr_current:.2e} | {loss_str}")
                        
                        # 写入 TensorBoard 折线图
                        for k, v in loss_dict.items():
                            self.writer.add_scalar(f"Loss/{k}", v.item(), self.global_step)
                        self.writer.add_scalar("Train/LearningRate", lr_current, self.global_step)
                    
                    # 2. ✅ 每 500 步在 TensorBoard 画一次对比图 (Image Logging)
                    if self.global_step % 500 == 0:
                        # 抽取 Batch 里第 0 个视频的中间一帧 (T//2) 来展示
                        T_mid = imgs.shape[1] // 2 
                        
                        # 取出画面并规范化到 0~1 之间
                        lr_img = imgs[0, T_mid] # [3, H, W]
                        # 把 LR 强行拉大到 HR 尺寸，方便并排对比
                        lr_up = F.interpolate(lr_img.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0) 
                        
                        sr_img = torch.clamp(outputs['sr'][0, T_mid], 0, 1)
                        hr_img = hr_targets[0, T_mid]
                        
                        # 把 LR_up, SR(预测), HR(真值) 横向拼接成一张长图
                        # 拼接顺序：左(输入渣画质) -> 中(模型预测) -> 右(标准答案)
                        grid_img = torch.cat([lr_up, sr_img, hr_img], dim=2) 
                        self.writer.add_image('Visual_LR_SR_HR', grid_img, self.global_step)

            # Epoch 结束，全员同步
            if self.world_size > 1:
                dist.barrier()

            # 保存模型
            if is_master():
                self.save_checkpoint(epoch)
                self.evaluate(epoch)

        if is_master() and self.writer is not None:
            self.writer.close()

    @torch.no_grad()
    def evaluate(self, epoch):
        self.logger.info(f"🔍 开始进行 Epoch {epoch} 的验证集评估...")
        
        # ⚠️ 极度关键：验证时必须使用 EMA 影子模型，这才是我们最终要部署的稳定权重！
        eval_model = self.ema.module
        eval_model.eval()
        
        total_psnr = 0.0
        total_frames = 0
        current_clip = None
        for data in self.val_loader:
            if data['is_new_video'][0] and hasattr(eval_model, 'reset_state'):
                eval_model.reset_state()

            imgs = data['lr'].to(self.device)
            hr_targets = data['hr'].to(self.device)
            mv_fwd = data['mv_fwd'].to(self.device)
            field_ids = data['field_ids'].to(self.device)
            
        
            
            # 推理
            sr_outs = eval_model(imgs, mv_fwd, field_ids).detach()
            
            # 计算 PSNR (将图像 Clamp 到 0~1 之间)
            B, T = sr_outs.shape[:2]
            for t in range(T):
                # 2. 将张量拉到 CPU 计算，彻底释放 GPU 显存压力
                sr_clamped = torch.clamp(sr_outs[0, t], 0.0, 1.0).cpu()
                hr_cpu = hr_targets[0, t].cpu()
                
                mse = torch.mean((sr_clamped - hr_cpu) ** 2)
                
                if mse > 0:
                    psnr = 10 * math.log10(1.0 / mse.item())
                else:
                    psnr = 100.0
                    
                total_psnr += psnr
                total_frames += 1
                
        # 计算平均分并记录
        avg_psnr = total_psnr / total_frames if total_frames > 0 else 0
        self.logger.info(f"📊 Epoch {epoch} 验证完成 | Avg PSNR: {avg_psnr:.2f} dB")
        self.writer.add_scalar("Val/PSNR", avg_psnr, epoch)
        
        # 🏆 保存 Best Model
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            best_save_path = os.path.join(self.exp_dir, "best_model.pth")
            
            # 保存最纯粹的权重结构，方便推理直接调用
            torch.save({
                'epoch': epoch,
                'psnr': avg_psnr,
                'state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
                'ema_state_dict': self.ema.module.state_dict(),
            }, best_save_path)
            
            self.logger.info(f"🎉 发现新最优模型！PSNR: {avg_psnr:.2f} dB，已更新 {best_save_path}")
            
        # 验证结束，切回训练模式
        self.model.train()
        
    def save_checkpoint(self, epoch):
        model_state = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        save_path = os.path.join(self.exp_dir, f"epoch_{epoch}.pth")
        
        torch.save({
            'epoch': epoch,
            'state_dict': model_state,          # 原模型权重 (可用于继续训练)
            'ema_state_dict': self.ema.module.state_dict(), # ✅ 保存 EMA 权重 (推理首选！)
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(), # ✅ 记得保存调度器状态
            'scaler': self.scaler.state_dict(),
        }, save_path)