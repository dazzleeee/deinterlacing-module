# runners/train_runner.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # ✅ 引入 TensorBoard
import logging
import os

from motionVectorDeinterlacing.models.registry import ARCH_REGISTRY, LOSS_REGISTRY

from motionVectorDeinterlacing.datasets.builder import build_dataloader
from motionVectorDeinterlacing.utils.distributionUtil import is_master, get_dist_info
from motionVectorDeinterlacing.utils.ema import ModelEMA

class TrainRunner:
    def __init__(self, cfg, exp_dir):
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.rank, self.world_size = get_dist_info()
        self.device = torch.device(f'cuda:{self.rank}')
        
        # 1. 数据加载
        self.train_loader = build_dataloader(cfg.dataset['train'], self.world_size, self.rank)
        
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
                    
                   
                    
                    targets = {'hr': hr_targets}
                    loss_dict = self.loss_aggregator(outputs, targets, epoch)
                    total_loss = loss_dict['total_loss']
                
                # 反向传播
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # ✅ 步进：调度器更新学习率
                self.scheduler.step()
                
                # ✅ 步进：更新 EMA 影子模型的权重
                self.ema.update(self.model)
                
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

        if is_master() and self.writer is not None:
            self.writer.close()

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