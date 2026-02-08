import os, glob, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def imread_rgb(p):
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_tensor01(img):
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

class RedsDeintDataset(Dataset):
    """
    初始版本 Dataset: 
    1. 连续加载帧长度为 seq_len
    2. 步长为 1 滑动取样
    """
    def __init__(self, root_dir, split,
                 seq_len=15, crop_lr=None, augment=False,
                 img_tmpl="{:08d}.png"):
        
        self.seq_len = seq_len 
        self.crop_lr = crop_lr
        self.augment = augment
        self.img_tmpl = img_tmpl
        if split == 'train':
            # 训练集对应这两个
            self.gt_root = os.path.join(root_dir, 'REDS_data_GT')
            self.codec_root = os.path.join(root_dir, 'outputs_interlaced')
        else:
            # 验证集/测试集对应这两个
            self.gt_root = os.path.join(root_dir, 'REDS_data_GT_val')
            self.codec_root = os.path.join(root_dir, 'outputs')
        if not os.path.exists(self.gt_root):
            raise RuntimeError(f"GT 路径不存在: {self.gt_root}")
        if not os.path.exists(self.codec_root):
            raise RuntimeError(f"LR/MV 路径不存在: {self.codec_root}")


        # 自动搜索所有 clip 文件夹
        self.clips = sorted([
            d for d in os.listdir(self.codec_root)
            if os.path.isdir(os.path.join(self.codec_root, d))
        ])

        self.samples = []
        for clip in self.clips:
            lr_dir = os.path.join(self.codec_root, clip, "lr")
            lr_frames = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
            total_frames = len(lr_frames)
            
            # 🔥 初始版本：需求帧数即为 seq_len
            required_frames = self.seq_len 
            
            if total_frames >= required_frames:
                # 步长为 1，确保覆盖所有可能的起始点
                for s in range(0, total_frames - required_frames + 1):
                    self.samples.append((clip, s))

        print(f"Dataset loaded: {len(self.samples)} samples from {len(self.clips)} clips.")

    def _load_seq(self, clip, start):
        lr_dir   = os.path.join(self.codec_root, clip, "lr")
        gt_dir   = os.path.join(self.gt_root, clip)
        mvf_dir  = os.path.join(self.codec_root, clip, "mv_fwd")
        meta_dir = os.path.join(self.codec_root, clip, "meta")

        imgs_lr, imgs_gt = [], []
        mv_fwd_list = []

        fid_path = os.path.join(meta_dir, "field_ids.npy")
        all_fids = np.load(fid_path)

        # 🔥 初始版本：加载范围为 [start, start + seq_len)
        indices = [start + i for i in range(self.seq_len)]
        seq_fids = all_fids[indices] 

        for t_idx in indices:
            lr_fn = self.img_tmpl.format(t_idx)
            gt_fn = self.img_tmpl.format(t_idx)
            
            lr_path = os.path.join(lr_dir, lr_fn)
            gt_path = os.path.join(gt_dir, gt_fn)
            
            imgs_lr.append(imread_rgb(lr_path))
            imgs_gt.append(imread_rgb(gt_path))
            
            H_lr, W_lr, _ = imgs_lr[-1].shape
            base_name = os.path.splitext(lr_fn)[0] 

            # 加载前向 MV
            p_fwd = os.path.join(mvf_dir, f"{base_name}_mv_fwd.npz")
            if os.path.exists(p_fwd):
                f = np.load(p_fwd)["flow_fwd"].astype(np.float32)
                if f.ndim == 3 and f.shape[2] == 2: 
                    f = np.transpose(f, (2, 0, 1))
                mvf = f
            else:
                mvf = np.zeros((2, H_lr, W_lr), np.float32)
            mv_fwd_list.append(mvf)

        return (
            np.stack(imgs_lr, axis=0),      
            np.stack(imgs_gt, axis=0),      
            np.stack(mv_fwd_list, axis=0),  
            None, 
            seq_fids                        
        )

    def _random_crop(self, lr, gt, mv_fwd, mv_bwd=None):
        T, H_in, W, _ = lr.shape 
        crop_h = self.crop_lr 
        crop_w = self.crop_lr

        if crop_h is None or H_in < crop_h or W < crop_w:
            return lr, gt, mv_fwd, mv_bwd

        y0 = random.randint(0, H_in - crop_h)
        x0 = random.randint(0, W - crop_w)

        lr = lr[:, y0 : y0+crop_h, x0 : x0+crop_w, :]
        mv_fwd = mv_fwd[:, :, y0 : y0+crop_h, x0 : x0+crop_w]
        
        gt_y0 = y0 * 2
        gt_crop_h = crop_h * 2
        gt = gt[:, gt_y0 : gt_y0+gt_crop_h, x0 : x0+crop_w, :]

        return lr, gt, mv_fwd, mv_bwd

    def __getitem__(self, idx):
        clip, s = self.samples[idx]
        lr, gt, mv_fwd, _, fids = self._load_seq(clip, s)

        if self.crop_lr is not None:
            lr, gt, mv_fwd, _ = self._random_crop(lr, gt, mv_fwd, None)
        
        if self.augment and random.random() < 0.5:
            lr = lr[:, :, ::-1]
            gt = gt[:, :, ::-1]
            mv_fwd = mv_fwd[:, :, :, ::-1]
            mv_fwd[:, 0] *= -1.0 

        imgs = torch.stack([to_tensor01(im) for im in lr], dim=0)  
        gts  = torch.stack([to_tensor01(im) for im in gt], dim=0)  
        mv_fwd_t = torch.from_numpy(np.ascontiguousarray(mv_fwd)).float()
        fids_t = torch.from_numpy(np.ascontiguousarray(fids)).long()
        
        # 返回: [T, 3, H, W]
        return imgs, gts, mv_fwd_t, torch.zeros_like(mv_fwd_t), fids_t, clip, s

    def __len__(self):
        return len(self.samples)
    
if __name__ == "__main__":
# 1. 初始化
    ds = RedsDeintDataset(root_dir="/home/sihanuo/scratch/SihanZhangData", split='train', seq_len=5)
    
    # 2. 取第一组数据
    # imgs: [T, 3, H, W], gts: [T, 3, 2H, W]
    imgs, gts, _, _, _, clip, s = ds[0]
    
    # 3. 取出第一帧进行比对 (Tensor -> Numpy)
    # 取出第一帧 [3, H, W] -> 换轴 [H, W, 3] -> 转回 0-255
    lr_0 = (imgs[0].permute(1, 2, 0).numpy() * 255).astype('uint8')
    gt_0 = (gts[0].permute(1, 2, 0).numpy() * 255).astype('uint8')
    
    # 4. 把 540 高的 LR 强行拉伸到 1080 高
    h_gt, w_gt = gt_0.shape[:2]
    lr_0_resized = cv2.resize(lr_0, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)
    
    # 5. 左右拼接并保存
    # 左边是原始 GT，右边是你的 LR 拉伸版
    canvas = np.hstack([gt_0, lr_0_resized])
    
    # 注意：cv2 保存需要转回 BGR
    cv2.imwrite("debug_alignment.png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"验证图已保存至 debug_alignment.png，请检查画面是否对齐。")