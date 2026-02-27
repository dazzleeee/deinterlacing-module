import os
import glob
import random
import numpy as np
import cv2
import torch
import lmdb
import pickle
import io
from torch.utils.data import Dataset

def to_tensor01(img):
    """将 numpy array [H, W, C] 转换为张量 [C, H, W] 并归一化到 0~1"""
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

class RedsDeintDataset(Dataset):
    """
    MVDNet 专属 Dataset (支持极速 LMDB 引擎)
    """
    def __init__(self, root_dir, split='train',
                 seq_len=5, crop_lr=None, augment=False,
                 img_tmpl="{:08d}.png", use_lmdb=True, **kwargs):  # ✅ 新增 use_lmdb 开关
        
        self.split = split
        self.seq_len = seq_len 
        self.crop_lr = crop_lr
        self.augment = augment
        self.img_tmpl = img_tmpl
        self.use_lmdb = use_lmdb
        
        # 懒加载环境变量，防止 PyTorch 多进程死锁
        self.env_codec = None
        self.env_gt = None
        
        # 1. 目录路由
        if split == 'train':
            self.gt_root = os.path.join(root_dir, 'REDS_data_GT.lmdb' if use_lmdb else 'REDS_data_GT')
            # 💡 关键修改：改为你在 scratch 下打包出的真实文件夹名
            self.codec_root = os.path.join(root_dir, 'REDS_processed.lmdb' if use_lmdb else 'REDS_processed')
        else:
            self.gt_root = os.path.join(root_dir, 'REDS_data_GT_val.lmdb' if use_lmdb else 'REDS_data_GT_val')
            # 💡 关键修改：改为验证集真实文件夹名
            self.codec_root = os.path.join(root_dir, 'REDS_processed_val.lmdb' if use_lmdb else 'REDS_processed_val')
            
        if not os.path.exists(self.gt_root) or not os.path.exists(self.codec_root):
            raise RuntimeError(f"数据路径不存在: {self.gt_root} 或 {self.codec_root}")

        self.samples = []
        
        # 2. 获取数据列表 (通过 LMDB 的 meta_info 或 传统 glob)
        if self.use_lmdb:
            # ✅ 从 LMDB 打包时生成的 pickle 文件中读取所有 keys
            meta_path = os.path.join(self.codec_root, 'meta_info.pkl')
            with open(meta_path, 'rb') as f:
                keys = pickle.load(f)
            
            # 提取出所有的 clip 名字 (例如从 '000/lr/00000000.png' 提取出 '000')
            lr_keys = [k for k in keys if '/lr/' in k]
            self.clips = sorted(list(set([k.split('/')[0] for k in lr_keys])))
            
            # 计算每个 clip 有多少帧
            for clip in self.clips:
                clip_frames = [k for k in lr_keys if k.startswith(f"{clip}/lr/")]
                total_frames = len(clip_frames)
                if total_frames >= self.seq_len:
                    for s in range(0, total_frames - self.seq_len + 1):
                        self.samples.append((clip, s))
        else:
            # 传统文件夹遍历方式
            self.clips = sorted([d for d in os.listdir(self.codec_root) if os.path.isdir(os.path.join(self.codec_root, d))])
            for clip in self.clips:
                lr_dir = os.path.join(self.codec_root, clip, "lr")
                total_frames = len(glob.glob(os.path.join(lr_dir, "*.png")))
                if total_frames >= self.seq_len:
                    for s in range(0, total_frames - self.seq_len + 1):
                        self.samples.append((clip, s))

        print(f"[{split.upper()}] Dataset loaded: {len(self.samples)} samples from {len(self.clips)} clips. (LMDB: {self.use_lmdb})")

    def _init_lmdb(self):
        """懒加载：只在第一个 batch 被读取时，由当前 Worker 进程打开 LMDB 环境"""
        if self.env_codec is None:
            self.env_codec = lmdb.open(self.codec_root, readonly=True, lock=False, readahead=False, meminit=False)
        if self.env_gt is None:
            self.env_gt = lmdb.open(self.gt_root, readonly=True, lock=False, readahead=False, meminit=False)

    def _read_img(self, env, root, rel_path):
        """统一读取图片接口"""
        if self.use_lmdb:
            with env.begin(write=False) as txn:
                buf = txn.get(rel_path.encode('ascii'))
                if buf is None: raise ValueError(f"LMDB Key not found: {rel_path}")
                img_np = np.frombuffer(buf, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(os.path.join(root, rel_path), cv2.IMREAD_COLOR)
            if img is None: raise FileNotFoundError(f"File not found: {os.path.join(root, rel_path)}")
            
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_seq(self, clip, start):
        # 确保 LMDB 在当前进程已打开
        if self.use_lmdb: self._init_lmdb()

        imgs_lr, imgs_gt, mv_fwd_list = [], [], []

        # 1. 读取场极性标识 (npy)
        fid_rel_path = f"{clip}/meta/field_ids.npy"
        if self.use_lmdb:
            with self.env_codec.begin(write=False) as txn:
                buf = txn.get(fid_rel_path.encode('ascii'))
                with io.BytesIO(buf) as f:
                    all_fids = np.load(f)
        else:
            all_fids = np.load(os.path.join(self.codec_root, fid_rel_path))
            
        indices = [start + i for i in range(self.seq_len)]
        seq_fids = all_fids[indices] 

        for t_idx in indices:
            fn = self.img_tmpl.format(t_idx)
            
            # 读取 LR 和 GT 图像
            lr_rel_path = f"{clip}/lr/{fn}"
            gt_rel_path = f"{clip}/{fn}" # 假设 GT 里面直接是 png
            
            imgs_lr.append(self._read_img(self.env_codec, self.codec_root, lr_rel_path))
            imgs_gt.append(self._read_img(self.env_gt, self.gt_root, gt_rel_path))
            
            H_lr, W_lr, _ = imgs_lr[-1].shape
            base_name = os.path.splitext(fn)[0] 

            # 2. 读取光流 (npz)
            mvf_rel_path = f"{clip}/mv_fwd/{base_name}_mv_fwd.npz"
            mvf = np.zeros((2, H_lr, W_lr), np.float32) # 默认全 0，容错机制
            
            if self.use_lmdb:
                with self.env_codec.begin(write=False) as txn:
                    buf = txn.get(mvf_rel_path.encode('ascii'))
                    if buf is not None:
                        with io.BytesIO(buf) as f:
                            f_arr = np.load(f)["flow_fwd"].astype(np.float32)
                            if f_arr.ndim == 3 and f_arr.shape[2] == 2: 
                                f_arr = np.transpose(f_arr, (2, 0, 1))
                            mvf = f_arr
            else:
                p_fwd = os.path.join(self.codec_root, mvf_rel_path)
                if os.path.exists(p_fwd):
                    f_arr = np.load(p_fwd)["flow_fwd"].astype(np.float32)
                    if f_arr.ndim == 3 and f_arr.shape[2] == 2: 
                        f_arr = np.transpose(f_arr, (2, 0, 1))
                    mvf = f_arr
                    
            mv_fwd_list.append(mvf)

        return (
            np.stack(imgs_lr, axis=0),      # [T, H, W, 3]
            np.stack(imgs_gt, axis=0),      # [T, H*2, W, 3]
            np.stack(mv_fwd_list, axis=0),  # [T, 2, H, W]
            seq_fids                        # [T]
        )

    # ... 下面的 _random_crop, __getitem__, __len__ 完全保持你原来的代码不变 ...
    def _random_crop(self, lr, gt, mv_fwd):
        T, H_in, W, _ = lr.shape 
        crop_h = self.crop_lr 
        crop_w = self.crop_lr

        if crop_h is None or H_in < crop_h or W < crop_w:
            return lr, gt, mv_fwd

        y0 = random.randint(0, H_in - crop_h)
        x0 = random.randint(0, W - crop_w)

        lr = lr[:, y0 : y0+crop_h, x0 : x0+crop_w, :]
        mv_fwd = mv_fwd[:, :, y0 : y0+crop_h, x0 : x0+crop_w]
        
        gt_y0 = y0 * 2
        gt_crop_h = crop_h * 2
        gt = gt[:, gt_y0 : gt_y0+gt_crop_h, x0 : x0+crop_w, :]

        return lr, gt, mv_fwd

    def __getitem__(self, idx):
        clip, s = self.samples[idx]
        lr, gt, mv_fwd, fids = self._load_seq(clip, s)

        if self.split == 'train':
            if self.crop_lr is not None:
                lr, gt, mv_fwd = self._random_crop(lr, gt, mv_fwd)
            
            if self.augment and random.random() < 0.5:
                lr = lr[:, :, ::-1, :].copy()
                gt = gt[:, :, ::-1, :].copy()
                mv_fwd = mv_fwd[:, :, :, ::-1].copy()
                mv_fwd[:, 0, :, :] *= -1.0 

        imgs_t = torch.stack([to_tensor01(im) for im in lr], dim=0)  
        gts_t  = torch.stack([to_tensor01(im) for im in gt], dim=0)  
        mv_fwd_t = torch.from_numpy(np.ascontiguousarray(mv_fwd)).float()
        fids_t = torch.from_numpy(np.ascontiguousarray(fids)).long()
        
        return {
            'lr': imgs_t,               
            'hr': gts_t,                
            'mv_fwd': mv_fwd_t,         
            'field_ids': fids_t,        
            'is_new_video': (s == 0),   
            'clip_name': clip,
            'start_frame': s
        }

    def __len__(self):
        return len(self.samples)