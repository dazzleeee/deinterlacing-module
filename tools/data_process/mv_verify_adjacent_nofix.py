import cv2
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import pandas as pd

def calculate_psnr(img1, img2):
    """计算 PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def warp_direct(img, flow):
    """
    最原始的 Warp，不做任何额外偏移
    img: 参考帧 (Source)
    flow: 原始运动向量 (2, H, W)
    """
    H, W = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    
    # 直接使用 MV，不加任何 shift
    map_x = grid_x.astype(np.float32) + flow[0]
    map_y = grid_y.astype(np.float32) + flow[1]
    
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def process_baseline_no_correction(clip_root, output_dir, save_images=True):
    # 1. 准备路径
    mv_dir = os.path.join(clip_root, "mv_fwd")
    lr_dir = os.path.join(clip_root, "lr")
    
    vis_dir = os.path.join(output_dir, "vis_baseline_raw")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 2. 获取所有 MV 文件
    mv_files = sorted(glob.glob(os.path.join(mv_dir, "*_mv_fwd.npz")))
    
    results = []
    print(f"开始对照组实验: 直接使用原始 MV (t->t-2) Warp (t-1)")
    print(f"预期结果: 运动幅度过大 (Over-warping)")

    for mv_file in tqdm(mv_files, desc="Processing"):
        # 当前帧 t
        filename = os.path.basename(mv_file)
        try:
            curr_idx = int(filename.split('_')[0]) # t
        except ValueError: continue
            
        ref_idx = curr_idx - 1                 # t-1
        
        if ref_idx < 0: continue
        
        path_target = os.path.join(lr_dir, f"{curr_idx:08d}.png")
        path_ref    = os.path.join(lr_dir, f"{ref_idx:08d}.png")
        
        if not os.path.exists(path_ref) or not os.path.exists(path_target):
            continue

        # 读取图像
        img_target = cv2.imread(path_target).astype(np.float32) / 255.0
        img_ref    = cv2.imread(path_ref).astype(np.float32) / 255.0
        
        # 读取 MV (t -> t-2)
        mv_data = np.load(mv_file)
        if 'flow_fwd' not in mv_data: continue
        flow_orig = mv_data['flow_fwd'] 
        
        # === 核心修改点 ===
        # 1. 不做时间缩放 (No Scaling): flow_used = flow_orig
        # 2. 不做空间对齐 (No Shift): y_shift = 0
        
        img_warped = warp_direct(img_ref, flow_orig)
        
        # 计算误差
        psnr_val = calculate_psnr(img_target, img_warped)
        
        results.append({
            "curr_idx": curr_idx,
            "ref_idx": ref_idx,
            "type": "No_Scale_No_Shift",
            "psnr": psnr_val
        })
        
        # === 可视化 ===
        if save_images:
            diff = np.abs(img_target - img_warped)
            diff_viz = np.mean(diff, axis=2)
            diff_viz = (diff_viz * 5).clip(0, 1)
            diff_viz = cv2.cvtColor(diff_viz, cv2.COLOR_GRAY2BGR)
            
            row1 = np.hstack([img_ref, img_target])
            row2 = np.hstack([img_warped, diff_viz])
            grid = np.vstack([row1, row2])
            
            grid = (grid * 255).astype(np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            cv2.putText(grid, f"Ref(t-1): {ref_idx}", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(grid, f"Target(t): {curr_idx}", (10 + img_ref.shape[1], 30), font, 1, (0, 255, 0), 2)
            
            # 标注这是未处理的
            cv2.putText(grid, f"BAD Warp (Raw MV)", (10, 30 + img_ref.shape[0]), font, 1, (0, 0, 255), 2) 
            cv2.putText(grid, f"PSNR: {psnr_val:.2f}", (10, 70 + img_ref.shape[0]), font, 1, (0, 0, 255), 2)
            
            cv2.imwrite(os.path.join(vis_dir, f"raw_{curr_idx:08d}.jpg"), grid)

    # 保存结果
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "baseline_raw_results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n对照组完成!")
    print(f"平均 PSNR: {df['psnr'].mean():.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--no_img", action="store_true")
    args = parser.parse_args()
    
    process_baseline_no_correction(args.clip_path, args.out_dir, save_images=not args.no_img)