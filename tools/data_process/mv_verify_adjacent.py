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

def warp_with_shift(img, flow, y_shift):
    """
    带有垂直偏移的 Warp
    img: 参考帧 (Source, 这里是 t-1)
    flow: 缩放后的运动向量 (2, H, W)
    y_shift: 垂直方向的补偿值
    """
    H, W = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    
    # 核心逻辑 (Pull/Backward Warp):
    # 我们要填满当前帧 (Target) 的像素 (x, y)
    # 我们去 Source (t-1) 里找对应位置
    # 位置 = (x + mv_x_half, y + mv_y_half + shift)
    map_x = grid_x.astype(np.float32) + flow[0]
    map_y = grid_y.astype(np.float32) + flow[1] + y_shift
    
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def process_field_adaptation(clip_root, output_dir, save_images=True):
    # 1. 准备路径
    mv_dir = os.path.join(clip_root, "mv_fwd")
    lr_dir = os.path.join(clip_root, "lr")
    meta_dir = os.path.join(clip_root, "meta")
    
    vis_dir = os.path.join(output_dir, "vis_field_adapt")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 2. 加载 field_ids.npy
    field_ids_path = os.path.join(meta_dir, "field_ids.npy")
    if not os.path.exists(field_ids_path):
        print(f"错误: 找不到 {field_ids_path}")
        return
    field_ids = np.load(field_ids_path)
    
    # 3. 获取 MV 文件
    mv_files = sorted(glob.glob(os.path.join(mv_dir, "*_mv_fwd.npz")))
    
    results = []
    print(f"开始验证: 用 t-1 场 + 修改后的 MV 预测 t 场")

    for mv_file in tqdm(mv_files, desc="Processing"):
        # === 索引逻辑修正 ===
        # 文件名是当前帧 t 的索引 (因为 MV 属于 t)
        filename = os.path.basename(mv_file)
        curr_idx = int(filename.split('_')[0]) # t (Target)
        
        ref_idx = curr_idx - 1                 # t-1 (Source, 异极性)
        
        # 边界检查
        if ref_idx < 0: continue
        
        # 读取路径
        path_target = os.path.join(lr_dir, f"{curr_idx:08d}.png") # t
        path_ref    = os.path.join(lr_dir, f"{ref_idx:08d}.png")  # t-1
        
        if not os.path.exists(path_ref) or not os.path.exists(path_target):
            continue

        # 读取图像
        img_target = cv2.imread(path_target).astype(np.float32) / 255.0
        img_ref    = cv2.imread(path_ref).astype(np.float32) / 255.0
        
        # 读取 MV (t -> t-2)
        mv_data = np.load(mv_file)
        flow_orig = mv_data['flow_fwd'] # (2, H, W)
        
        # === 核心逻辑: 修改 MV ===
        
        # 1. 时间缩放: t->t-2 变成 t->t-1，距离减半
        flow_half = flow_orig * 0.5
        
        # 2. 空间偏移: 根据场极性决定
        curr_field_type = field_ids[curr_idx] # 0(Top) or 1(Bot)
        
        if curr_field_type == 0: 
            # 当前是 Top (t)，参考是 Bot (t-1)
            # 目标: Top -> Bot
            type_str = "Top(t) from Bot(t-1)"
            # 这里设置你要测试的偏移量，比如 -0.5
            y_shift = -0.5 
        else:
            # 当前是 Bot (t)，参考是 Top (t-1)
            # 目标: Bot -> Top
            type_str = "Bot(t) from Top(t-1)"
            # 对称操作
            y_shift = 0.5 
            
        # 3. 执行 Warp (用 t-1 预测 t)
        img_warped = warp_with_shift(img_ref, flow_half, y_shift)
        
        # 4. 计算误差
        psnr_val = calculate_psnr(img_target, img_warped)
        
        results.append({
            "curr_idx": curr_idx,
            "ref_idx": ref_idx,
            "type": type_str,
            "psnr": psnr_val
        })
        
        # === 可视化 ===
        if save_images:
            # 左上: Ref (t-1)
            # 右上: Target (t) - 真实值
            # 左下: Warped Ref (t-1 + MV_mod) - 预测值
            # 右下: Error
            
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
            cv2.putText(grid, f"Pred w/ Shift {y_shift}", (10, 30 + img_ref.shape[0]), font, 1, (0, 255, 255), 2)
            cv2.putText(grid, f"PSNR: {psnr_val:.2f}", (10, 70 + img_ref.shape[0]), font, 1, (0, 255, 255), 2)
            
            cv2.imwrite(os.path.join(vis_dir, f"adapt_{curr_idx:08d}.jpg"), grid)

    # 保存结果
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "field_adaptation_results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n平均 PSNR: {df['psnr'].mean():.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--no_img", action="store_true")
    args = parser.parse_args()
    
    process_field_adaptation(args.clip_path, args.out_dir, save_images=not args.no_img)