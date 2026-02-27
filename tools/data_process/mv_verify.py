import cv2
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import pandas as pd

def calculate_psnr(img1, img2):
    """计算峰值信噪比 (PSNR)，数值越大越好"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def warp(img, flow):
    H, W = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = grid_x.astype(np.float32) + flow[0]
    map_y = grid_y.astype(np.float32) + flow[1]
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def process_clip(clip_root, output_dir, save_images=True):
    """批量处理单个片段的所有帧"""
    
    # 1. 准备路径
    mv_dir = os.path.join(clip_root, "mv_fwd")
    lr_dir = os.path.join(clip_root, "lr")
    
    # 创建保存结果的文件夹
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 2. 获取所有 MV 文件
    mv_files = sorted(glob.glob(os.path.join(mv_dir, "*_mv_fwd.npz")))
    
    if not mv_files:
        print(f"在 {mv_dir} 中没找到任何 .npz 文件")
        return

    results = []
    
    print(f"开始处理: {clip_root}")
    print(f"结果将保存到: {output_dir}")

    # 3. 循环处理每一帧
    for mv_file in tqdm(mv_files, desc="Verifying frames"):
        # 从文件名提取帧索引 (例如 00000150_mv_fwd.npz -> 150)
        filename = os.path.basename(mv_file)
        frame_idx = int(filename.split('_')[0])
        
        # 确定参考帧 (Top找上一个Top, Bot找上一个Bot)
        ref_idx = frame_idx - 2
        
        # 构建图片路径
        path_target = os.path.join(lr_dir, f"{frame_idx:08d}.png")
        path_ref = os.path.join(lr_dir, f"{ref_idx:08d}.png")
        
        if not os.path.exists(path_ref):
            continue # 如果参考帧不存在（极少数情况），跳过
            
        # 读取数据
        img_target = cv2.imread(path_target).astype(np.float32) / 255.0
        img_ref = cv2.imread(path_ref).astype(np.float32) / 255.0
        
        # 读取 MV
        mv_data = np.load(mv_file)
        flow = mv_data['flow_fwd']
        
        # === 核心：Warp ===
        img_warped = warp(img_ref, flow)
        
        # === 计算误差 ===
        # PSNR: 越高越好 (通常 > 30dB 说明对齐得不错)
        psnr_val = calculate_psnr(img_target, img_warped)
        
        # 记录数据
        results.append({
            "frame_idx": frame_idx,
            "ref_idx": ref_idx,
            "psnr": psnr_val,
            "type": "Top" if frame_idx % 2 == 0 else "Bot"
        })
        
        # === 保存可视化图片 (可选) ===
        if save_images:
            # 拼接图片方便查看: [参考帧, 目标帧, Warp后帧, 误差热力图]
            
            # 1. 误差图 (放大5倍以便观察)
            diff = np.abs(img_target - img_warped)
            diff_viz = np.mean(diff, axis=2) # 转灰度
            diff_viz = (diff_viz * 5).clip(0, 1) # 放大亮度
            diff_viz = cv2.cvtColor(diff_viz, cv2.COLOR_GRAY2BGR) # 转回3通道以便拼接
            
            # 2. 拼图 (2行2列)
            # Row 1: Ref | Target
            row1 = np.hstack([img_ref, img_target])
            # Row 2: Warped | Error
            row2 = np.hstack([img_warped, diff_viz])
            # Combine
            grid = np.vstack([row1, row2])
            
            # 添加文字说明
            grid = (grid * 255).astype(np.uint8)
            cv2.putText(grid, f"Ref {ref_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(grid, f"Target {frame_idx}", (10 + img_ref.shape[1], 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(grid, f"Warped (PSNR: {psnr_val:.2f}dB)", (10, 30 + img_ref.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(grid, f"Residual Error (x5)", (10 + img_ref.shape[1], 30 + img_ref.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            save_path = os.path.join(vis_dir, f"verify_{frame_idx:08d}.jpg")
            cv2.imwrite(save_path, grid)

    # 4. 保存统计结果到 CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "verification_results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n完成！")
    print(f"统计数据已保存: {csv_path}")
    print(f"可视化图片已保存: {vis_dir}")
    print(f"平均 PSNR: {df['psnr'].mean():.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, required=True, help="输入片段路径 (如 REDS_processed/000)")
    parser.add_argument("--out_dir", type=str, required=True, help="结果保存路径 (如 verification_output)")
    parser.add_argument("--no_img", action="store_true", help="如果不加这个参数，会生成图片；加上则只计算CSV数据")
    
    args = parser.parse_args()
    
    process_clip(args.clip_path, args.out_dir, save_images=not args.no_img)