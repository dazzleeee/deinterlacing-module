import os
import glob
import pandas as pd

def scan_verification_directories():
    root_dir = "."
    
    # 1. 获取当前目录下所有的子文件夹
    # 使用 os.listdir 获取所有名称，然后筛选出是文件夹且名字带 "verification" 的
    all_items = os.listdir(root_dir)
    target_dirs = []
    
    for item in all_items:
        full_path = os.path.join(root_dir, item)
        if os.path.isdir(full_path) and "verification" in item.lower():
            target_dirs.append(full_path)
            
    target_dirs.sort()

    print("\n" + "="*90)
    print(f"{'文件夹名称':<45} | {'CSV 文件名':<25} | {'平均 PSNR'}")
    print("-" * 90)

    if not target_dirs:
        print("未找到任何名字包含 'verification' 的文件夹。")
        return

    # 2. 遍历这些特定的文件夹
    for dir_path in target_dirs:
        dir_name = os.path.basename(dir_path)
        
        # 在该文件夹下找所有的 .csv 文件
        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
        
        if not csv_files:
            print(f"{dir_name:<45} | {'(无 CSV 文件)':<25} | --")
            continue
            
        for csv_file in csv_files:
            csv_name = os.path.basename(csv_file)
            
            try:
                df = pd.read_csv(csv_file)
                
                # 寻找 PSNR 列 (忽略大小写)
                psnr_col = None
                for col in df.columns:
                    if "psnr" in col.lower():
                        psnr_col = col
                        break
                
                if psnr_col:
                    avg = df[psnr_col].mean()
                    print(f"{dir_name:<45} | {csv_name:<25} | {avg:.2f} dB")
                else:
                    # 如果CSV里没有psnr列（比如是存其他数据的），就不显示数值
                    pass
                    
            except Exception as e:
                print(f"{dir_name:<45} | {csv_name:<25} | [读取错误]")

    print("="*90 + "\n")

if __name__ == "__main__":
    scan_verification_directories()