import sys
import os.path as osp
import glob
import pickle
import lmdb
import argparse
from tqdm import tqdm

def make_lmdb(data_root, lmdb_path, commit_interval=1000):
    """
    制作 LMDB 数据库 (支持 png, jpg, npz, npy)
    """
    
    # 1. 准备文件列表
    print(f"Scanning files in {data_root}...")
    
    # 定义我们要抓取的所有后缀名
    extensions = ['*.png', '*.npz', '*.npy']
    files = []
    
    for ext in extensions:
        # recursive=True 会扫描所有子文件夹
        found = glob.glob(osp.join(data_root, '**', ext), recursive=True)
        files.extend(found)
        
    files.sort()
    
    if not files:
        print(f"Error: No files found in {data_root}")
        return

    # 2. 估算所需的存储空间
    # 统计所有文件大小，并乘 1.2 倍作为缓冲区 (LMDB map_size)
    data_size = sum(osp.getsize(f) for f in files)
    map_size = int(data_size * 1.2)
    
    print(f"Total files: {len(files)}")
    print(f"Estimated size: {data_size / 1024**3:.2f} GB. Map size set to: {map_size / 1024**3:.2f} GB")

    if osp.exists(lmdb_path):
        print(f"Warning: Folder {lmdb_path} already exists. Deleting it...")
        import shutil
        shutil.rmtree(lmdb_path)

    # 3. 创建 LMDB 环境
    # map_size 决定了数据库最大能多大，必须预留够
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    
    # 用于存储所有 Key 的列表，方便 Dataset 快速索引
    keys = []

    for idx, path in enumerate(tqdm(files)):
        # 4. 生成 Key (相对路径)
        # 例如: 
        #   data_root/meta/field_ids.npy -> Key: meta/field_ids.npy
        #   data_root/000/lr/00000001.png -> Key: 000/lr/00000001.png
        rel_path = osp.relpath(path, data_root)
        key = rel_path.replace('\\', '/') # 强制使用 Linux 风格斜杠
        
        # 5. 读取二进制数据
        # 无论是 png 图片还是 npy 数组，本质都是 bytes
        with open(path, 'rb') as f:
            data = f.read()
            
        # 6. 写入 LMDB
        if not txn.put(key.encode('ascii'), data):
            print(f"Write failed: {key}")
        
        keys.append(key)

        # 定期 Commit
        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    
    # 7. 保存 meta_info (Keys 列表)
    print("Saving meta info...")
    keys_cache_file = osp.join(lmdb_path, 'meta_info.pkl')
    with open(keys_cache_file, 'wb') as f:
        pickle.dump(keys, f)
        
    env.close()
    print(f"Finish. LMDB saved at {lmdb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="原始数据文件夹路径")
    parser.add_argument("--dst", type=str, required=True, help="输出 LMDB 文件夹路径")
    args = parser.parse_args()
    
    make_lmdb(args.src, args.dst)