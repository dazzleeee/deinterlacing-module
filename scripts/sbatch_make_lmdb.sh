#!/bin/bash
#SBATCH --job-name=make_lmdb
#SBATCH --output=logs/lmdb_log_%j.out
#SBATCH --time=04:00:00        # 预估 4 个小时应该足够了
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # 给 8 个 CPU 核心来加速 I/O 扫描
#SBATCH --mem=32000M           # 申请 32GB 内存，防止缓存文件名时 OOM

mkdir -p logs

module purge
module load StdEnv/2023
module load python/3.11
module load opencv/4.9.0

# 激活你全新的虚拟环境
source /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/venv_mvd/bin/activate

echo "Job started on $(hostname) at $(date)"

# 定义数据根目录
DATA_ROOT="/home/sihanuo/scratch/SihanZhangData"

# 1. 打包训练集输入 (REDS_processed)
echo "Making LMDB for REDS_processed..."
python create_lmdb.py --src $DATA_ROOT/REDS_processed --dst $DATA_ROOT/REDS_processed.lmdb

# 2. 打包训练集真值 (REDS_data_GT)
echo "Making LMDB for REDS_data_GT..."
python create_lmdb.py --src $DATA_ROOT/REDS_data_GT --dst $DATA_ROOT/REDS_data_GT.lmdb

# 3. 打包验证集输入 (REDS_processed_val)
echo "Making LMDB for REDS_processed_val..."
python create_lmdb.py --src $DATA_ROOT/REDS_processed_val --dst $DATA_ROOT/REDS_processed_val.lmdb

# 4. 打包验证集真值 (REDS_data_GT_val)
echo "Making LMDB for REDS_data_GT_val..."
python create_lmdb.py --src $DATA_ROOT/REDS_data_GT_val --dst $DATA_ROOT/REDS_data_GT_val.lmdb

echo "All LMDB packages created successfully at $(date)"