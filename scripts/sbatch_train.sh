#!/bin/bash
#SBATCH --job-name=mvd_train
#SBATCH --output=logs/train_log_%j.out
#SBATCH --error=logs/train_err_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000M

# 1. 明确进入代码根目录 [cite: 20, 21]
cd /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/deinterlacing-module

# 2. 准备基础目录 
mkdir -p logs
mkdir -p data/jobs  # 专门存放各任务专属链接的目录

# 3. 加载环境 [cite: 256]
module purge
module load StdEnv/2023
module load python/3.11
module load opencv/4.9.0
source /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/.venv/bin/activate

# 4. 性能优化与 DDP 端口设置 [cite: 42, 256]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# =================================================================
# 🚀 你的核心参数逻辑：接收配置文件路径
# =================================================================
CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified! Usage: sbatch run_train.sh <path_to_yaml>"
    exit 1
fi

# =================================================================
# 🚀 隔离魔法：为每个任务 ID 创建专属的“私人传送门”
# =================================================================
MY_JOB_DIR="data/jobs/${SLURM_JOBID}"
mkdir -p "$MY_JOB_DIR"

# 5. 准备本地高速 SSD 空间 [cite: 25, 68]
mkdir -p $SLURM_TMPDIR/REDS_data
echo "Copying LMDB from Project to Local SSD..."
# ✅ 源路径已改为你刚才搬运后的 Project 目录
rsync -avP /home/sihanuo/projects/def-jyzhao/SihanZhang/REDSdata/*.lmdb $SLURM_TMPDIR/REDS_data/

# 6. 建立任务专属链接：MY_JOB_DIR/REDS_link -> 本地固态硬盘 [cite: 67, 68]
ln -s $SLURM_TMPDIR/REDS_data "$MY_JOB_DIR/REDS_link"

# 7. 动态重定向配置：生成一份临时 YAML，将路径指向该任务的专属链接
TMP_CONFIG="logs/tmp_config_${SLURM_JOBID}.yaml"
# 用 sed 将配置文件里原有的 'data/REDS_link' 替换为当前任务的物理隔离路径 
sed "s|data/REDS_link|$MY_JOB_DIR/REDS_link|g" "$CONFIG_FILE" > "$TMP_CONFIG"

echo "Starting training with unique config: $TMP_CONFIG"

# 8. 启动训练 [cite: 41, 47]
torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    train.py \
    -c "$TMP_CONFIG"