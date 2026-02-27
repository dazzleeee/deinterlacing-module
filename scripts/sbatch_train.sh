#!/bin/bash
#SBATCH --job-name=mvd_train
#SBATCH --output=logs/train_log_%j.out
#SBATCH --error=logs/train_err_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.80gb:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000M

mkdir -p logs

module purge
module load StdEnv/2023
module load python/3.11
module load opencv/4.9.0

source /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/.venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Job started on $(hostname) at $(date)"

# =================================================================
# 🚀 终极 I/O 优化：将数据拷贝到计算节点本地固态硬盘 ($SLURM_TMPDIR)
# =================================================================
# 1. 在本地极速固态硬盘创建目录
mkdir -p $SLURM_TMPDIR/REDS_data

echo "Copying LMDB datasets from scratch to \$SLURM_TMPDIR..."
# 2. 从 scratch 拷贝数据 (确保包含所有 .lmdb 文件夹)
time cp -r /home/sihanuo/scratch/SihanZhangData/*.lmdb $SLURM_TMPDIR/REDS_data/
echo "Copy finished!"

# 3. 建立软链接魔法：确保指向刚才创建的 REDS_data 目录 [cite: 24]
rm -rf data/REDS_link  
ln -s $SLURM_TMPDIR/REDS_data data/REDS_link

# ⚠️ 此时，YAML 配置文件里的 root_dir: 'data/REDS_link' 就能通过这个“传送门”
# 访问到本地固态硬盘里的 REDS_processed.lmdb 等文件了。

# 启动训练
torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    train.py \
    -c config/mvdeinter/v2_sota_attention_convex.yaml

echo "Job finished at $(date)"
