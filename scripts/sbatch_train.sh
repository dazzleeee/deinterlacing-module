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

source /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/venv_mvd/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Job started on $(hostname) at $(date)"

# =================================================================
# 🚀 终极 I/O 优化：将数据拷贝到计算节点本地固态硬盘 ($SLURM_TMPDIR)
# =================================================================
echo "Copying LMDB datasets to \$SLURM_TMPDIR..."
time cp -r /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/data/REDS $SLURM_TMPDIR/REDS
echo "Copy finished!"

# 建立软链接魔法：把你项目里的 data/REDS 临时指向固态硬盘
# 这样你代码里的 YAML 依然可以写 root_dir: 'data/REDS'，但实际上读的是本地固态！
rm -rf data/REDS_link  # 清理可能存在的旧链接
ln -s $SLURM_TMPDIR/REDS data/REDS_link

# ⚠️ 注意：你需要在你的 YAML 配置文件里，把 root_dir 改为 'data/REDS_link'
# =================================================================

# 启动训练
torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    train.py \
    -c config/mvdeinter/v2_sota_attention_convex.yaml

echo "Job finished at $(date)"