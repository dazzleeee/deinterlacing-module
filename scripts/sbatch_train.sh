#!/bin/bash
#SBATCH --job-name=mvd_train
#SBATCH --output=scripts/logs/train_log_%j.out
#SBATCH --error=scripts/logs/train_err_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000M

# =================================================================
# 🚀 1. 核心参数逻辑：接收配置文件路径并转化为绝对路径
# =================================================================
CONFIG_FILE=$1
RESUME_CKPT=$2

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified!"
    echo "Usage: sbatch scripts/run_train.sh config/mvdeinter/<your_yaml>.yaml"
    exit 1
fi
if [ -n "$RESUME_CKPT" ]; then
    # 获取权重的绝对路径，防止路径错乱
    RESUME_CKPT_ABS=$(realpath "$RESUME_CKPT")
    RESUME_ARGS="--resume $RESUME_CKPT_ABS"
    echo "Resuming from checkpoint: $RESUME_CKPT_ABS"
else
    RESUME_ARGS=""
    echo "Starting training from scratch..."
fi
# 获取原文件的物理绝对路径
CONFIG_FILE_ABS=$(realpath "$CONFIG_FILE")

if [ ! -f "$CONFIG_FILE_ABS" ]; then
    echo "Error: Config file not found at $CONFIG_FILE_ABS"
    exit 1
fi

# =================================================================
# 🚀 2. 明确进入代码根目录
# =================================================================
PROJECT_ROOT="/home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacingGit/deinterlacing-module"
cd "$PROJECT_ROOT" || { echo "Fatal Error: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "Current working directory is: $(pwd)"
echo "Absolute config file path is: $CONFIG_FILE_ABS"

# 3. 准备基础目录 
mkdir -p logs
mkdir -p data/jobs

# 4. 加载环境
module purge
module load StdEnv/2023
module load python/3.11
module load opencv/4.9.0
source /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/.venv/bin/activate

# 5. 性能优化与 DDP 端口设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# =================================================================
# 🚀 6. 隔离魔法与数据搬运
# =================================================================
MY_JOB_DIR="data/jobs/${SLURM_JOBID}"
mkdir -p "$MY_JOB_DIR"

mkdir -p $SLURM_TMPDIR/REDS_data
echo "Copying LMDB from Project to Local SSD..."
rsync -avP /home/sihanuo/projects/def-jyzhao/SihanZhang/REDSdata/*.lmdb $SLURM_TMPDIR/REDS_data/

ln -s $SLURM_TMPDIR/REDS_data "$MY_JOB_DIR/REDS_link"

# =================================================================
# 🚀 7. 动态重定向配置 (终极解法：在原目录生成同级临时文件)
# =================================================================
# 提取原配置文件的所在目录和文件名
CONFIG_DIR=$(dirname "$CONFIG_FILE_ABS")
CONFIG_BASENAME=$(basename "$CONFIG_FILE_ABS")

# 在 config/mvdeinter/ 目录下生成一个带 JobID 的隐藏文件
TMP_CONFIG_ABS="${CONFIG_DIR}/.tmp_${SLURM_JOBID}_${CONFIG_BASENAME}"

# 替换数据集路径并输出到这个同级临时文件
sed "s|data/REDS_link|$MY_JOB_DIR/REDS_link|g" "$CONFIG_FILE_ABS" > "$TMP_CONFIG_ABS"

echo "Starting training with unique config: $TMP_CONFIG_ABS"

# 8. 启动训练
torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    train.py \
    -c "$TMP_CONFIG_ABS"
    $RESUME_ARGS

# 9. 训练结束后，自动清理这个临时文件，保持目录干净
rm -f "$TMP_CONFIG_ABS"