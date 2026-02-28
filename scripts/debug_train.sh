#!/bin/bash

# =================================================================
# 交互式 Debug 专用脚本
# 用法: ./debug_train.sh config/mvdeinter/<your_yaml>.yaml [checkpoint_path]
# =================================================================

CONFIG_FILE=$1
RESUME_CKPT=$2

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified!"
    echo "Usage: ./debug_train.sh config/mvdeinter/<your_yaml>.yaml"
    exit 1
fi

if [ -n "$RESUME_CKPT" ]; then
    RESUME_CKPT_ABS=$(realpath "$RESUME_CKPT")
    RESUME_ARGS="--resume $RESUME_CKPT_ABS"
    echo "Resuming from checkpoint: $RESUME_CKPT_ABS"
else
    RESUME_ARGS=""
    echo "Starting training from scratch..."
fi

CONFIG_FILE_ABS=$(realpath "$CONFIG_FILE")
if [ ! -f "$CONFIG_FILE_ABS" ]; then
    echo "Error: Config file not found at $CONFIG_FILE_ABS"
    exit 1
fi

# 1. 确保在正确的根目录
PROJECT_ROOT="/home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacingGit/deinterlacing-module"
cd "$PROJECT_ROOT" || { echo "Fatal Error: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "Current working directory is: $(pwd)"

# 2. 准备基础目录 
mkdir -p logs
mkdir -p data/jobs

# 3. 加载环境 (直接运行这部分，确保交互式环境和提交任务环境一致)
module purge
module load StdEnv/2023
module load python/3.11
module load opencv/4.9.0
source /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/.venv/bin/activate

# 4. 环境变量设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 刚才你申请了6个CPU核心，所以这里改为6
export OMP_NUM_THREADS=6 
export MASTER_PORT=$((10000 + RANDOM % 10000)) # 交互式下随机生成一个端口避免冲突

# 5. 数据搬运到本地 SSD (这步在 Debug 时可能要等几分钟，但能避免卡顿)
MY_JOB_DIR="data/jobs/${SLURM_JOBID:-debug_$$}"
mkdir -p "$MY_JOB_DIR"
mkdir -p $SLURM_TMPDIR/REDS_data

echo "Copying LMDB from Project to Local SSD..."
# 注意：如果数据量特别大且你只 debug 几张图，这里可以临时注释掉，直接软链接原路径
#rsync -avP /home/sihanuo/projects/def-jyzhao/SihanZhang/REDSdata/*.lmdb $SLURM_TMPDIR/REDS_data/

ln -s $SLURM_TMPDIR/REDS_data "$MY_JOB_DIR/REDS_link"

# 6. 动态重定向配置
CONFIG_DIR=$(dirname "$CONFIG_FILE_ABS")
CONFIG_BASENAME=$(basename "$CONFIG_FILE_ABS")
TMP_CONFIG_ABS="${CONFIG_DIR}/.tmp_debug_${CONFIG_BASENAME}"

sed "s|data/REDS_link|$MY_JOB_DIR/REDS_link|g" "$CONFIG_FILE_ABS" > "$TMP_CONFIG_ABS"
echo "Starting debug training with config: $TMP_CONFIG_ABS"

# 7. 启动训练 (注意：交互式单卡环境，改为 nproc_per_node=1)
torchrun --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    train.py \
    -c "$TMP_CONFIG_ABS" \
    $RESUME_ARGS

# 8. 清理
rm -f "$TMP_CONFIG_ABS"
echo "Debug session finished."