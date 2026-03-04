#!/bin/bash
#SBATCH --job-name=mvd_train
#SBATCH --output=scripts/logs/train_log_%j.out
#SBATCH --error=scripts/logs/train_err_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:2            
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000M

# =================================================================
# 🚀 1. 自动定位并进入代码根目录 (SLURM 终极解法)
# =================================================================
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
fi
PROJECT_ROOT=$(pwd)

echo "Current working directory is: $PROJECT_ROOT"

# =================================================================
# 🚀 2. 核心参数逻辑
# =================================================================
CONFIG_FILE=$1
RESUME_CKPT=$2

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified!"
    echo "Usage: sbatch scripts/run_train.sh config/mvdeinter/<your_yaml>.yaml"
    exit 1
fi

CONFIG_FILE_ABS=$(realpath "$CONFIG_FILE")
if [ ! -f "$CONFIG_FILE_ABS" ]; then
    echo "Error: Config file not found at $CONFIG_FILE_ABS"
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

# =================================================================
# 🚀 3. 定位原始数据目录 (Nibi 上的路径)
# =================================================================
ORIGINAL_DATA_DIR="/home/sihanuo/projects/def-jyzhao/SihanZhang/REDSdata"
if [ ! -d "$ORIGINAL_DATA_DIR" ]; then
    echo "Error: Original data directory not found at $ORIGINAL_DATA_DIR"
    exit 1
fi

# =================================================================
# 🚀 4. 加载环境与性能优化
# =================================================================
mkdir -p scripts/logs

module purge
module load StdEnv/2023 python/3.11 opencv/4.9.0
# Nibi 上的虚拟环境路径
source /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/.venv/bin/activate

# 强制 Python 无缓冲输出，用于实时用 tail -f 查看 train_log_xxx.out
export PYTHONUNBUFFERED=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# ================= 🚨 拯救双卡卡死的 NCCL 魔法 =================
export MASTER_ADDR="127.0.0.1"    
export NCCL_SOCKET_IFNAME=lo      
export NCCL_DEBUG=INFO            
# ===============================================================

# =================================================================
# 🚀 5. 数据极速搬运 (向计算节点的本地 SSD 拷贝)
# =================================================================
NODE_LOCAL_DATA="$SLURM_TMPDIR/REDS_data"
mkdir -p "$NODE_LOCAL_DATA"

echo "Copying LMDB from $ORIGINAL_DATA_DIR to Local SSD ($NODE_LOCAL_DATA)..."
rsync -aP "$ORIGINAL_DATA_DIR/"*.lmdb "$NODE_LOCAL_DATA/"

# =================================================================
# 🚀 6. 动态生成专属配置文件 (比以前的 sed 安全 100 倍)
# =================================================================
CONFIG_DIR=$(dirname "$CONFIG_FILE_ABS")
CONFIG_BASENAME=$(basename "$CONFIG_FILE_ABS")
TMP_CONFIG_ABS="${CONFIG_DIR}/tmp_${SLURM_JOBID}_${CONFIG_BASENAME}"

cp "$CONFIG_FILE_ABS" "$TMP_CONFIG_ABS"

cat <<EOF >> "$TMP_CONFIG_ABS"

dataset:
  train:
    root_dir: '$NODE_LOCAL_DATA'
  val:
    root_dir: '$NODE_LOCAL_DATA'
EOF

echo "Starting training with unique config: $TMP_CONFIG_ABS"

# =================================================================
# 🚀 7. 启动 DDP 训练 (注意这里修好了漏掉的斜杠)
# =================================================================
torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    train.py \
    -c "$TMP_CONFIG_ABS" \
    $RESUME_ARGS

# 8. 训练结束后自动清理临时文件
rm -f "$TMP_CONFIG_ABS"
echo "Training job completed."