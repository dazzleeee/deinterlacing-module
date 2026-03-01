#!/bin/bash

# =================================================================
# Local single-GPU debug/training script
# Usage: ./local_train.sh config/mvdeinter/<your_yaml>.yaml [checkpoint_path]
# =================================================================

CONFIG_FILE=$1
RESUME_CKPT=$2

# 1️⃣ 参数校验
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified!"
    echo "Usage: ./local_train.sh config/mvdeinter/<your_yaml>.yaml"
    exit 1
fi

# 2️⃣ 项目根目录 (根据脚本位置)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# 绝对路径
CONFIG_FILE_ABS="$PROJECT_ROOT/$CONFIG_FILE"
TRAIN_SCRIPT="$PROJECT_ROOT/train.py"

# 检查 train.py 是否存在
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: train.py not found at $TRAIN_SCRIPT"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE_ABS" ]; then
    echo "Error: Config file not found at $CONFIG_FILE_ABS"
    exit 1
fi

# 3️⃣ 本地数据路径 (请修改为你实际的 REDS LMDB 目录)
LOCAL_DATA_DIR=/home/mcrlab/Documents/REDS_lmdb
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo "Error: Local data directory not found at $LOCAL_DATA_DIR"
    exit 1
fi

# 4️⃣ 处理 checkpoint 参数
if [ -n "$RESUME_CKPT" ]; then
    RESUME_ARGS="--resume $(realpath "$RESUME_CKPT")"
else
    RESUME_ARGS=""
fi

# 5️⃣ 激活虚拟环境
VENV_PATH="$PROJECT_ROOT/.ven/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"

# 6️⃣ 环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MASTER_PORT=$((10000 + RANDOM % 10000))

# 7️⃣ 临时配置文件，利用 YAML 深度合并机制覆盖数据路径
CONFIG_DIR=$(dirname "$CONFIG_FILE_ABS")
CONFIG_BASENAME=$(basename "$CONFIG_FILE_ABS")
TMP_CONFIG_ABS="${CONFIG_DIR}/.tmp_local_${CONFIG_BASENAME}"

# 先复制主配置文件的内容
cp "$CONFIG_FILE_ABS" "$TMP_CONFIG_ABS"

# 在临时文件末尾追加本地数据集路径，利用 load_config 的合并机制覆盖 base 配置
cat <<EOF >> "$TMP_CONFIG_ABS"

dataset:
  train:
    root_dir: '$LOCAL_DATA_DIR'

  val:
    root_dir: '$LOCAL_DATA_DIR'

train:
    log_freq: 1

EOF

echo "Starting local training with config: $TMP_CONFIG_ABS"

# 8️⃣ 启动训练
torchrun --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    "$TRAIN_SCRIPT" \
    -c "$TMP_CONFIG_ABS" \
    $RESUME_ARGS

# 9️⃣ 清理
rm -f "$TMP_CONFIG_ABS"
echo "Local session finished."