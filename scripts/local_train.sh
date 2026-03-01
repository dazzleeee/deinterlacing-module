#!/bin/bash

# =================================================================
# 本地单卡 Debug/训练脚本
# 用法: ./local_train.sh config/mvdeinter/<your_yaml>.yaml [checkpoint_path]
# =================================================================

CONFIG_FILE=$1
RESUME_CKPT=$2

# 1. 参数校验
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified!"
    echo "Usage: ./local_train.sh config/mvdeinter/<your_yaml>.yaml"
    exit 1
fi

# 2. 路径处理
# 获取脚本所在目录的绝对路径，确保在任何地方运行都能找到根目录
PROJECT_ROOT=$(cd "$(dirname "$0")"; pwd)
cd "$PROJECT_ROOT"

# 定义你的本地数据存放路径 (请修改为你本地 LMDB 文件夹的实际位置)
# 获取 PROJECT_ROOT 上一级的 data/REDS_data 目录的绝对路径
LOCAL_DATA_DIR=$(realpath "$PROJECT_ROOT/../data/REDS_data")

if [ -n "$RESUME_CKPT" ]; then
    RESUME_ARGS="--resume $(realpath "$RESUME_CKPT")"
else
    RESUME_ARGS=""
fi

CONFIG_FILE_ABS=$(realpath "$CONFIG_FILE")

# 3. 环境准备 (本地不需要 module load)
# 如果你使用 conda，请取消下面这行的注释并修改环境名
# source activate your_env_name
# 或者使用 venv
source .venv/bin/activate 

# 4. 环境变量设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 
export MASTER_PORT=$((10000 + RANDOM % 10000))

# 5. 动态重定向配置 (核心：将配置文件中的路径指向本地数据)
# 原脚本通过 sed 替换 data/REDS_link。
# 本地最简单的做法是：确保 LOCAL_DATA_DIR 路径下有 REDS_data_GT.lmdb 等文件
CONFIG_DIR=$(dirname "$CONFIG_FILE_ABS")
CONFIG_BASENAME=$(basename "$CONFIG_FILE_ABS")
TMP_CONFIG_ABS="${CONFIG_DIR}/.tmp_local_${CONFIG_BASENAME}"

# 将配置文件里的数据路径占位符替换为本地实际绝对路径
sed "s|root_dir: '.*'|root_dir: '$LOCAL_DATA_DIR'|g" "$CONFIG_FILE_ABS" > "$TMP_CONFIG_ABS"

echo "Starting local training with config: $TMP_CONFIG_ABS"

# 6. 启动训练 (单机单卡)
# 虽然是单卡，建议保留 torchrun，因为 train.py 中包含了 DDP 初始化逻辑 [cite: 18, 41]
torchrun --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    train.py \
    -c "$TMP_CONFIG_ABS" \
    $RESUME_ARGS

# 7. 清理
rm -f "$TMP_CONFIG_ABS"
echo "Local session finished."
