#!/bin/bash

# 1. 进入你的项目根目录
cd /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacingGit/deinterlacing-module || exit 1
echo "当前工作目录: $(pwd)"

# 2. 加载环境并激活虚拟环境
module purge
module load StdEnv/2023 python/3.11 opencv/4.9.0
source /home/sihanuo/projects/def-jyzhao/SihanZhang/deinterlacing/.venv/bin/activate

# 3. 创建结果保存文件夹
mkdir -p results/20260303

# 4. 执行推理命令
echo "🚀 开始执行视频推理..."
python inference.py \
-c "config/mvdeinter/v1_vanilla_fast_baseline.yaml" \
-w "work_dirs/.tmp_9549913_v1_vanilla_fast_baseline_20260301_104042/best_model.pth" \
-i "/home/sihanuo/scratch/SihanZhangData/REDS_processed_val/000/lr/interlaced_input.mp4" \
-m "/home/sihanuo/scratch/SihanZhangData/REDS_processed_val/000/mv_fwd" \
-o "results/20260303/v1_val_000.mp4" \
--csv_log "results/20260303/v1_val_000.csv"

echo "✅ 推理任务完成！"

