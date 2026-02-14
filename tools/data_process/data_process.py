#!/usr/bin/env python3
"""
REDS数据集隔行视频生成与运动向量提取工具

功能:
    1. 将逐行GT帧转换为隔行场图像
    2. 分别对Top/Bot场流进行H.264编码
    3. 从编码结果中提取运动向量(MV)


"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import subprocess
import glob
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# =============================================================================
# 全局配置
# =============================================================================

# 运动向量提取工具路径
MV_EXTRACTOR_PATH = "./extract_mvs"

# FFmpeg/x264 编码参数
X264_PARAMS = "me=umh:subme=10:merange=32:blocks=all"
FPS = 25  # 帧率
FRAME_DURATION = 1.0 / FPS  # 每帧时长(秒)

# 图像尺寸常量
MIN_DIMENSION = 1

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 核心函数
# =============================================================================

def parse_motion_vector_csv(
    csv_path: Path, 
    frame_height: int, 
    frame_width: int,
    debug: bool = False
) -> Dict[int, np.ndarray]:
    """
    解析运动向量CSV文件，生成像素级光流场
    
    CSV格式说明:
        - framenum: 帧编号(从1开始，1为I帧无MV)
        - source: -1表示前向参考(参考前一帧)
        - motion_x, motion_y: 原始运动向量值
        - motion_scale: 运动向量缩放因子
        - dstx, dsty: 宏块中心点坐标
        - blockw, blockh: 宏块半宽、半高
    
    Args:
        csv_path: CSV文件路径
        frame_height: 图像高度
        frame_width: 图像宽度
        debug: 是否打印调试信息
    
    Returns:
        字典: {流内帧索引: flow数组}, flow数组shape为(2, H, W)
              flow[0]为x方向运动, flow[1]为y方向运动
    """
    # 检查文件有效性
    if not csv_path.exists():
        logger.warning(f"CSV文件不存在: {csv_path}")
        return {}
    
    if csv_path.stat().st_size == 0:
        logger.warning(f"CSV文件为空: {csv_path}")
        return {}
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # 清理列名空格
    
    if debug:
        _print_csv_debug_info(df, frame_width, frame_height)
    
    motion_vectors = {}
    
    for _, row in df.iterrows():
        # 只处理前向参考帧(source=-1)
        if row['source'] != -1:
            continue
        
        # 转换帧索引: framenum从1开始，转为0-based
        # framenum=2表示第2帧的MV(指向第1帧)，对应索引1
        frame_idx = int(row['framenum']) - 1
        
        # 获取缩放因子
        scale = float(row['motion_scale'])
        if scale == 0:
            logger.debug(f"帧{frame_idx}: scale为0，跳过")
            continue
        
        # 计算实际运动向量
        mv_x = row['motion_x'] / scale
        mv_y = row['motion_y'] / scale
        
        # 获取宏块信息(dstx/dsty是中心点坐标)
        center_x = int(row['dstx'])
        center_y = int(row['dsty'])
        half_width = int(row['blockw'])
        half_height = int(row['blockh'])
        
        # 初始化该帧的flow数组
        if frame_idx not in motion_vectors:
            motion_vectors[frame_idx] = np.zeros(
                (2, frame_height, frame_width), 
                dtype=np.float32
            )
        
        # 计算宏块边界(从中点向两侧扩展)
        x_start = max(0, center_x - half_width)
        y_start = max(0, center_y - half_height)
        x_end = min(frame_width, center_x + half_width)
        y_end = min(frame_height, center_y + half_height)
        
        # 填充运动向量到对应区域
        if x_start < x_end and y_start < y_end:
            # 使用广播将(mv_x, mv_y)填充到整个块
            motion_vectors[frame_idx][:, y_start:y_end, x_start:x_end] = \
                np.array([mv_x, mv_y])[:, None, None]
    
    return motion_vectors


def _print_csv_debug_info(df: pd.DataFrame, width: int, height: int) -> None:
    """打印CSV调试信息"""
    print("\n" + "=" * 60)
    print("CSV数据调试信息")
    print("=" * 60)
    print(f"\n前10行数据:")
    print(df[['framenum', 'source', 'dstx', 'dsty', 'blockw', 'blockh']].head(10))
    print(f"\n图像尺寸: {width} x {height}")
    print(f"dstx范围: {df['dstx'].min()} ~ {df['dstx'].max()}")
    print(f"dsty范围: {df['dsty'].min()} ~ {df['dsty'].max()}")
    print(f"blockw范围: {df['blockw'].min()} ~ {df['blockw'].max()}")
    print(f"blockh范围: {df['blockh'].min()} ~ {df['blockh'].max()}")
    print("=" * 60 + "\n")


def create_ffmpeg_concat_file(
    image_paths: List[str], 
    output_file: Path
) -> None:
    """
    创建FFmpeg concat demuxer所需的输入文件列表
    
    FFmpeg concat格式要求:
        file '/absolute/path/to/image1.png'
        duration 0.04
        file '/absolute/path/to/image2.png'
        duration 0.04
        ...
        file '/absolute/path/to/last.png'  # 最后一帧无需duration
    
    Args:
        image_paths: 图片路径列表(按时间顺序)
        output_file: 输出的list.txt文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, img_path in enumerate(image_paths):
            # 转换为绝对路径，确保跨平台兼容
            abs_path = Path(img_path).resolve()
            # 统一使用正斜杠(FFmpeg要求)
            safe_path = str(abs_path).replace('\\', '/')
            
            f.write(f"file '{safe_path}'\n")
            
            # 最后一帧不写duration(FFmpeg concat格式要求)
            if i < len(image_paths) - 1:
                f.write(f"duration {FRAME_DURATION:.4f}\n")


def encode_video_and_extract_mv(
    image_paths: List[str],
    temp_dir: Path,
    stream_name: str,
    frame_height: int,
    frame_width: int,
    debug: bool = False
) -> Dict[int, np.ndarray]:
    """
    将一组图片编码为视频并提取运动向量
    
    处理流程:
        1. 创建concat输入文件
        2. FFmpeg编码为H.264视频
        3. 使用extract_mvs工具提取运动向量CSV
        4. 解析CSV为flow数组
        5. 清理临时文件
    
    Args:
        image_paths: 图片路径列表
        temp_dir: 临时文件目录
        stream_name: 流名称(用于命名临时文件，如'top'或'bot')
        frame_height: 图像高度
        frame_width: 图像宽度
        debug: 是否打印调试信息
    
    Returns:
        字典: {流内帧索引: flow数组}
    """
    # 至少需要2帧才能产生运动向量
    if len(image_paths) < 2:
        logger.warning(f"{stream_name}流帧数不足2，跳过")
        return {}
    
    # 定义临时文件路径
    concat_list_path = temp_dir / f"{stream_name}_concat.txt"
    video_path = temp_dir / f"{stream_name}.mp4"
    csv_path = temp_dir / f"{stream_name}.csv"
    
    try:
        # Step 1: 创建concat输入文件
        create_ffmpeg_concat_file(image_paths, concat_list_path)
        
        # Step 2: FFmpeg编码
        _run_ffmpeg_encoding(concat_list_path, video_path)
        
        # Step 3: 提取运动向量CSV
        _extract_motion_vector_csv(video_path, csv_path)
        
        # Step 4: 解析CSV
        motion_vectors = parse_motion_vector_csv(
            csv_path, frame_height, frame_width, debug
        )
        
        return motion_vectors
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{stream_name}流处理失败: {e}")
        return {}
    
    #finally:
    #    # 清理临时文件
    #    _cleanup_temp_files([concat_list_path, video_path, csv_path])


def _run_ffmpeg_encoding(input_list: Path, output_video: Path) -> None:
    """执行FFmpeg编码"""
    cmd = [
        "ffmpeg", "-y",                    # 覆盖输出
        "-f", "concat", "-safe", "0",      # concat输入格式
        "-i", str(input_list),             # 输入文件
        "-c:v", "libx264",                 # H.264编码器
        "-crf", "18",                      # 质量参数(越小越好)
        "-pix_fmt", "yuv420p",             # 像素格式
        "-bf", "0",                        # 禁用B帧(保证MV简单)
        "-g", "1000",                      # GOP大小(减少I帧)
        "-refs", "1",                      # 单参考帧
        "-x264-params", X264_PARAMS,       # x264高级参数
        str(output_video)
    ]
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"FFmpeg编码失败:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _extract_motion_vector_csv(video_path: Path, csv_path: Path) -> None:
    """从视频中提取运动向量CSV"""
    with open(csv_path, 'w', encoding='utf-8') as csv_file:
        result = subprocess.run(
            [MV_EXTRACTOR_PATH, str(video_path)],
            stdout=csv_file,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"MV提取失败:\n{result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, [MV_EXTRACTOR_PATH])


def _cleanup_temp_files(file_paths: List[Path]) -> None:
    """清理临时文件"""
    for path in file_paths:
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                logger.warning(f"无法删除临时文件 {path}: {e}")


def extract_interlaced_field(
    frame: np.ndarray, 
    is_top_field: bool
) -> np.ndarray:
    """
    从逐行帧中提取隔行场
    
    Args:
        frame: 输入帧，shape为(H, W, 3)
        is_top_field: True提取Top场(偶数行)，False提取Bot场(奇数行)
    
    Returns:
        场图像，shape为(H/2, W, 3)
    """
    height = frame.shape[0]
    
    if is_top_field:
        # Top场: 第0, 2, 4, ... 行
        return frame[0:height:2, :, :]
    else:
        # Bot场: 第1, 3, 5, ... 行
        return frame[1:height:2, :, :]


def process_single_clip(
    gt_clip_dir: Path, 
    output_root: Path,
    debug: bool = False
) -> Tuple[int, int]:
    """
    处理单个视频片段
    
    Args:
        gt_clip_dir: GT帧目录
        output_root: 输出根目录
        debug: 是否打印调试信息
    
    Returns:
        (top_mv_count, bot_mv_count): 提取的MV数量
    """
    clip_name = gt_clip_dir.name
    clip_output_dir = output_root / clip_name
    
    # 创建输出子目录
    lr_output_dir = clip_output_dir / "lr"          # 隔行场图像
    mv_output_dir = clip_output_dir / "mv_fwd"      # 前向运动向量
    meta_output_dir = clip_output_dir / "meta"      # 元数据
    
    for directory in [lr_output_dir, mv_output_dir, meta_output_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 获取GT帧列表
    gt_frame_paths = sorted(glob.glob(str(gt_clip_dir / "*.png")))
    if not gt_frame_paths:
        logger.warning(f"片段 {clip_name} 中没有找到PNG文件")
        return (0, 0)
    
    # 读取第一帧获取尺寸信息
    first_frame = cv2.imread(gt_frame_paths[0])
    if first_frame is None:
        logger.error(f"无法读取帧: {gt_frame_paths[0]}")
        return (0, 0)
    
    full_height, full_width = first_frame.shape[:2]
    field_height = full_height // 2
    
    # ==========================================================================
    # Step 1: 隔行采样 - 从逐行帧提取场
    # ==========================================================================
    
    field_paths = []              # 所有场图像路径
    field_type_labels = []        # 场类型标签(0=Top, 1=Bot)
    
    for frame_idx, gt_path in enumerate(gt_frame_paths):
        frame = cv2.imread(gt_path)
        if frame is None:
            logger.warning(f"无法读取帧: {gt_path}")
            continue
        
        # 偶数帧索引(0,2,4,...) → Top场
        # 奇数帧索引(1,3,5,...) → Bot场
        is_top_field = (frame_idx % 2 == 0)
        
        # 提取场
        field = extract_interlaced_field(frame, is_top_field)
        
        # 保存场图像
        field_filename = f"{frame_idx:08d}.png"
        field_save_path = lr_output_dir / field_filename
        cv2.imwrite(str(field_save_path), field)
        
        field_paths.append(str(field_save_path))
        field_type_labels.append(0 if is_top_field else 1)
    
    # 保存元数据
    np.save(meta_output_dir / "field_ids.npy", np.array(field_type_labels, dtype=np.int8))
    np.savez(
        meta_output_dir / "video_info.npz",
        original_height=full_height,
        original_width=full_width,
        field_height=field_height,
        num_frames=len(field_paths)
    )
    
    # ==========================================================================
    # Step 2: 分组 - 将Top场和Bot场分开
    # ==========================================================================
    
    # Top组: 原始帧索引 0, 2, 4, ...
    # Bot组: 原始帧索引 1, 3, 5, ...
    top_field_indices = list(range(0, len(field_paths), 2))
    bot_field_indices = list(range(1, len(field_paths), 2))
    
    top_field_paths = [field_paths[i] for i in top_field_indices]
    bot_field_paths = [field_paths[i] for i in bot_field_indices]
    
    # ==========================================================================
    # Step 3: 分别编码并提取运动向量
    # ==========================================================================
    
    logger.info(f"  处理Top流 ({len(top_field_paths)} 帧)...")
    top_motion_vectors = encode_video_and_extract_mv(
        top_field_paths, 
        clip_output_dir, 
        "top", 
        field_height, 
        full_width,
        debug
    )
    
    logger.info(f"  处理Bot流 ({len(bot_field_paths)} 帧)...")
    bot_motion_vectors = encode_video_and_extract_mv(
        bot_field_paths, 
        clip_output_dir, 
        "bot", 
        field_height, 
        full_width,
        debug
    )
    
    # ==========================================================================
    # Step 4: 映射回原始帧索引并保存
    # ==========================================================================
    
    # 映射关系:
    #   流内索引 stream_idx → 原始帧索引 real_idx
    #   top_motion_vectors[stream_idx] 对应 top_field_indices[stream_idx]
    
    # 保存Top流MV
    for stream_idx, real_idx in enumerate(top_field_indices):
        if stream_idx in top_motion_vectors:
            output_filename = f"{real_idx:08d}_mv_fwd.npz"
            np.savez_compressed(
                mv_output_dir / output_filename,
                flow_fwd=top_motion_vectors[stream_idx]
            )
    
    # 保存Bot流MV
    for stream_idx, real_idx in enumerate(bot_field_indices):
        if stream_idx in bot_motion_vectors:
            output_filename = f"{real_idx:08d}_mv_fwd.npz"
            np.savez_compressed(
                mv_output_dir / output_filename,
                flow_fwd=bot_motion_vectors[stream_idx]
            )
    
    return (len(top_motion_vectors), len(bot_motion_vectors))


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="REDS数据集隔行视频生成与运动向量提取工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--hr_root", 
        required=True,
        help="REDS GT帧根目录"
    )
    parser.add_argument(
        "--out_root", 
        required=True,
        help="输出根目录"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印调试信息"
    )
    
    args = parser.parse_args()
    
    # 验证输入目录
    hr_root = Path(args.hr_root)
    if not hr_root.exists():
        logger.error(f"输入目录不存在: {hr_root}")
        return
    
    # 创建输出目录
    output_root = Path(args.out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 获取所有片段目录
    clip_dirs = sorted([d for d in hr_root.iterdir() if d.is_dir()])
    
    if not clip_dirs:
        logger.error(f"未找到片段目录: {hr_root}")
        return
    
    logger.info(f"找到 {len(clip_dirs)} 个片段")
    logger.info(f"输出目录: {output_root}")
    
    # 处理每个片段
    for clip_dir in tqdm(clip_dirs, desc="处理进度"):
        top_count, bot_count = process_single_clip(
            clip_dir, 
            output_root,
            debug=args.debug
        )
        logger.info(f"✓ {clip_dir.name}: Top MVs={top_count}, Bot MVs={bot_count}")
    
    logger.info("全部处理完成!")


if __name__ == "__main__":
    main()