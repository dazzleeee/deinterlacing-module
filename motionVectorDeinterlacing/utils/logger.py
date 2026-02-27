import logging
import os
from .distributionUtil import is_master

def get_root_logger(logger_name='MVDNet', log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    # 如果已经配置过，直接返回，防止重复打印
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 只有 Rank 0 的主进程才允许输出日志，防止多卡打印刷屏
    if is_master():
        # 1. 终端输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. 文件输出
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger