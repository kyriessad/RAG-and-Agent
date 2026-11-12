"""
日志工具
"""
import logging
import os
from datetime import datetime

from utils.path_tool import get_abs_path

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")  # C:\Users\Administrator\PycharmProjects\Agent项目智能扫地机器人\logs

os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置  error info debug
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(name="agent", console_level=logging.INFO, file_level=logging.DEBUG, log_file=None)->logging.Logger:
    logger = logging.getLogger(name)  # 拿一个叫 "agent" 的记录员
    logger.setLevel(logging.DEBUG)  # 设置它记录的最低级别

    if logger.handlers:  # 如果已经配置过了,直接返回  logger.handlers = []
        return logger  # (防止重复打印同一条日志)

    # 控制台Handler
    console_handler = logging.StreamHandler()  # 控制台输出
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # 文件Handler
    if not log_file:  # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y.%m.%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


# 快捷获取日志器
logger = get_logger()


if __name__ == '__main__':
    logger.info("开始执行信息")
    logger.error("执行失败错误")
    logger.warning("天气工具调用次数接近上限")
    logger.debug("调试模式")