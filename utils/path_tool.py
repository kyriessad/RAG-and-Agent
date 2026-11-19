"""
路径工具，为工程提供绝对路径
"""

import os


# 获取项目的根目录
def get_project_root() -> str:
    # 当前文件的绝对路径  C:\Users\Administrator\PycharmProjects\Agent项目智能扫地机器人\utils\path_tool.py
    current_file = os.path.abspath(__file__)

    current_dir = os.path.dirname(current_file)

    # C:\Users\Administrator\PycharmProjects\Agent项目智能扫地机器人
    project_root = os.path.dirname(current_dir)

    return project_root


# 获得传入文件的绝对路径
def get_abs_path(file_relative_path: str) -> str:
    return os.path.join(get_project_root(), file_relative_path)
