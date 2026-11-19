"""
提示词加载工具
"""
from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


# 加载系统提示词
def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf['system_prompt_path'])
    except KeyError as e:
        logger.error("prompts.yml配置文件中没有system_prompt_path配置项")
        raise e
    try:
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    except Exception as e:
        logger.error(f"解析system prompt出错,{str(e)}")
        raise e


