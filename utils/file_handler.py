"""
文件处理工具
"""
import hashlib
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from utils.logger_handler import logger
from langchain_core.documents import Document


# 获取文件的md5的十六进制字符串
def get_file_md5_hex(file_path: str) -> str | None:
    # 如果文件（文件夹不存在）
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在")
        return None
    # 如果不是文件（是文件夹）
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]路径{file_path}不是文件")
        return None

    # 哈希对象，计算器
    md5_obj = hashlib.md5()

    chunk_size = 4096  # 4KB分片，避免文件过大 爆内存
    try:
        with open(file_path, "rb") as f:  # 二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

        md5_hex = md5_obj.hexdigest()  # 转成16进制字符串
        return md5_hex
    except Exception as e:
        logger.error(f"计算文件{file_path}的md5失败, {str(e)}")  # 权限、文件被占用等其他异常情况
        return None


# 返回文件夹内的文件列表（允许的文件） allowed_types允许的文件类型 (txt,pdf,docx...)
def listdir_with_allowed_type(dir_path: str, allowed_types: tuple[str]) -> list[str]:
    files = []

    # 如果不是文件夹
    if not os.path.isdir(dir_path):
        logger.error(f"[listdir_with_allowed_type]{dir_path}不是文件夹")
        return []

    # os.listdir 把这个文件夹下的文件放进列表 [a.txt, b.pdf, c.docx....]
    for f in os.listdir(dir_path):
        if f.endswith(allowed_types):
            # files = ["dir_path\a.txt", "dir_path\b.pdf", ...]
            files.append(os.path.join(dir_path, f))

    return files


# 统一文件加载器，返回 LangChain 的 Document 列表
def file_loader(filepath: str, password: str = None) -> list[Document]:
    if not os.path.exists(filepath):
        logger.error(f"[file_loader] 文件不存在: {filepath}")
        return []

    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".txt":
            return TextLoader(filepath, encoding="utf-8").load()

        elif ext == ".pdf":
            return PyPDFLoader(filepath).load()

        elif ext == ".docx":
            return Docx2txtLoader(filepath).load()

        else:
            logger.warning(f"[file_loader] 不支持的文件格式: {ext}")
            return []

    except Exception as e:  # 文件已被打开、没有权限等异常情况
        logger.error(f"[file_loader] 加载文件 {filepath} 失败: {str(e)}")
        return []


if __name__ == '__main__':
    # print(listdir_with_allowed_type("C:/Users/Administrator/PycharmProjects/Agent项目智能扫地机器人/data", ('txt',)))
    print(file_loader("C:/Users/Administrator/PycharmProjects/Agent项目智能扫地机器人/data/故障排除.txt"))

