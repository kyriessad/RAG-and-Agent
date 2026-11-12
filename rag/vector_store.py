"""
根据用户的提示词获取相关文本块
提示词转成向量，再在向量数据库上检索

get_retriever返回检索器，可以加入链调用invoke返回相关文本块
"""
import os.path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import file_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


# 向量存储对象
class VectorStoreService(object):
    def __init__(self, ):
        # 向量存储的实例对象
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],  # 数据库的表名
            embedding_function=embedding_model,
            persist_directory=chroma_conf["persist_directory"]  # 数据库本地存储文件夹
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    # 返回向量检索器(runnable下的类)，可以加入链
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})  # 返回几个匹配的文本块

    # 从数据文件夹内读数据文件、传入向量库，要计算文件的md5去重
    def load_document(self) -> None:
        # 检查字符串是否存在在md5.txt文件中
        def check_md5(md5: str) -> bool:
            file_abs_path = get_abs_path(chroma_conf["md5_hex_store"])  # md5.txt的绝对路径
            if not os.path.exists(file_abs_path):
                open(file_abs_path, "w", encoding="utf-8").close()  # 创建md5.txt文件
                return False

            with open(file_abs_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    if md5 == line.strip():
                        return True  # str字符串在向量库中存在
                return False  # 不在向量库

        # 把md5字符串存入md5.txt文件
        def save_md5(md5: str) -> None:
            if not check_md5(md5):
                with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding='utf-8') as f:
                    f.write(md5 + "\n")

        # 拿到文件的数据(返回值是list[Document]类型)
        def get_file_documents(read_path: str) -> list[Document]:
            return file_loader(read_path)

        # 允许加入向量库的文件(路径)
        allowed_file_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_file_path:
            # 获取文件md5
            md5_hex = get_file_md5_hex(path)
            if check_md5(md5_hex):
                logger.info(f"[加载知识库]{path}已经存在于知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{documents}没有内容，跳过")
                    continue
                # 分片
                split_document: list[Document] = self.spliter.split_documents(documents)
                # 内容存入向量库
                self.vector_store.add_documents(split_document)
                # 记录该文件的md5
                save_md5(md5_hex)
                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path} 加载失败 {str(e)}", exc_info=True)  # 记录详细的报错堆栈
                continue


if __name__ == '__main__':
    # 导入数据
    vs = VectorStoreService()
    vs.load_document()

    # res = vs.get_retriever().invoke("机器人")
    # for r in res:
    #     print(r.page_content)
