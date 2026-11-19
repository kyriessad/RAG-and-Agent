"""
根据用户的提示词获取相关文本块
提示词转成向量，再在向量数据库上检索

get_retriever返回检索器，可以加入链调用invoke返回相关文本块
"""
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import file_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from uuid import uuid5, NAMESPACE_URL


# 向量存储对象
class VectorStoreService(object):
    def __init__(self, ):
        # 数据库本地存储文件夹；目录不存在时先建好，避免 chromadb 直接报 code 14
        self.persist_directory = get_abs_path(chroma_conf["persist_directory"])
        os.makedirs(self.persist_directory, exist_ok=True)

        # 向量存储的实例对象
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],  # 数据库的表名
            embedding_function=embedding_model,
            persist_directory=self.persist_directory,
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
    def load_document(self) -> dict[str, int]:
        # 获取资料目录的绝对路径  C:\Users\Administrator\PycharmProjects\Agent项目智能扫地机器人\data
        data_dir = get_abs_path(chroma_conf["data_path"])
        # 整个资料目录不存在时，停止同步，保留数据库
        if not os.path.isdir(data_dir):
            raise RuntimeError(f"知识库资料目录不存在：{data_dir}")
        # 允许处理的文件(路径)
        paths = listdir_with_allowed_type(
            data_dir,
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        # 目录可以为空，因为可能需要清理已经删除的资料
        if not paths:
            logger.warning("资料目录中没有可导入的文件，将检查旧记录")

        updated = 0  # 更新的文件数
        skipped = 0  # 没变的文件数

        # 更新文件，删除修改了的文件
        for path in sorted(paths):
            # 1. 计算文件当前内容的指纹
            md5_hex = get_file_md5_hex(path)
            if md5_hex is None:
                raise RuntimeError(f"无法计算文件指纹：{path}")

            # 2. 查询这个文件以前存入数据库的记录
            """
                        old_records = {
                            "ids": [
                                "uuid-001",
                                "uuid-002",
                                # ...
                                "uuid-051"
                            ],
                            "metadatas": [
                                {
                                    "source": path,
                                    "file_md5": "abed6409...",
                                    "file_chunk_count": 51
                                },
                                {
                                    "source": path,
                                    "file_md5": "abed6409...",
                                    "file_chunk_count": 51
                                },
                                # ...共51个元数据字典
                            ]
                        }
            """
            old_records: dict[str, []] = self.vector_store.get(
                where={"source": path},
                include=["metadatas"],
            )

            old_ids = old_records["ids"]
            old_metadatas: list = old_records["metadatas"] or []

            # 3. 判断这个文件是否需要重新导入  假设文件没有变化，后面发现问题就改成 False
            unchanged = True
            # 数据库中实际查到的文本块数量
            old_count = len(old_ids)
            if old_count == 0:
                # 没有旧记录，说明需要导入
                unchanged = False
            elif len(old_metadatas) != old_count:
                # 文本块数量与元数据数量对不上，需要重新导入
                unchanged = False
            else:
                # 逐个检查旧文本块的元数据 metadata = {"source": path, "file_md5": "abed6409...", "file_chunk_count": 51}
                for metadata in old_metadatas:
                    if not metadata:
                        # 这个文本块缺少元数据
                        unchanged = False
                        break
                    old_md5 = metadata.get("file_md5")
                    if old_md5 != md5_hex:
                        # 旧版本的 MD5 与当前文件不同，说明内容变了
                        unchanged = False
                        break
                    expected_count = metadata.get("file_chunk_count")
                    if expected_count != old_count:
                        # 记录的总数量与实际数量不同，说明记录不完整
                        unchanged = False
                        break
            # 全部检查通过，才跳过这个文件
            if unchanged:
                skipped += 1
                logger.info(f"[知识库同步]未变化，跳过：{path}")
                continue

            # 4. 读取文件并切分
            documents: list[Document] = file_loader(path)

            if not documents:
                raise RuntimeError(f"文件读取失败或没有内容：{path}")

            chunks: list[Document] = self.spliter.split_documents(documents)

            # 如果为空
            if not chunks or not any(
                    chunk.page_content.strip() for chunk in chunks
            ):
                raise RuntimeError(f"文件没有可用于检索的文本：{path}")

            # 5. 给每个文本块记录来源、版本和总数量
            for chunk in chunks:
                chunk.metadata.update({
                    "source": path,
                    "file_md5": md5_hex,
                    "file_chunk_count": len(chunks),
                })

            # 同一个文件、同一个版本、同一个序号，得到相同的 ID
            new_ids = []
            for index in range(len(chunks)):
                # 拼出唯一的内容字符串
                unique_text = f"{path}|{md5_hex}|{index}"
                # 算出唯一 ID 只要文件内容改变，里面的所有块的id都会变，根据unique_text算id
                chunk_id = str(uuid5(NAMESPACE_URL, unique_text))
                # 塞进列表
                new_ids.append(chunk_id)

            # 6. 先写入新版；发生异常时，不继续删除旧版
            self.vector_store.add_documents(documents=chunks, ids=new_ids,)

            # 7. 只删除这个文件不再使用的旧文本块
            new_id_set = set(new_ids)
            obsolete_ids = [old_id for old_id in old_ids if old_id not in new_id_set]
            if obsolete_ids:
                self.vector_store.delete(ids=obsolete_ids)
            updated += 1
            logger.info(
                f"[知识库同步]更新完成：{path}，"
                f"当前 {len(chunks)} 个文本块"
            )

        # 如果文件被删了，知识库里的内容依然存在。检查知识库中的source是否都存在
        # 读取数据库中所有文本块的 ID 和元数据
        records = self.vector_store.get(include=["metadatas"])
        deleted_ids = []
        for index in range(len(records["ids"])):
            metadata: dict = records["metadatas"][index]
            # 无法确定来源的记录，先保留
            if not metadata:
                continue
            source = metadata.get("source")
            if not source:
                continue
            # 原文件不存在，就记录这个文本块的 ID
            if not os.path.exists(source):
                deleted_ids.append(records["ids"][index])
        # 一次性删除这些文本块
        if deleted_ids:
            self.vector_store.delete(ids=deleted_ids)
            logger.info(f"[知识库同步]删除了 {len(deleted_ids)} 个文本块")

        return {
            "updated": updated,
            "skipped": skipped,
            "deleted_chunks": len(deleted_ids),
        }


if __name__ == '__main__':
    # 导入数据
    vs = VectorStoreService()
    vs.load_document()
