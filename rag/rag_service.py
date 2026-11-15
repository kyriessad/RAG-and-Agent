from rag.vector_store import VectorStoreService
from langchain_core.documents import Document
import os


class RagService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()

    # 检索出来的文档Document列表
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    # 用户提问，得到相关文本块
    def rag_search(self, query: str) -> str:
        # 检索相关文本块
        context_docs = self.retriever_docs(query)
        if not context_docs:
            return "知识库未检索到相关资料。"
        # 整理资料，直接返回给 Agent
        parts = []
        for index, doc in enumerate(context_docs, start=1):
            source = os.path.basename(doc.metadata.get("source", "未知来源"))
            parts.append(
                f"【参考资料{index}】\n"
                f"来源：{source}\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)


