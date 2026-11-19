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

    def search_with_sources(self, query: str) -> tuple[str, list[dict]]:
        """输入问题，返回给模型的文本和给页面的原文列表。调用嵌入服务并读取索引，失败时抛出异常。"""
        parts = []
        sources = []  # 例如 [{"source": "家庭血压测量与记录.txt", "excerpt": "……"}]。
        for doc in self.retriever_docs(query):
            # 文件名和原文都来自索引。缺失来源时跳过，避免给回答附上无法核对的材料。
            name = os.path.basename(doc.metadata.get("source") or "")
            if not name or not doc.page_content.strip():
                continue
            entry = {"source": name, "excerpt": doc.page_content}
            if entry in sources:
                continue
            sources.append(entry)
            parts.append(f"【参考资料{len(sources)}】\n来源：{name}\n{doc.page_content}")
        if not sources:
            return "知识库未检索到可核对的原文，当前资料不足以回答。", []
        return "\n\n".join(parts), sources
