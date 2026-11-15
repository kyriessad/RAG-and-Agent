from abc import ABC, abstractmethod
from typing import Optional
import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from utils.config_handler import rag_conf
from utils.logger_handler import logger


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        return ChatOpenAI(
            model=rag_conf["CHAT_MODEL_NAME"],
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=rag_conf["BASE_URL"],
            timeout=30,
            max_retries=2,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return OpenAIEmbeddings(
            model=rag_conf["EMBEDDING_MODEL_NAME"],
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=rag_conf["BASE_URL"],
            check_embedding_ctx_length=False,
            chunk_size=rag_conf["BATCH_SIZE"]
        )


# 聊天模型对象
chat_model = ChatModelFactory().generator()
# 嵌入模型对象
embedding_model = EmbeddingsFactory().generator()

if __name__ == '__main__':
    logger.info(chat_model.invoke("你好，你是谁，能做什么？").content)