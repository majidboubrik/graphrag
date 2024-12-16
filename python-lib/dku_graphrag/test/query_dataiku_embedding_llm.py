import asyncio
import logging
from typing import Any

import dataiku
from graphrag.query.llm.base import BaseTextEmbedding

embedding_logger = logging.getLogger("QueryDataikuEmbeddingLLM")
embedding_logger.setLevel(logging.DEBUG)
embedding_handler = logging.FileHandler("/tmp/graphrag/query_dataiku_embedding_llm.log")
embedding_handler.setLevel(logging.DEBUG)
embedding_logger.addHandler(embedding_handler)

class QueryDataikuEmbeddingLLM(BaseTextEmbedding):
    def __init__(self, project_key: str, embedding_model_id: str):
        self.project_key = project_key
        self.embedding_model_id = embedding_model_id
        self.client = dataiku.api_client()
        self.project = self.client.get_project(self.project_key)
        self.llm = self.project.get_llm(self.embedding_model_id)

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        # Assuming the Dataiku LLM embedding endpoint:
        # If not available, you'll need to handle differently.
        # The hypothetical `self.llm.compute_embedding(text)` method returns an embedding vector (list of floats).
        embedding = self.llm.compute_embedding(text)
        return embedding

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, text, **kwargs)