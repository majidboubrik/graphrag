import asyncio
import logging
from typing import Any
import time 
import dataiku
from graphrag.query.llm.base import BaseTextEmbedding
from concurrent.futures import ThreadPoolExecutor


class QueryDataikuEmbeddingLLM(BaseTextEmbedding):
    def __init__(self, embedding_model_id: str):
        self.embedding_model_id = embedding_model_id
        self.client = dataiku.api_client()
        self.project = self.client.get_default_project()
        self.emb_model = self.project.get_llm(self.embedding_model_id)
        self.logger = logging.getLogger(__name__)

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        # Assuming the Dataiku LLM embedding endpoint:
        # If not available, you'll need to handle differently.
        # The hypothetical `self.llm.compute_embedding(text)` method returns an embedding vector (list of floats).
        
        self.emb_model.new_embeddings()

        emb_query = self.emb_model.new_embeddings()

        emb_query.add_text(text)

        start_time = time.perf_counter()  # Start timing
        emb_resp = emb_query.execute()
        end_time = time.perf_counter()  # End timing
        execution_time = end_time - start_time
        self.logger.info(f"Execution time: {execution_time:.4f} seconds")
        embeddings = emb_resp.get_embeddings()
        return embeddings[0] if embeddings else [] 

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            emb_resp = await loop.run_in_executor(executor, self.embed, text, **kwargs)
        return emb_resp