import asyncio
import logging
from typing import Any, cast

import dataiku
from fnllm import EmbeddingsLLM, LLMOutput
from fnllm.types.generics import TJsonModel, THistoryEntry
import time
import asyncio
import numpy as np

from concurrent.futures import ThreadPoolExecutor

class EmbeddingsContainer:
    def __init__(self, embeddings):
        self.embeddings = embeddings

class DataikuEmbeddingsLLM(EmbeddingsLLM):
    """
    DataikuEmbeddingsLLM integrates a Dataiku LLM embedding endpoint with the fnllm EmbeddingsLLM protocol.

    This class:
    - Connects to a Dataiku project and retrieves a configured embedding model.
    - Calls `emb_model.new_embeddings()` to create a new embedding request.
    - Adds the input prompt text to the embedding query.
    - Executes the query to fetch the embeddings.
    - Returns the embeddings wrapped in an LLMOutput object.

    Note: Since this is an embeddings model, it doesn't handle history or conversational aspects.
    """

    def __init__(self, project_key: str, embedding_model_id: str):
        """
        Initialize the DataikuEmbeddingsLLM.

        :param project_key: The Dataiku project key where the embedding model is defined.
        :param embedding_model_id: The embedding model identifier within the project.
        """
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initializing DataikuEmbeddingsLLM with project_key=%s, model_id=%s", project_key, embedding_model_id)
        self.project_key = project_key
        self.embedding_model_id = embedding_model_id

        self.client = dataiku.api_client()
        self.project = self.client.get_project(self.project_key)
        self.emb_model = self.project.get_llm(self.embedding_model_id)


    async def __call__(
        self,
        prompt: str,
        **kwargs: Any
    ) -> LLMOutput[Any, TJsonModel, THistoryEntry]:
        """
        Asynchronously generate embeddings for the given prompt.

        :param prompt: The input text to generate embeddings for.
        :param kwargs: Additional arguments (not typically used for embeddings).
        :return: LLMOutput with the embeddings in `output["content"]`.
        """
        self.logger.info("Received prompt for embedding generation: %r", prompt)

        emb_query = self.emb_model.new_embeddings()

        if isinstance(prompt, str):
            prompt = [prompt]

        for text in prompt:
            emb_query.add_text(text)


        start_time = time.perf_counter()  # Start timing
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
                emb_resp = await loop.run_in_executor(executor, emb_query.execute)
        end_time = time.perf_counter()  # End timing
        execution_time = end_time - start_time
        self.logger.info(f"Execution time: {execution_time:.4f} seconds")

        #emb_resp = emb_query.execute()
        
        # Retrieve embeddings from response
        #embeddings = emb_resp.get_embeddings()[0] if emb_resp.get_embeddings() else []
        embeddings = emb_resp.get_embeddings()
        
        self.logger.info("Retrieved embeddings from response: length=%d", len(embeddings))

        # Construct the LLMOutput
        llm_output = LLMOutput(
            output=EmbeddingsContainer(embeddings),
            parsed_json=cast("TJsonModel", None),
        )
        self.logger.debug("Successfully generated embeddings for prompt.")
        return llm_output

    def child(self, name: str):
        """
        Create a child LLM instance. Not applicable for embeddings models.

        Since embeddings typically do not involve hierarchical or named child instances,
        this method is not implemented.
        """
        self.logger.debug("child() method called with name=%s, but not implemented.", name)
        raise NotImplementedError("Child instances are not supported by DataikuEmbeddingsLLM.")