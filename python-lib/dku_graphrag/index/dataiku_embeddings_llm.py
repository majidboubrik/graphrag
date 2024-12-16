import asyncio
import logging
from typing import Any, cast

import dataiku
from fnllm import EmbeddingsLLM, LLMOutput
from fnllm.types.generics import TJsonModel, THistoryEntry




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
        self.logger.debug("Received prompt for embedding generation: %r", prompt)

        # Create a new embeddings request
        emb_query = self.emb_model.new_embeddings()

        # Add the prompt text
        emb_query.add_text(prompt)

        # Execute the query (this is synchronous, so if very large or long it might block)
        # If needed, this could be run in a thread executor for true async, but not required if the API is fast enough.
        try:
            emb_resp = emb_query.execute()
        except Exception as e:
            self.logger.exception("Error executing embeddings query: %s", e)
            # Return an empty embedding or handle error as needed
            return LLMOutput(
                output={"content": []},
                parsed_json=cast("TJsonModel", None),
            )

        # Retrieve embeddings from response
        embeddings = emb_resp.get_embeddings()[0] if emb_resp.get_embeddings() else []
        self.logger.debug("Retrieved embeddings from response: length=%d", len(embeddings))

        # Construct the LLMOutput
        llm_output = LLMOutput(
            output={"content": embeddings},
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