from fnllm import EmbeddingsLLM
from fnllm.types.generics import TJsonModel, THistoryEntry, TModelParameters
from fnllm import LLMOutput
from typing import Any, cast
import dataiku

class DataikuEmbeddingsLLM(EmbeddingsLLM):
    def __init__(self, project_key: str, embedding_model_id: str):
        self.project_key = project_key
        self.embedding_model_id = embedding_model_id
        self.client = dataiku.api_client()
        self.project = self.client.get_project(project_key)
        self.emb_model = self.project.get_llm(self.embedding_model_id)

    async def __call__(
        self,
        prompt: str,
        **kwargs
    ) -> LLMOutput[Any, TJsonModel, THistoryEntry]:
        # This should return embeddings for the given prompt text
        emb_query = self.emb_model.new_embeddings()
        emb_query.add_text(prompt)
        emb_resp = emb_query.execute()
        embeddings = emb_resp.get_embeddings()[0]  # single input case

        # LLMOutput expects an output and parsed_json. EmbeddingsLLM 
        # can just store embeddings in output, possibly as a custom object
        # or you can define a schema.
        # Let's store embeddings in output as a content field:
        return LLMOutput(
            output={"content": embeddings},
            parsed_json=cast("TJsonModel", None),
        )

    def child(self, name):
        # If needed, implement child logic or raise NotImplementedError
        raise NotImplementedError