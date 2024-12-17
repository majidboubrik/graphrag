# monkey_patch_all_dataiku.py
from dataiku_chat_llm import DataikuChatLLM
from dataiku_embeddings_llm import DataikuEmbeddingsLLM
from graphrag.config.enums import LLMType
from graphrag.index.llm.load_llm import loaders




def _load_dataiku_chat_llm(on_error, cache, config):

    project_key = "AGENTSANDBOX"
    llm_id = "openai:bs-openai:gpt-4o-mini"
    return DataikuChatLLM(project_key, llm_id)

def _load_dataiku_embeddings_llm(on_error, cache, config):
    project_key = "AGENTSANDBOX"
    embedding_model_id = "openai:bs-openai:text-embedding-3-small"
    return DataikuEmbeddingsLLM(project_key, embedding_model_id)

# Override OpenAIChat loader
loaders[LLMType.OpenAIChat] = {
    "load": _load_dataiku_chat_llm,
    "chat": True,
}

# Override the azure_openai_embedding entry to load our embeddings LLM
loaders[LLMType.OpenAIEmbedding] = {
    "load": _load_dataiku_embeddings_llm,
    "chat": False,
}
print("  =============== Monkey patching loaders DONE ")