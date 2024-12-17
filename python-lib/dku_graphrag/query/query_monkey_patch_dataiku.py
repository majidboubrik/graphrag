from dku_graphrag.query.query_dataiku_chat_llm import QueryDataikuChatLLM
from dku_graphrag.query.query_dataiku_embedding_llm import QueryDataikuEmbeddingLLM

import graphrag.query.llm.get_client

def monkeypatched_get_llm(config):
    llm_id = "openai:bs-openai:gpt-4o-mini"
    return QueryDataikuChatLLM(llm_id)

def monkeypatched_get_text_embedder(config):
    embedding_model_id = "openai:bs-openai:text-embedding-3-small"
    return QueryDataikuEmbeddingLLM(embedding_model_id)

graphrag.query.llm.get_client.get_llm = monkeypatched_get_llm
graphrag.query.llm.get_client.get_text_embedder = monkeypatched_get_text_embedder

print("  =============== Monkey patching query LLMs DONE")