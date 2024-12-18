
import dku_graphrag.query.query_monkey_patch_dataiku 



import asyncio
import json
from pathlib import Path
import pandas as pd
import logging
import dataiku
from dataiku.llm.python import BaseLLM
from dataiku.langchain.dku_tracer import LangchainToDKUTracer
from graphrag.config.load_config import load_config
from graphrag.api.query import global_search, local_search

def ensure_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def get_span_builder_for_callback_manager_2(callbacks):
    if hasattr(callbacks, "handlers"):
        for handler in callbacks.handlers:
            if isinstance(handler, LangchainToDKUTracer):
                if hasattr(callbacks, "parent_run_id"):
                    return handler.run_id_to_span_map[str(callbacks.parent_run_id)]
                else:
                    raise Exception("Callbacks %s don't have a parent_run_id" % callbacks)
        raise Exception("Callbacks %s don't have a LangchainToDKUTracer handler" % callbacks)
    else:
        raise Exception("Callbacks %s don't have handlers" % callbacks)


class MyLLM(BaseLLM):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        pass

    def set_config(self, config, plugin_config):
        self.config = config
        self.folder_path  = Path(dataiku.Folder(config.get("index_folder_id") ).get_path())
        self.search_type = config.get("search_type", "local")       
        self.default_community_level = config.get("default_community_level", 0) 
        self.response_type = config.get("response_type", "multiple paragraphs")    
        self.graphrag_config = load_config(self.folder_path)  
        if "db_uri" in self.graphrag_config.embeddings.vector_store:
        # Update the 'db_uri' value
            self.graphrag_config.embeddings.vector_store["db_uri"] = str(self.folder_path  / self.graphrag_config.embeddings.vector_store["db_uri"])
        else:
            print("Error: 'db_uri' key not found in vector_store")  
        index_output_folder = self.folder_path  / Path(self.graphrag_config.storage.base_dir) 
        self.nodes = pd.read_parquet(index_output_folder / 'create_final_nodes.parquet')
        self.entities = pd.read_parquet(index_output_folder / 'create_final_entities.parquet')
        self.communities = pd.read_parquet(index_output_folder / 'create_final_communities.parquet')
        self.community_reports = pd.read_parquet(index_output_folder / 'create_final_community_reports.parquet')
        self.text_units = None
        self.relationships = None
        self.covariates_df = None
        if self.search_type == "local":
            self.text_units = pd.read_parquet(index_output_folder / 'create_final_text_units.parquet')
            self.relationships = pd.read_parquet(index_output_folder / 'create_final_relationships.parquet')
            covariates_path = index_output_folder / 'create_final_covariates.parquet'
            # covariates are optional
            if covariates_path.exists():
                self.covariates_df = pd.read_parquet(covariates_path)
        self.logger.info(f"Agent config initialized with: search_type={self.search_type}, folder_path ={self.folder_path}, default_community_level={self.default_community_level},  response_type ={self.response_type }")

    async def search(self, query: str):
        self.logger.info(f"Agent search: search_type={self.search_type}, query ={query}")
        if self.search_type == "global":
            response, context = await global_search(
                config=self.graphrag_config,
                nodes=self.nodes,
                entities=self.entities,
                communities=self.communities,
                community_reports=self.community_reports,
                community_level=None,
                dynamic_community_selection=False,
                response_type=self.response_type,
                query=query
            )
        else:
            response, context = await local_search(
                config=self.graphrag_config,
                nodes=self.nodes,
                entities=self.entities,
                community_reports=self.community_reports,
                text_units=self.text_units,
                relationships=self.relationships,
                covariates=self.covariates_df,
                community_level=self.default_community_level,
                response_type=self.response_type,
                query=query
            )
        return response, context 

    def process(self, query, settings, trace):
        query = query["messages"][0]["content"]
        loop = ensure_event_loop()
        response, context = loop.run_until_complete(self.search(query))
        context_str = json.dumps(context, indent=2)
        resp_text = f"{response}\n\n\n###### context: ######\n{context_str}"
        return {"text": resp_text}
           
