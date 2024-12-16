from dataiku.llm.python import BaseLLM

#from langchain import hub
#from langchain.agents import AgentExecutor, create_openai_tools_agent
#from langchain.tools import tool
#from langchain_core.callbacks import Callbacks
#from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI
#import os, random
#from dataiku.llm.python import GenericLangChainAgentWrapper
#from dataiku.langchain.dku_llm import DKUChatLLM
from dataiku.langchain.dku_tracer import LangchainToDKUTracer
#from langchain_core.callbacks import Callbacks, BaseCallbackManager, BaseCallbackHandler



import dku_graphrag.query.query_monkey_patch_dataiku 

import asyncio
import pandas as pd

from graphrag.config.load_config import load_config
from graphrag.api.query import global_search

from pathlib import Path
import dataiku

async def search(query: str):
    # Load configuration the same way the CLI does
    folder_id = "ldwkTWoV"
    output_folder = dataiku.Folder(folder_id)
    output_folder_path = output_folder.get_path()

    root_dir = Path(output_folder_path)
    config = load_config(root_dir)
    nodes = pd.read_parquet("output/create_final_nodes.parquet")
    entities = pd.read_parquet("output/create_final_entities.parquet")
    communities = pd.read_parquet("output/create_final_communities.parquet")
    community_reports = pd.read_parquet("output/create_final_community_reports.parquet")

    # Now global_search will use the monkeypatched functions and thus DataikuChatLLM
    response, context = await global_search(
        config=config,
        nodes=nodes,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=None,
        dynamic_community_selection=False,
        response_type="multiple paragraphs",
        query=query
    )
    return response, context 
    #print("Response:", response)
    #print("Context:", context)






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
        pass

    def set_config(self, config, plugin_config):
        self.config = config
        
       

    def process(self, query, settings, trace):
        query = query["messages"][0]["content"]
        
        response, context = asyncio.run(search(query))
        resp_text = "%s \n\n\n ###### context: ###### \n%s" % (response, context)

        return {"text": resp_text}
           
