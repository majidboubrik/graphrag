import query_monkey_patch_dataiku 

import asyncio
import pandas as pd

from graphrag.config.load_config import load_config
from graphrag.api.query import global_search

from pathlib import Path


async def main():
    # Load configuration the same way the CLI does
    root_dir = Path("./")
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
        query="Summarize the studies please"
    )

    print("Response:", response)
    #print("Context:", context)

if __name__ == "__main__":
    asyncio.run(main())