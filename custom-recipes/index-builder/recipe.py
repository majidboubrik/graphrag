import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config, get_recipe_resource

import asyncio
import os
import shutil
from pathlib import Path
from dku_graphrag.index.dataiku_graph_index_builder import DataikuGraphragIndexBuilder  

import logging
from graphrag.config.load_config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("graphrag recipe")

# Retrieve input dataset
input_dataset_name = get_input_names_for_role('input_dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)

# Retrieve output managed folder
output_folder_name = get_output_names_for_role('output_folder')[0]
output_folder = dataiku.Folder(output_folder_name)
output_folder_path = output_folder.get_path()

# Remove old output folder if it exists
if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)

# Retrieve recipe parameters
config = get_recipe_config()
text_column = config.get('text_column')
attribute_columns = config.get('attribute_columns', [])
verbose_mode = config.get('verbose_mode', False)
chat_completion_llm_id = config.get("chat_completion_llm_id")
embedding_llm_id = config.get("embedding_llm_id")


if verbose_mode:
    logging.basicConfig(level=logging.DEBUG)  # Set minimum level to DEBUG

resource_folder_path = get_recipe_resource()
settings_source = os.path.join(resource_folder_path, "settings.yaml")
settings_target = os.path.join(output_folder_path, "settings.yaml")

if os.path.exists(settings_target):
    logger.debug("Deleting old settings.yaml file")
    os.remove(settings_target)

with output_folder.get_writer("settings.yaml") as writer:
    with open(settings_source, "rb") as f:
        writer.write(f.read())
    logger.debug(f"Copied new settings.yaml file from resources to {settings_target}")


prompts_source = os.path.join(resource_folder_path, "prompts")
prompts_target = os.path.join(output_folder_path, "prompts")

if os.path.exists(prompts_target):
    logger.debug("Deleting old prompts folder")
    shutil.rmtree(prompts_target)

logger.debug("Copying new prompts from plugin")
shutil.copytree(prompts_source, prompts_target)

input_dir = os.path.join(output_folder_path, "input")

logger.debug("Deleting old input directory")
if os.path.exists(input_dir):
    shutil.rmtree(input_dir)

os.makedirs(input_dir)
logger.debug("Created input directory")




# Write the entire dataset to a CSV file named after the dataset in the input directory
selected_columns = [text_column] + attribute_columns
df = input_dataset.get_dataframe(columns=selected_columns)
input_file_path = os.path.join(input_dir, f"{input_dataset_name}.csv")
df.to_csv(input_file_path, index=False, encoding='utf-8')

logger.debug(f"Input dataset copied to indexinf folder: {input_file_path}")


logger.debug(f"Selected columns for indexing: {selected_columns}")
logger.debug(f"DataFrame shape: {df.shape}")

root_dir = Path(output_folder_path)
graph_rag_config = load_config(root_dir, None)


# Override relative paths with absolute paths for vector db
if "db_uri" in graph_rag_config.embeddings.vector_store:
    graph_rag_config.embeddings.vector_store["db_uri"] = root_dir / graph_rag_config.embeddings.vector_store["db_uri"]
    logger.info(f"Setting vector db absolite path: {graph_rag_config.embeddings.vector_store['db_uri'] }")
else:
    logger.error("Error: 'db_uri' key not found in vector_store")



# --- Run the builder ---
builder = DataikuGraphragIndexBuilder(chat_completion_llm_id, embedding_llm_id)
asyncio.run(builder.run_build_index_pipeline(
        config=graph_rag_config,
        verbose=True,
        resume=None,
        memprofile=False
))