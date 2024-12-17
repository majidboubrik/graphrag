import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config, get_recipe_resource

import asyncio
import os
import shutil
from pathlib import Path
from graphrag.logger.factory import LoggerType
from dku_graphrag.index.index import DataikuGraphragIndexBuilder  

import logging
from graphrag.config.load_config import load_config



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

if verbose_mode:
    logging.basicConfig(level=logging.DEBUG)  # Set minimum level to DEBUG


# Combine text and attribute columns (if needed)
selected_columns = [text_column] + attribute_columns

# Read data from input dataset
df = input_dataset.get_dataframe(columns=selected_columns)

if verbose_mode:
    print(f"Selected columns for indexing: {selected_columns}")
    print(f"DataFrame shape: {df.shape}")

# --- Step 1: Copy settings.yaml into the output folder ---
resource_folder_path = get_recipe_resource()
settings_source = os.path.join(resource_folder_path, "settings.yaml")
settings_target = os.path.join(output_folder_path, "settings.yaml")

# Remove old settings.yaml if it exists
if os.path.exists(settings_target):
    os.remove(settings_target)

with output_folder.get_writer("settings.yaml") as writer:
    with open(settings_source, "rb") as f:
        writer.write(f.read())


# --- Step 2: Copy prompts folder to the output folder ---
prompts_source = os.path.join(resource_folder_path, "prompts")
prompts_target = os.path.join(output_folder_path, "prompts")

# Remove old prompts folder if it exists
if os.path.exists(prompts_target):
    shutil.rmtree(prompts_target)

# Copy prompts directory
shutil.copytree(prompts_source, prompts_target)

# --- Step 3: Create input directory in output folder and copy the dataset there ---
input_dir = os.path.join(output_folder_path, "input")

# Remove old input directory if it exists
if os.path.exists(input_dir):
    shutil.rmtree(input_dir)

os.makedirs(input_dir)



# Write the entire dataset to a CSV file named after the dataset in the input directory
input_file_path = os.path.join(input_dir, f"{input_dataset_name}.csv")
df_full = input_dataset.get_dataframe()  # get full dataset if needed, or reuse df if partial columns suffice
df_full.to_csv(input_file_path, index=False, encoding='utf-8')
# --- Load and modify config ---
root_dir = Path(output_folder_path)

config = load_config(root_dir, None)


# Override relative paths with absolute paths
# Example: if "db_uri": "output/lancedb", then make it absolute by joining with root_dir
print(config.embeddings.vector_store)
print(dir(config.embeddings.vector_store))

if "db_uri" in config.embeddings.vector_store:
    # Update the 'db_uri' value
    config.embeddings.vector_store["db_uri"] = root_dir / config.embeddings.vector_store["db_uri"]
else:
    print("Error: 'db_uri' key not found in vector_store")


# --- Run the builder ---
builder = DataikuGraphragIndexBuilder(logger_type=LoggerType.RICH)
if True:
    asyncio.run(builder.run_build_index_pipeline(
        config=config,
        verbose=True,
        resume=None,
        memprofile=False
    ))