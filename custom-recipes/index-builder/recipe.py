import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config




import asyncio
from pathlib import Path
from graphrag.logger.factory import LoggerType
from dku_graphrag.index.index import DataikuGraphragIndexBuilder  

# Retrieve input dataset
input_dataset_name = get_input_names_for_role('input_dataset')[0]
input_dataset = dataiku.Dataset(input_dataset_name)

# Retrieve output managed folder
output_folder_name = get_output_names_for_role('output_folder')[0]
output_folder = dataiku.Folder(output_folder_name)
output_folder_path = output_folder.get_path()


# Retrieve recipe parameters
config = get_recipe_config()
text_column = config.get('text_column')
attribute_columns = config.get('attribute_columns', [])
verbose_mode = config.get('verbose_mode', False)

# Combine text and attribute columns
selected_columns = [text_column] + attribute_columns

# Read data from input dataset
df = input_dataset.get_dataframe(columns=selected_columns)

# Implement your indexing logic here
if verbose_mode:
    print(f"Selected columns for indexing: {selected_columns}")
    print(f"DataFrame shape: {df.shape}")

# Write the selected columns to a CSV file in the managed folder
output_file_path = 'selected_columns.csv'
with output_folder.get_writer(output_file_path) as writer:
    writer.write(df.to_csv(index=False).encode('utf-8'))  # Encode the string to bytes


builder = DataikuGraphragIndexBuilder(logger_type=LoggerType.RICH)

asyncio.run(builder.run_build_index_pipeline( 
        root_dir=Path(output_folder_path),
        verbose=True,
        resume=None,
        memprofile=False,
        cache=True,
        config_filepath=None,
        skip_validation=False,
        output_dir=None))
        