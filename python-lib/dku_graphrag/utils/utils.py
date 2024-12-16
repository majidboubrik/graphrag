# -*- coding: utf-8 -*-
import dataiku
import os
import json
import pandas as pd

# Graphrag imports
from graphrag.index import GraphIndexBuilder
from graphrag.embeddings import HuggingFaceEmbedder
from graphrag.store import LocalGraphStore
from graphrag.retrieve import GraphRetriever

###############################################################################
# Configuration Parameters
###############################################################################
os.environ["DKU_CURRENT_PROJECT_KEY"] = "AGENTSANDBOX"
DATASET_NAME = "data"         # Name of the Dataiku dataset to index
TEXT_COLUMN = "content"                  # Column in the dataset that contains the text
METADATA_COLUMNS = ["title", "author"]   # Additional metadata columns
MANAGED_FOLDER_ID = "mZkLZUFc"  # ID of a Dataiku managed folder for storage

# Embedding model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Example Hugging Face model
CHUNK_SIZE = 500      # Optional: length of text chunks if you want to split long texts
CHUNK_OVERLAP = 100   # Optional: overlap between consecutive chunks

###############################################################################
# Load Data from Dataiku
###############################################################################
dataset = dataiku.Dataset(DATASET_NAME)
df = dataset.get_dataframe()


import sys
sys.exit()

# Fill NAs in text column to avoid errors
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")

# Prepare a list of documents (text) and a corresponding list of metadata dictionaries
documents = df[TEXT_COLUMN].tolist()
metadata_list = []
for _, row in df.iterrows():
    metadata = {col: row[col] for col in METADATA_COLUMNS if col in df.columns}
    metadata_list.append(metadata)

###############################################################################
# Setup Graph Storage in Managed Folder
###############################################################################
folder = dataiku.Folder(MANAGED_FOLDER_ID)
folder_path = folder.get_path()
GRAPH_STORAGE_PATH = os.path.join(folder_path, "graph_index")

# Ensure storage directory exists
if not os.path.exists(GRAPH_STORAGE_PATH):
    os.makedirs(GRAPH_STORAGE_PATH, exist_ok=True)

###############################################################################
# Create Embedder and Graph Store
###############################################################################
# The HuggingFaceEmbedder will use the specified model to embed text
embedder = HuggingFaceEmbedder(model_name=EMBEDDING_MODEL)

# LocalGraphStore stores index data locally; consider other stores for production
store = LocalGraphStore(base_dir=GRAPH_STORAGE_PATH)

###############################################################################
# Build the Graph-Based Index
###############################################################################
index_builder = GraphIndexBuilder(
    embedder=embedder,
    graph_store=store,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Build the index from documents and their metadata
# text_key is just a label used for referencing the text field in the index
index_builder.build_index(documents=documents, metadata=metadata_list, text_key=TEXT_COLUMN)

print("Index building completed! Graph-based index is stored at:", GRAPH_STORAGE_PATH)

###############################################################################
# Demonstrate Retrieval
###############################################################################
# We create a retriever using the same embedder and store
retriever = GraphRetriever(embedder=embedder, graph_store=store)

# Example query
user_query = "What does the author say about machine learning?"
top_k = 5  # How many results you want to retrieve

results = retriever.retrieve(query=user_query, top_k=top_k)

print("Retrieved Results:")
for idx, res in enumerate(results, start=1):
    print(f"Result {idx}:")
    print("Text:", res.text)
    print("Metadata:", json.dumps(res.metadata, indent=2))
    print("-" * 40)

# The retrieved chunks can then be passed as context to a large language model (LLM)
# in a Retrieval-Augmented Generation (RAG) pipeline to produce a final answer.