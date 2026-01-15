import chromadb
import pandas as pd
import pyarrow.parquet as pq
import os
from tqdm import tqdm
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'complaint_embeddings.parquet')
VECTOR_DB_PATH = os.path.join(BASE_DIR, 'vector_store')
COLLECTION_NAME = 'complaints_production'

def index_production_data():
    print(f"Opening Parquet file: {PARQUET_PATH}")
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"File not found: {PARQUET_PATH}")

    # Initialize Chroma
    print(f"Initializing Vector Store at {VECTOR_DB_PATH}...")
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    # Delete if exists to start fresh
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Collection did not exist or could not be deleted: {e}")

    # Create collection
    # We do NOT provide an embedding function because we already have embeddings
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created collection: {COLLECTION_NAME}")

    # Iterative processing
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    batch_size = 5000
    
    print(f"Total rows (approx): {parquet_file.metadata.num_rows}")
    
    count = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        
        ids = df['id'].astype(str).tolist()
        documents = df['document'].astype(str).tolist()
        embeddings = df['embedding'].tolist()
        
        # Flatten metadata
        # The schema showed 'metadata' as a struct. In pandas it likely comes as a dict.
        # We need to ensure it's a dict of supported types (str, int, float, bool)
        
        metadatas = []
        for meta in df['metadata']:
            # meta should be a dictionary-like object
            # Filter/Convert to ensure compatibility
            clean_meta = {}
            if meta:
                # Common fields of interest
                fields = ['product', 'issue', 'sub_issue', 'company', 'state', 'complaint_id']
                for key in fields:
                    if key in meta and meta[key] is not None:
                        clean_meta[key] = str(meta[key]) # Convert to string to be safe
                
                # Add chunk info if present
                if 'chunk_index' in meta:
                    clean_meta['chunk_index'] = int(meta['chunk_index'])
                    
            metadatas.append(clean_meta)
            
        # Add to Chroma
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        count += len(ids)
        print(f"Indexed {count} documents...", end='\r')
        
    print(f"\nIndexing Complete. Total Indexed: {count}")

if __name__ == "__main__":
    index_production_data()
