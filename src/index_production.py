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

def inspect_parquet_structure(parquet_path):
    """
    Inspect the pre-built vector store to detect embedding model and structure.
    Returns: (model_name, embedding_dim)
    """
    print("=" * 60)
    print("INSPECTING PRE-BUILT VECTOR STORE")
    print("=" * 60)
    
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Read first batch as sample
    first_batch = parquet_file.read_row_group(0)
    sample_df = first_batch.to_pandas().head(5)
    
    print(f"\n📊 Total rows in Parquet: {parquet_file.metadata.num_rows:,}")
    print(f"📋 Columns: {sample_df.columns.tolist()}")
    print(f"\nData Types:")
    print(sample_df.dtypes)
    
    # --- Check Embedding Dimensions ---
    sample_embedding = sample_df['embedding'].iloc[0]
    embedding_dim = len(sample_embedding)
    print(f"\n🔢 Embedding Dimensions: {embedding_dim}")
    print(f"   Sample values: {sample_embedding[:5]}...")
    
    # --- Detect Embedding Model ---
    print(f"\n🤖 Detecting Embedding Model:")
    
    # Method 1: Check metadata
    sample_meta = sample_df['metadata'].iloc[0]
    detected_model = None
    
    if 'embedding_model' in sample_meta:
        detected_model = sample_meta['embedding_model']
        print(f"   ✓ Found in metadata: '{detected_model}'")
    else:
        print(f"   ⚠️  'embedding_model' not found in metadata")
        print(f"   Inferring from dimensions...")
        
        # Method 2: Infer from dimensions
        dimension_map = {
            384: "all-MiniLM-L6-v2",
            768: "all-mpnet-base-v2 or BERT-base",
            1024: "OpenAI ada-002 or similar",
            1536: "OpenAI text-embedding-3-small"
        }
        
        inferred_model = dimension_map.get(embedding_dim, "Unknown")
        print(f"   Best guess: {inferred_model}")
        
        if embedding_dim == 384:
            detected_model = "all-MiniLM-L6-v2"
            print(f"   ✓ Assuming 'all-MiniLM-L6-v2' (most common 384-dim model)")
        else:
            print(f"   ⚠️  WARNING: Uncommon dimension size!")
            print(f"   You MUST verify the model before querying!")
    
    # --- Check Metadata Fields ---
    print(f"\n📝 Sample Metadata Fields:")
    for key, value in sample_meta.items():
        print(f"   {key}: {value}")
    
    # --- Sample Documents ---
    print(f"\n📄 Sample Document (first 150 chars):")
    print(f"   {sample_df['document'].iloc[0][:150]}...")
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    
    return detected_model, embedding_dim

def index_production_data():
    """
    Load pre-built embeddings into ChromaDB for production use.
    """
    print(f"Opening Parquet file: {PARQUET_PATH}")
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"File not found: {PARQUET_PATH}")
    
    # --- Step 1: Inspect the Parquet file ---
    detected_model, embedding_dim = inspect_parquet_structure(PARQUET_PATH)
    
    if not detected_model:
        print("\n⚠️  ERROR: Could not detect embedding model!")
        print("Cannot proceed without knowing which model to use for queries.")
        return
    
    print(f"\n✓ Will use '{detected_model}' for queries\n")

    # --- Step 2: Initialize Chroma ---
    print(f"Initializing Vector Store at {VECTOR_DB_PATH}...")
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    # Delete if exists to start fresh
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Collection did not exist or could not be deleted: {e}")

    # --- Step 3: Create collection with metadata ---
    collection = client.create_collection(
        name=COLLECTION_NAME,
        #should an embedding model be specified here when creating the collection?
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": detected_model,
            "embedding_dimensions": embedding_dim
        }
    )
    print(f"Created collection: {COLLECTION_NAME}")
    print(f"  Embedding model: {detected_model}")
    print(f"  Dimensions: {embedding_dim}")

    # --- Step 4: Load data in batches ---
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    batch_size = 5000
    
    print(f"\nLoading {parquet_file.metadata.num_rows:,} rows in batches of {batch_size}...")
    
    count = 0
    for batch in tqdm(parquet_file.iter_batches(batch_size=batch_size), desc="Indexing"):
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
        
    print(f"\n✓ Indexing Complete!")
    print(f"  Total documents: {count:,}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Embedding model: {detected_model}")
    
    return detected_model, embedding_dim

if __name__ == "__main__":
    index_production_data()
