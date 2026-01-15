import pandas as pd
import os

file_path = r'data/raw/complaint_embeddings.parquet'

try:
    print(f"Reading {file_path}...")
    # Read a few rows if possible, or just metadata
    # PyArrow allows reading metadata without loading the whole file
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(file_path)
    print("Metadata:")
    print(parquet_file.metadata)
    print("\nSchema:")
    print(parquet_file.schema)
    
    # Read first batch
    first_batch = next(parquet_file.iter_batches(batch_size=5))
    df = first_batch.to_pandas()
    print("\nFirst 5 rows:")
    print(df)
    print("\nColumns:", df.columns.tolist())
    
    # Check embedding column type
    if 'embedding' in df.columns:
        print("\nEmbedding sample:")
        print(df['embedding'].iloc[0])
        print("Embedding Case type:", type(df['embedding'].iloc[0]))
        if len(df['embedding'].iloc[0]) > 0:
             print("Embedding Dimension:", len(df['embedding'].iloc[0]))

except Exception as e:
    print(f"Error: {e}")
