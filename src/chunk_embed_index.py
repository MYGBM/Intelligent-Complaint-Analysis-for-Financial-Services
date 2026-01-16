import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm

class ChunkEmbedIndex:
    def __init__(self, data_path, vector_db_path="vector_store", collection_name="complaints_prototype"):
        """
        Initializes the RAG indexing pipeline.
        
        Args:
            data_path (str): Path to the preprocessed CSV file.
            vector_db_path (str): Path to store the ChromaDB.
            collection_name (str): Name of the collection in ChromaDB.
        """
        self.data_path = data_path
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.df = None
        self.client = None
        self.collection = None
        
        # Use SentenceTransformers for embedding
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def load_processed_data(self, nrows=None, stratified=True):
        """
        Loads the preprocessed data with optional stratified sampling.
        
        Args:
            nrows (int): Total number of rows to load. If None, loads all data.
            stratified (bool): If True, samples proportionally from each Product category.
        """
        print(f"Loading data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")
        
        if nrows is None or not stratified:
            # Load all data or sequential sample
            self.df = pd.read_csv(self.data_path, nrows=nrows)
        else:
            # Stratified sampling: load proportionally from each product
            print(f"Using stratified sampling to load {nrows} rows...")
            full_df = pd.read_csv(self.data_path)
            
            # Get product distribution
            product_counts = full_df['Product'].value_counts()
            total_available = len(full_df)
            
            print(f"\nTotal available rows: {total_available}")
            print(f"Product distribution in full dataset:")
            print(product_counts)
            
            # Calculate samples per product proportionally
            samples_per_product = {}
            for product, count in product_counts.items():
                proportion = count / total_available
                sample_size = int(nrows * proportion)
                # Ensure at least 1 sample for each product
                samples_per_product[product] = max(1, sample_size)
            
            print(f"\nSampling strategy:")
            for product, sample_size in samples_per_product.items():
                print(f"  {product}: {sample_size} rows")
            
            # Sample from each product
            sampled_dfs = []
            for product, sample_size in samples_per_product.items():
                product_df = full_df[full_df['Product'] == product]
                # Sample min of (requested, available) to avoid errors
                actual_sample = min(sample_size, len(product_df))
                sampled = product_df.sample(n=actual_sample, random_state=42)
                sampled_dfs.append(sampled)
            
            self.df = pd.concat(sampled_dfs, ignore_index=True)
            # Shuffle to mix products
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            
        # Drop rows where cleaned_narrative might be NaN (just in case)
        self.df = self.df.dropna(subset=['cleaned_narrative'])
        print(f"\n✓ Loaded {len(self.df)} rows.")
        return self.df
    
    def get_sample_stats(self):
        """
        Returns statistics about the loaded sample.
        """
        if self.df is None:
            return "No data loaded."
        
        stats = {
            "total_rows": len(self.df),
            "product_distribution": self.df['Product'].value_counts().to_dict(),
            "avg_narrative_length_words": self.df['cleaned_narrative'].apply(lambda x: len(str(x).split())).mean(),
            "avg_narrative_length_chars": self.df['cleaned_narrative'].apply(lambda x: len(str(x))).mean()
        }
        return stats

    def initialize_vector_store(self):
        """
        Initializes the persistent ChromaDB client and collection.
        Deletes existing collection if it exists to start fresh (for prototype).
        """
        print(f"Initializing Vector Store at {self.vector_db_path}...")
        # Ensure directory exists
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        
        # Reset/Get collection
        # try:
        #     self.client.delete_collection(name=self.collection_name)
        #     print(f"Deleted existing collection: {self.collection_name}")
        # except ValueError:
        #     pass # Collection didn't exist
            
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Cosine similarity
        )
        print(f"Created collection: {self.collection_name}")

    def process_and_index(self, chunk_size=1100, chunk_overlap=275, batch_size=100):
        """
        Chunks the text, and inserts into ChromaDB in batches.
        
        Args:
            chunk_size (int): Characters per chunk. Default 1100 (~200 words).
            chunk_overlap (int): Character overlap between chunks. Default 275 (~50 words).
            batch_size (int): Number of chunks to insert per batch.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
            
        print("Initializing Text Splitter...")
        print(f"Chunk size: {chunk_size} characters (~{chunk_size//5.5:.0f} words)")
        print(f"Chunk overlap: {chunk_overlap} characters (~{chunk_overlap//5.5:.0f} words)")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        print("Starting Chunking and Indexing...")
        
        # We will process row by row, chunk, and prepare for batch insert
        # To avoid memory issues with huge DFs, we insert in batches
        
        ids = []
        documents = []
        metadatas = []
        
        count = 0
        chunks_per_product = {}  # Track chunks by product
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Complaints"):
            complaint_id = str(row.get('Complaint ID', f"row_{idx}"))
            text = row.get('cleaned_narrative', "")
            product = row.get('Product', "Unknown")
            issue = row.get('Issue', "Unknown")
            
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Chunking
            chunks = text_splitter.split_text(text)
            
            # Track chunks per product
            if product not in chunks_per_product:
                chunks_per_product[product] = 0
            chunks_per_product[product] += len(chunks)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{complaint_id}_chunk_{i}"
                
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "complaint_id": complaint_id,
                    "product": product,
                    "issue": issue,
                    "chunk_index": i
                })
                
                # Batch Insert
                if len(ids) >= batch_size:
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                    count += len(ids)
                    ids = []
                    documents = []
                    metadatas = []
        
        # Insert remaining
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            count += len(ids)
            
        print(f"\n✓ Indexing Complete. Total Chunks Indexed: {count}")
        print(f"\nChunks per product:")
        for product, chunk_count in sorted(chunks_per_product.items()):
            print(f"  {product}: {chunk_count} chunks")
        
        return {"total_chunks": count, "chunks_per_product": chunks_per_product}

if __name__ == "__main__":
    # Test run
    # Use absolute path to ensure it works regardless of CWD
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "cleaned_complaints.csv")
    
    indexer = ChunkEmbedIndex(data_path)
    # Load a sample to test
    try:
        indexer.load_processed_data(nrows=15000, stratified=True) 
        print("\nSample statistics:")
        print(indexer.get_sample_stats())
        indexer.initialize_vector_store()
        indexer.process_and_index(chunk_size=1100, chunk_overlap=275)
    except FileNotFoundError as e:
        print(e)
        print("Please run processed_data extraction first.")
