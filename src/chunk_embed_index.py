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

    def load_processed_data(self, nrows=None):
        """
        Loads the preprocessed data.
        """
        print(f"Loading data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")
            
        self.df = pd.read_csv(self.data_path, nrows=nrows)
        # Drop rows where cleaned_narrative might be NaN (just in case)
        self.df = self.df.dropna(subset=['cleaned_narrative'])
        print(f"Loaded {len(self.df)} rows.")
        return self.df

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

    def process_and_index(self, chunk_size=500, chunk_overlap=50, batch_size=100):
        """
        Chunks the text, and inserts into ChromaDB in batches.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
            
        print("Initializing Text Splitter...")
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
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Complaints"):
            complaint_id = str(row.get('Complaint ID', f"row_{idx}"))
            text = row.get('cleaned_narrative', "")
            product = row.get('Product', "Unknown")
            issue = row.get('Issue', "Unknown")
            
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Chunking
            chunks = text_splitter.split_text(text)
            
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
            
        print(f"Indexing Complete. Total Chunks Indexed: {count}")

if __name__ == "__main__":
    # Test run
    # Use absolute path to ensure it works regardless of CWD
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "cleaned_complaints.csv")
    
    indexer = ChunkEmbedIndex(data_path)
    # Load a sample to test
    try:
        indexer.load_processed_data(nrows=10000) 
        indexer.initialize_vector_store()
        indexer.process_and_index()
    except FileNotFoundError as e:
        print(e)
        print("Please run processed_data extraction first.")
