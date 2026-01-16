import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class RAGPipeline:
    #it seems as if index
    def __init__(self, vector_db_path="vector_store", collection_name="complaints_production", llm_model="google/flan-t5-base"):
        """
        Initializes the RAG Retrieval and Generation pipeline.
        """
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.llm_model_name = llm_model
        
        print("Initializing Vector Store Client...")
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                # embedding_function=self.embedding_fn
            )
            print(f"Connected to collection: {self.collection_name}")
            
            # --- Validate Embedding Model ---
            stored_metadata = self.collection.metadata
            stored_model = stored_metadata.get('embedding_model', 'Unknown')
            stored_dim = stored_metadata.get('embedding_dimensions', 'Unknown')
            
            print(f"📦 Collection Metadata:")
            print(f"   Stored model: {stored_model}")
            print(f"   Dimensions: {stored_dim}")
            
            # Check if models match
            if stored_model != 'Unknown':
                query_model = "all-MiniLM-L6-v2"
                
                if stored_model != query_model:
                    print(f"\n⚠️  WARNING: Embedding model mismatch!")
                    print(f"   Stored: {stored_model}")
                    print(f"   Query: {query_model}")
                    print(f"   Results may be incorrect!\n")
                else:
                    print(f"✓ Embedding model validated: {query_model}\n")
            else:
                print(f"⚠️  Model metadata not found - assuming all-MiniLM-L6-v2\n")
                
        except Exception as e:
            raise ValueError(f"Collection {self.collection_name} not found. Please run indexing first. Error: {e}")

        print(f"Loading LLM: {self.llm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)
        
        # Use CPU or GPU
        device = 0 if torch.cuda.is_available() else -1
        
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            device=device
        )
        print("LLM Loaded successfully.")

    def retrieve(self, query, n_results=5):
        """
        Retrieves relevant documents from ChromaDB based on the query.
        Task 3 requirement: k=5 for top-5 retrieval.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Extract documents and metadata
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        return docs, metas

    def generate_answer(self, query, context_docs):
        """
        Generates an answer using the LLM given the context.
        Uses Task 3 prompt template for CrediTrust financial analyst assistant.
        """
        # Construct context from complaint excerpts
        context_text = "\n\n".join([
            f"Complaint {i+1}: {doc}" 
            for i, doc in enumerate(context_docs)
        ])
        
        # Task 3 Prompt Template
        prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints based on the provided complaint excerpts.

Complaint Excerpts:
{context_text}

Question: {query}

Answer:"""
        
        # Generate answer using Flan-T5
        response = self.generator(prompt, max_length=200, do_sample=False)
        return response[0]['generated_text']

    def query(self, user_query, n_results=5):
        """
        End-to-end RAG: Retrieve + Generate.
        Task 3 default: k=5 for retrieval.
        """
        # 1. Retrieve
        docs, metas = self.retrieve(user_query, n_results)
        
        # 2. Augment & Generate
        if not docs:
            return "No relevant complaints found to answer your query.", []
            
        answer = self.generate_answer(user_query, docs)
        
        return answer, docs, metas
