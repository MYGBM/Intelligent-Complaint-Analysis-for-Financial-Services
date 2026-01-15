import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class RAGPipeline:
    def __init__(self, vector_db_path="vector_store", collection_name="complaints_prototype", llm_model="google/flan-t5-base"):
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
                embedding_function=self.embedding_fn
            )
            print(f"Connected to collection: {self.collection_name}")
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

    def retrieve(self, query, n_results=3):
        """
        Retrieves relevant documents from ChromaDB based on the query.
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
        """
        # Construct prompt
        context_text = "\n\n".join(context_docs)
        
        prompt = f"""
        You are a helpful assistant analyzing financial consumer complaints.
        Use the following context to answer the question. If the answer is not in the context, say "I don't know based on the available information."
        
        Context:
        {context_text}
        
        Question: {query}
        
        Answer:
        """
        
        # Determine max tokens based on model
        # Flan-T5 has 512 limit often, need to be careful with prompt length
        # For simplicity, we just pass the prompt. The pipeline handles truncation slightly or we can manually truncate context if needed.
        
        response = self.generator(prompt, max_length=200, do_sample=False)
        return response[0]['generated_text']

    def query(self, user_query, n_results=3):
        """
        End-to-end RAG: Retrieve + Generate.
        """
        # 1. Retrieve
        docs, metas = self.retrieve(user_query, n_results)
        
        # 2. Augment & Generate
        if not docs:
            return "No relevant complaints found to answer your query.", []
            
        answer = self.generate_answer(user_query, docs)
        
        return answer, docs, metas
