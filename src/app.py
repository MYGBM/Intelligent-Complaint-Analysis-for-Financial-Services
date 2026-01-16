import gradio as gr
from rag_pipeline import RAGPipeline
import os

# Initialize RAG Pipeline
# Ensure we point to the correct paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")

try:
    print("Initializing Application...")
    # Explicitly specify collection name for clarity
    rag = RAGPipeline(
        vector_db_path=VECTOR_DB_PATH,
        collection_name="complaints_production"
    )
    print("✅ RAG Pipeline initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing RAG: {e}")
    print("Make sure you've run index_production_data() first!")
    rag = None

def chat_response(message, history):
    """
    Handle user queries through the RAG pipeline.
    Returns answer + sources formatted for display.
    """
    if rag is None:
        return "⚠️ System not initialized. Please ensure the vector store exists and is properly indexed."
    
    try:
        # Query RAG pipeline (returns answer, docs list, metas list)
        answer, docs, metas = rag.query(message)
        
        # Format sources for display
        sources_text = "\n\n**📚 Sources (5 retrieved complaints):**\n"
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            product = meta.get('product', 'N/A')
            issue = meta.get('issue', 'N/A')
            sources_text += f"\n**{i}.** Product: *{product}* | Issue: *{issue}*\n"
            # Truncate doc preview
            preview = doc[:150] + "..." if len(doc) > 150 else doc
            sources_text += f"> \"{preview}\"\n"
            
        return f"**Answer:**\n{answer}{sources_text}"
    
    except Exception as e:
        return f"❌ Error processing query: {str(e)}\n\nPlease try rephrasing your question."

# Create Gradio Interface
# Use only parameters that are widely supported across Gradio versions
demo = gr.ChatInterface(
    fn=chat_response,
    title="💰 CrediTrust Financial Complaint Assistant",
    description=(
        "Ask questions about consumer complaints (1.3M+ complaints indexed). "
        "The system retrieves 5 relevant complaint excerpts and generates answers using RAG (Retrieval-Augmented Generation)."
    ),
    examples=[
        "Why are customers complaining about credit card fees?",
        "What issues do people have with savings accounts?",
        "Tell me about problems with money transfers",
        "What are common complaints about personal loans?"
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)
