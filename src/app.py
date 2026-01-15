import gradio as gr
from rag_pipeline import RAGPipeline
import os

# Initialize RAG Pipeline
# Ensure we point to the correct paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")

try:
    print("Initializing Application...")
    rag = RAGPipeline(vector_db_path=VECTOR_DB_PATH)
except Exception as e:
    print(f"Error initializing RAG: {e}")
    rag = None

def chat_response(message, history):
    if rag is None:
        return "System not initialized correctly. Please check vector store."
        
    answer, docs, metas = rag.query(message)
    
    # Format sources for display
    sources_text = "\n\n**Sources:**\n"
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        sources_text += f"{i+1}. Product: {meta.get('product', 'N/A')}, Issue: {meta.get('issue', 'N/A')}\n"
        # Truncate doc preview
        preview = doc[:150] + "..." if len(doc) > 150 else doc
        sources_text += f"> \"{preview}\"\n"
        
    return f"{answer}{sources_text}"

# Create Gradio Interface
demo = gr.ChatInterface(
    fn=chat_response,
    title="💰 Intelligent Financial Complaint Assistant",
    description="Ask questions about consumer complaints. I will retrieve relevant narratives and summarize the findings.",
    examples=[
        "What are the common complaints about credit reporting?",
        "How do customers describe issues with mortgage payments?",
        "Are there complaints about unexpected fees?"
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=False)
