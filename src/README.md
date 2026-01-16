# Source Code Directory 📦

This directory contains production-ready Python modules that implement the core functionality demonstrated in the notebooks. Each module is designed for reusability, testability, and scalability.

---

## 📂 Module Overview

### Core Modules

#### 1. **`eda.py`** - Data Loading & Preprocessing
**Purpose**: Centralized data cleaning and preparation pipeline  
**Key Functions**:

- `load_raw_data(file_path)`: Loads raw complaints CSV with error handling
- `preprocess_data(df)`: Master preprocessing function that:
  - Maps raw product names to 4 target categories using `product_mapping` dictionary
  - Filters for target products only (Credit card, Personal loan, Savings account, Money transfers)
  - Removes rows with missing complaint narratives
  - Applies text cleaning via `clean_text()`
  - Performs sanity checks (duplicate complaint IDs, consistent Issue/Sub-issue mapping)
  
- `clean_text(text)`: Robust text cleaning that:
  - Converts to lowercase
  - Removes both uppercase and lowercase "XXXX" patterns (anonymization artifacts)
  - **Preserves** financial amounts ($200, 90%, etc.) and numbers for context
  - Removes special characters while keeping sentence structure
  - Handles NaN values gracefully

**Product Mapping Logic**:
```python
product_mapping = {
    "Credit card or prepaid card": "Credit card",
    "Checking or savings account": "Savings account",
    "Payday loan, title loan, or personal loan": "Personal loan",
    "Money transfer, virtual currency, or money service": "Money transfers",
    # ... handles 15+ raw product names → 4 categories
}
```

**Usage**:
```python
from src.eda import preprocess_data, load_raw_data

df = load_raw_data('../data/raw/complaints.csv')
df_clean = preprocess_data(df)
df_clean.to_csv('../data/processed/cleaned_complaints.csv', index=False)
```

**Dependencies**: pandas, re, nltk  
**Output**: Cleaned DataFrame with standardized product categories

---

#### 2. **`chunk_embed_index.py`** - Chunking, Embedding & Indexing (Prototype)
**Purpose**: Build Task 2 prototype vector store with stratified sampling  
**Key Functions**:

- `load_processed_data(file_path, stratified=True, sample_size=15000)`:
  - Loads cleaned complaints CSV
  - Performs stratified sampling proportional to product distribution
  - Returns sampled DataFrame with preserved class balance

- `get_sample_stats(df)`:
  - Computes and displays product distribution statistics
  - Useful for validation after sampling

- `process_and_index(df, chunk_size=1100, chunk_overlap=275, collection_name='complaints_prototype')`:
  - **Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter`
    - Default: 1100 chars (≈200 words), 275 char overlap (≈50 words)
    - Separators: `["\n\n", "\n", ". ", " ", ""]` for natural breaks
  - **Embedding**: Generates 384-dim vectors using `all-MiniLM-L6-v2`
  - **Indexing**: Stores in ChromaDB with metadata:
    - `product`, `issue`, `sub_issue`, `complaint_id`, `company`, `state`, `chunk_index`
  - **Batch Processing**: Processes in batches for memory efficiency
  - Returns collection object for validation

**Stratified Sampling Algorithm**:
```python
# Proportional sampling maintains original distribution:
# If Savings = 56% in full dataset, sample will also be 56% Savings
df_sample = df.groupby('Product', group_keys=False).apply(
    lambda x: x.sample(frac=sample_size/len(df), random_state=42)
)
```

**Usage**:
```python
from src.chunk_embed_index import load_processed_data, process_and_index

# Load with stratified sampling
df = load_processed_data('../data/processed/cleaned_complaints.csv', stratified=True, sample_size=15000)

# Create vector store
collection = process_and_index(
    df,
    chunk_size=1100,
    chunk_overlap=275,
    collection_name='complaints_prototype'
)

print(f"Indexed {collection.count()} chunks")
```

**Dependencies**: pandas, chromadb, sentence-transformers, langchain  
**Runtime**: ~8-12 minutes for 15K complaints  
**Output**: `complaints_prototype` collection in `../vector_store/`

---

#### 3. **`index_production.py`** - Production Vector Store Indexing
**Purpose**: Load pre-built embeddings from Parquet into ChromaDB (Task 3)  
**Key Functions**:

- `inspect_parquet_structure(parquet_path)`:
  - Analyzes pre-built vector store structure
  - **Detects embedding model** via:
    1. Checks metadata for `embedding_model` field (best case)
    2. Infers from embedding dimensions (fallback: 384 → all-MiniLM-L6-v2)
  - Displays sample data, metadata fields, document excerpts
  - Returns: `(model_name, embedding_dim)`

- `index_production_data()`:
  - Loads `complaint_embeddings.parquet` (1.3M+ rows)
  - Creates `complaints_production` collection in ChromaDB
  - **Batch Processing**: 5,000 rows per batch for memory efficiency
  - Stores detected embedding model in collection metadata for validation
  - Progress bar via `tqdm` for monitoring
  - Returns: `(detected_model, embedding_dim)`

**Key Design Decisions**:
1. **No Embedding Function at Creation**: Collection created without embedding function since embeddings are pre-computed
2. **Metadata Storage**: Stores `embedding_model` and `embedding_dimensions` for downstream validation
3. **Batch Size Optimization**: 5K rows balances memory vs. throughput (tested: 1K too slow, 10K OOM risk)

**Usage**:
```python
from src.index_production import inspect_parquet_structure, index_production_data, PARQUET_PATH

# Step 1: Inspect before loading
model_name, dim = inspect_parquet_structure(PARQUET_PATH)
print(f"Detected: {model_name} ({dim} dimensions)")

# Step 2: Load into ChromaDB
detected_model, detected_dim = index_production_data()
print(f"Indexed {detected_dim}-dimensional vectors using {detected_model}")
```

**Dependencies**: chromadb, pyarrow, tqdm  
**Runtime**: ~34 minutes for 1.3M complaints (276 batches @ 7.5s avg)  
**Output**: `complaints_production` collection with 1.3M+ document chunks

---

#### 4. **`rag_pipeline.py`** - RAG Query Engine
**Purpose**: End-to-end Retrieval-Augmented Generation pipeline  
**Class**: `RAGPipeline`

**Initialization**:
```python
class RAGPipeline:
    def __init__(self, vector_db_path="vector_store", collection_name="complaints_production", llm_model="google/flan-t5-base"):
```

- Connects to ChromaDB collection
- **Validates embedding model** match (stored metadata vs. query model)
- Loads embedding function: `all-MiniLM-L6-v2` (for query embedding)
- Loads LLM: Flan-T5-base for answer generation
- Device detection: GPU if available, else CPU

**Key Methods**:

1. `retrieve(query, n_results=5)`:
   - **Input**: User question string
   - **Process**:
     - Embeds query using `all-MiniLM-L6-v2` (via ChromaDB's `SentenceTransformerEmbeddingFunction`)
     - Performs cosine similarity search against stored embeddings
     - Retrieves top-k (default k=5) most relevant complaint excerpts
   - **Output**: `(docs, metas)` - lists of document texts and metadata dicts
   - **Note**: Uses `query_texts` parameter (ChromaDB embeds internally)

2. `generate_answer(query, context_docs)`:
   - **Input**: Query + list of retrieved document excerpts
   - **Process**:
     - Constructs Task 3 prompt template:
       ```
       You are a financial analyst assistant for CrediTrust. 
       Your task is to answer questions about customer complaints 
       based on the provided complaint excerpts.

       Complaint Excerpts:
       {context from top-5 docs}

       Question: {query}

       Answer:
       ```
     - Generates answer using Flan-T5-base (max_length=200 tokens)
   - **Output**: Generated answer string

3. `query(user_query, n_results=5)`:
   - **Input**: User question string
   - **Process**: Combines retrieve() + generate_answer()
   - **Output**: `(answer, docs, metas)` - answer + sources for attribution
   - **Error Handling**: Returns friendly message if no relevant docs found

**Embedding Model Validation**:
```python
# During init, validates stored model matches query model:
stored_model = collection.metadata.get('embedding_model')
query_model = "all-MiniLM-L6-v2"

if stored_model != query_model:
    print("⚠️  WARNING: Model mismatch! Results may be incorrect.")
```

**Usage**:
```python
from src.rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline(
    vector_db_path='../vector_store',
    collection_name='complaints_production'
)

# End-to-end query
answer, docs, metas = rag.query("Why are customers complaining about credit card fees?")

print(f"Answer: {answer}")
print(f"Based on {len(docs)} sources")
for i, (doc, meta) in enumerate(zip(docs, metas), 1):
    print(f"  {i}. Product: {meta['product']}, Issue: {meta['issue']}")
```

**Dependencies**: chromadb, transformers, torch, sentence-transformers  
**Runtime**: 
- Init: ~30-60 seconds (load models)
- Query: <1 second (after first-time ONNX setup)
- First query: ~9 minutes (ONNX model download/conversion, one-time cost)

---

#### 5. **`app.py`** - Gradio Web Interface
**Purpose**: Interactive chatbot for querying the RAG system  
**Key Components**:

- **Initialization**:
  - Loads `RAGPipeline` with `complaints_production` collection
  - Displays status messages (✅ success or ❌ error with troubleshooting tips)

- **`chat_response(message, history)` Function**:
  - Handles user queries through RAG pipeline
  - Formats response with:
    - Generated answer (from LLM)
    - 5 source excerpts with Product + Issue attribution
    - Error handling for failed queries or uninitialized system

- **Gradio Interface**:
  - `gr.ChatInterface` for conversational UI
  - Pre-loaded example queries (4 questions covering all products)
  - Clean formatting with markdown and emoji icons

**UI Features**:
- Title: "💰 CrediTrust Financial Complaint Assistant"
- Description: States 1.3M+ complaints indexed, explains RAG architecture
- Examples: Click-to-run queries for quick demo
- Source Attribution: Shows Product, Issue, and text excerpt for each retrieved complaint

**Usage**:
```bash
# From project root
python src/app.py

# Opens on http://127.0.0.1:7860
```

**Dependencies**: gradio, src.rag_pipeline  
**Runtime**: 
- Startup: ~30-60 seconds (load RAG pipeline)
- Per query: <2 seconds (after initial model warmup)

**Deployment Notes**:
- Set `share=False` for local-only access
- Change to `share=True` for public Gradio link (72-hour expiry)
- For production: Deploy behind reverse proxy (nginx) or use Gradio Enterprise

---

## 🔄 Module Dependencies & Data Flow

```
┌─────────────────┐
│  complaints.csv │  (raw data)
└────────┬────────┘
         │
         ▼
    ┌────────┐
    │ eda.py │  → cleaned_complaints.csv
    └────┬───┘
         │
         ├─────────────────────┬──────────────────────┐
         ▼                     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│chunk_embed_index.│  │index_production. │  │  (external source)  │
│       py         │  │       py         │  │complaint_embeddings.│
│                  │  │                  │  │     parquet         │
│ Task 2 Prototype │  │ Task 3 Production│  └──────────┬──────────┘
└────────┬─────────┘  └────────┬─────────┘             │
         │                     │                       │
         ▼                     ▼                       │
   ┌──────────────────────────────────────────────────┘
   │
   ▼
┌────────────────┐
│  vector_store/ │  (ChromaDB collections)
│  - complaints_ │
│    prototype   │
│  - complaints_ │
│    production  │
└────────┬───────┘
         │
         ▼
   ┌─────────────┐
   │rag_pipeline.│ ← User Query
   │     py      │
   └──────┬──────┘
          │
          ▼
    ┌─────────┐
    │ app.py  │  (Gradio UI)
    └─────────┘
          │
          ▼
   💬 User Interface
```

---

## 🧪 Testing

Each module includes docstrings and can be tested individually:

```bash
# Test EDA module
python -c "from src.eda import load_raw_data, preprocess_data; df = load_raw_data('data/raw/complaints.csv'); print(preprocess_data(df).shape)"

# Test chunking module
python -c "from src.chunk_embed_index import load_processed_data, get_sample_stats; df = load_processed_data('data/processed/cleaned_complaints.csv', sample_size=100); get_sample_stats(df)"

# Test RAG pipeline
python -c "from src.rag_pipeline import RAGPipeline; rag = RAGPipeline(); ans, _, _ = rag.query('test query'); print(ans)"
```

**Unit Tests**: See `../tests/` directory for comprehensive test suites.

---

## ⚙️ Configuration & Constants

### Default Paths (relative to project root):
- Raw data: `data/raw/complaints.csv`
- Processed data: `data/processed/cleaned_complaints.csv`
- Pre-built embeddings: `data/raw/complaint_embeddings.parquet`
- Vector store: `vector_store/`

### Model Configuration:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Dimensions: 384
  - Max sequence length: 256 tokens (~200 words)
  - Speed: ~1000 sequences/second on CPU
  
- **LLM**: `google/flan-t5-base`
  - Parameters: 250M
  - Max input: 512 tokens
  - Max output: 200 tokens
  - Speed: ~5 tokens/second on CPU, ~30 tokens/second on GPU

### ChromaDB Configuration:
- **Distance Metric**: Cosine similarity (`hnsw:space=cosine`)
- **Batch Size**: 
  - Prototype: Dynamic (based on available memory)
  - Production: 5,000 rows/batch
- **Persistence**: Disk-based storage (SQLite backend)

---

## 🚀 Performance Optimization Tips

1. **Speed up indexing**:
   ```python
   # Use larger batches if you have more RAM
   process_and_index(df, batch_size=10000)  # default: 5000
   ```

2. **Reduce first-query latency**:
   ```python
   # Pre-warm models during initialization
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("all-MiniLM-L6-v2")
   _ = model.encode("warmup")  # First call is slow
   ```

3. **Use GPU for faster LLM inference**:
   ```python
   # RAGPipeline automatically detects GPU
   # Ensure CUDA is installed: pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Cache frequent queries**:
   ```python
   # Add simple LRU cache to rag_pipeline.py
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def query_cached(self, user_query):
       return self.query(user_query)
   ```

---

## 📝 Adding New Modules

When extending the codebase:
1. Follow naming convention: `module_name.py` (lowercase, underscores)
2. Include docstrings for all functions (Google style)
3. Add type hints for function signatures
4. Update this README with module description
5. Add unit tests in `../tests/test_module_name.py`
6. Update `requirements.txt` if new dependencies are added

---

## 🐛 Known Issues & Workarounds

### Issue 1: Embedding Function Conflict
**Problem**: ChromaDB error "embedding function already exists" when collection created without function but queried with one.  
**Solution**: In `rag_pipeline.py`, we query using `query_texts` parameter which lets ChromaDB handle embedding internally. Ensure `index_production.py` creates collection without `embedding_function` parameter.

### Issue 2: Slow First Query
**Problem**: First query after app restart takes 9+ minutes.  
**Cause**: ChromaDB downloads and converts model to ONNX format (one-time cost).  
**Solution**: Pre-warm models during init or use persistent model cache location.

### Issue 3: Memory Usage During Indexing
**Problem**: Indexing 1.3M complaints uses 8-12GB RAM.  
**Solution**: Reduce batch size in `index_production.py` (trade speed for memory):
```python
# Change from 5000 to 2500
batch_size = 2500
```

---

## 📧 Support

For module-specific questions:
- **Data Issues**: Check `eda.py` and `../notebooks/eda.ipynb`
- **Indexing Issues**: Check `chunk_embed_index.py` or `index_production.py`
- **Query Issues**: Check `rag_pipeline.py` and `../notebooks/rag_demo.ipynb`
- **UI Issues**: Check `app.py` and Gradio documentation

See **Project Report** (`../project_report.md`) for comprehensive technical documentation.