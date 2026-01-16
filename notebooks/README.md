# Notebooks Directory 📓

This directory contains Jupyter notebooks documenting the complete workflow from data exploration to RAG pipeline evaluation. Each notebook represents a critical phase in the project lifecycle.

---

## 📚 Notebook Overview

### 1. **`eda.ipynb`** - Exploratory Data Analysis
**Purpose**: Initial data exploration and quality assessment  
**Key Sections**:
- **Data Loading**: Reads `complaints.csv` from `../data/raw/`
- **Product Distribution Analysis**: Visualizes complaint counts across 4 target products (Credit card, Personal loan, Savings account, Money transfers)
- **Text Statistics**: 
  - Narrative word count distribution (median: 114 words, 75th percentile: 209 words)
  - Missing value detection and handling
- **Data Quality Issues**:
  - Product name mapping (raw → standardized categories)
  - Duplicate complaint ID detection
  - Text cleaning demonstrations (preserving financial amounts like $200, 90%)
- **Outputs**: `cleaned_complaints.csv` in `../data/processed/`

**Runtime**: ~2-3 minutes  
**Key Visualizations**: Product distribution bar chart, word count histogram, missing value heatmap

---

### 2. **`chunk_embed_index.ipynb`** - Chunking, Embedding & Indexing (Prototype)
**Purpose**: Build Task 2 prototype vector store with stratified sampling  
**Key Sections**:
- **Chunking Strategy**:
  - Determined optimal chunk size: 200 words (1100 chars) with 50-word overlap (275 chars)
  - Based on EDA finding that 70-75% of complaints fit within 200 words
  - Uses `RecursiveCharacterTextSplitter` from LangChain
- **Stratified Sampling**:
  - 15,000 complaints sampled proportionally across 4 products
  - Preserves original distribution: Savings (56%), Credit card (32%), Personal loan (11%), Money transfers (1%)
- **Embedding Generation**:
  - Model: `all-MiniLM-L6-v2` (384 dimensions)
  - Batch processing for memory efficiency
- **ChromaDB Indexing**:
  - Collection: `complaints_prototype`
  - Cosine similarity metric
  - Metadata preserved: product, issue, sub_issue, complaint_id, company
- **Validation** (13 steps):
  - Sample distribution verification
  - Chunking correctness demo
  - Retrieval quality tests with sample queries
  - Embedding dimension checks

**Runtime**: ~8-12 minutes (including embedding generation)  
**Key Outputs**: 
- `complaints_prototype` collection in `../vector_store/`
- Validation plots: sample distribution, chunk size histogram

---

### 3. **`rag_demo.ipynb`** - RAG Pipeline with Production Vector Store
**Purpose**: Task 3 implementation - complete RAG workflow with pre-built embeddings  
**Key Sections**:

#### **Step 1: Inspect Pre-Built Vector Store**
- Detects embedding model from parquet metadata (fallback: dimension-based inference)
- Validates structure: `id`, `document`, `embedding`, `metadata` columns
- Confirms 384-dimensional embeddings → `all-MiniLM-L6-v2`

#### **Step 2: Load Production Embeddings**
- Reads `complaint_embeddings.parquet` (1,375,327 rows)
- Creates `complaints_production` collection in ChromaDB
- Batch processing: 5,000 rows/batch
- **Runtime**: ~34 minutes (276 batches @ 7.54s/batch avg)
- Stores embedding model metadata for validation

#### **Step 3: Initialize RAG Pipeline**
- Connects to `complaints_production` collection
- Validates embedding model match (query model vs. stored embeddings)
- Loads Flan-T5-base LLM for answer generation
- Sets default k=5 retrieval (Task 3 requirement)

#### **Step 4: Test Retrieval (k=5)**
- Query embedding: User question → 384-dim vector
- Cosine similarity search against 1.3M+ stored embeddings
- Returns top-5 most relevant complaint excerpts with metadata

#### **Step 5: Test Answer Generation**
- Prompt template: "You are a financial analyst assistant for CrediTrust..."
- Context: Top-5 retrieved complaint excerpts
- LLM: Flan-T5-base (text2text-generation)
- Max length: 200 tokens

#### **Step 6: End-to-End RAG Queries**
- Demonstrates full workflow: Retrieve → Augment → Generate
- Test queries across all 4 products
- Shows answer + source attribution

#### **Evaluation: RAG Quality Assessment**
- 4 test questions evaluated with Quality Scores (1-5):
  - Q1: Credit card fees - Score 4/5
  - Q2: Savings account issues - Score 2/5 (fragmented retrieval)
  - Q3: Money transfer problems - Score 3/5 (repetitive formatting)
  - Q4: Personal loan complaints - Score 3/5 (mixed retrieval set)
- Identifies weaknesses: under-synthesis, noisy fragments, needs product filtering
- Recommendations: Add filters, clean context, enforce tighter answer format

**Runtime**: 
- First-time setup: ~40 minutes (indexing + model downloads)
- Subsequent queries: < 1 second (models cached)
- **Challenge**: First query after restart takes 9+ minutes (ONNX model download/conversion)

**Key Outputs**:
- `complaints_production` collection (1.3M+ documents)
- Quality evaluation table with scores and analysis

---

### 4. **`evaluation.ipynb`** - Comparative Evaluation (Prototype vs Production)
**Purpose**: Compare Task 2 (prototype) vs Task 3 (production) system performance  
**Key Sections**:
- **Metrics**: Answer quality, relevance, diversity, hallucination detection
- **Test Set**: 5 standard financial queries across all products
- **Expected Outcome**: Production system (1.3M complaints) should provide more accurate and diverse context than prototype (15K complaints)
- **Analysis**: Side-by-side comparison of generated answers and retrieved sources

**Runtime**: ~5-8 minutes  
**Key Visualizations**: Quality score comparison charts, retrieval diversity plots

---

## 🔧 Setup & Usage

### Prerequisites
```bash
# Install dependencies
pip install -r ../requirements.txt

# Key packages needed:
# - pandas, numpy (data manipulation)
# - matplotlib, seaborn (visualization)
# - nltk (text processing)
# - sentence-transformers (embeddings)
# - chromadb (vector database)
# - langchain (text splitting)
# - transformers, torch (LLM)
# - pyarrow (parquet handling)
```

### Running Notebooks

1. **Start Jupyter**:
```bash
cd notebooks
jupyter notebook
```

2. **Recommended Execution Order**:
   - `eda.ipynb` → Understand data, generate cleaned dataset
   - `chunk_embed_index.ipynb` → Build prototype vector store (Task 2)
   - `rag_demo.ipynb` → Run production RAG pipeline (Task 3)
   - `evaluation.ipynb` → Compare systems (optional)

3. **Important Notes**:
   - `rag_demo.ipynb` requires `complaint_embeddings.parquet` in `../data/raw/`
   - First run of `rag_demo.ipynb` takes ~40 minutes due to indexing
   - Ensure at least 8GB RAM for embedding operations
   - GPU optional but speeds up LLM inference

---

## 📊 Key Findings from Notebooks

### Data Insights (EDA)
- **Total Complaints**: 248,000+ raw complaints
- **Product Distribution**: Heavily skewed toward Savings accounts (56%)
- **Text Length**: Most complaints are 50-250 words (suitable for 200-word chunks)
- **Missing Data**: ~5% of complaints have null narratives (filtered out)

### Chunking Strategy (Task 2)
- **Optimal Size**: 200 words covers 70-75% of complaints without truncation
- **Overlap**: 50 words (25% overlap) ensures context continuity across chunk boundaries
- **Stratified Sampling**: Preserves product distribution in 15K sample for fair evaluation

### RAG Performance (Task 3)
- **Retrieval Quality**: Generally strong for credit cards, money transfers, personal loans
- **Weak Points**: Savings account queries retrieve fragmented/noisy results (likely due to "Checking or savings account" combined product label)
- **Answer Quality**: Varies (2-4 out of 5) - some answers too generic or mirror single excerpts instead of synthesizing

### Technical Challenges Documented
1. **Indexing Time**: 34 minutes for 1.3M embeddings (production scale)
2. **Model Download**: First query takes 9+ minutes for ONNX conversion
3. **Embedding Function Conflict**: ChromaDB mismatch between collection creation (no function) and retrieval (with function) - resolved by manual embedding
4. **Memory Requirements**: Embedding 15K complaints requires ~4GB RAM peak usage

---

## 🎯 Next Steps

Based on notebook findings:
1. **Improve Retrieval**:
   - Add product-type filters to queries
   - Implement hybrid search (keyword + semantic)
   - Fine-tune chunk size per product type

2. **Enhance Generation**:
   - Upgrade to larger LLM (Flan-T5-large or Llama-2-7B)
   - Implement answer post-processing to remove formatting artifacts
   - Add chain-of-thought prompting for better synthesis

3. **Scale to Production**:
   - Optimize batch sizes for faster indexing
   - Implement incremental indexing for new complaints
   - Add caching layer for frequent queries

---

## 📝 Contributing

When adding new notebooks:
1. Follow naming convention: `task_description.ipynb`
2. Include markdown cells explaining each section
3. Add cell outputs for key visualizations (commit with outputs)
4. Update this README with notebook description and runtime estimates
5. Document any new dependencies in `../requirements.txt`

---

## 📧 Contact

For questions about the notebooks or methodology, refer to:
- **Project Report**: `../project_report.md` (detailed technical documentation)
- **Source Code**: `../src/` (production implementations of notebook experiments)
