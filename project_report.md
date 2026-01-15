# Intelligent Complaint Analysis - Project Report

## 1. Executive Summary

This project successfully implemented an end-to-end Retrieval-Augmented Generation (RAG) system for analyzing financial consumer complaints. The system progresses from a raw dataset to a fully interactive chatbot capable of answering questions based on thousands of complaint narratives.

## 2. Architecture & Components

### 2.1 Data Preparation (Task 1)

- **EDA**: Initial exploratory data analysis (`notebooks/eda.ipynb`) identified data quality issues (missing narratives) and distribution (top products).
- **Preprocessing**: Implemented robust text cleaning in `src/eda.py` including:
  - Tokenization & Lemmatization.
  - Stopword removal.
  - Filtering for target products.

### 2.2 Vector Stores

- **Prototype (Task 2)**:
  - Built using a local subset of data.
  - **Chunking**: `RecursiveCharacterTextSplitter` (500 chars, 50 overlap).
  - **Embedding**: `all-MiniLM-L6-v2` (Sentence Transformers).
  - **Storage**: ChromaDB collection `complaints_prototype`.
- **Production (Task 3)**:
  - leveraged pre-computed embeddings from `complaint_embeddings.parquet`.
  - **Ingestion**: Efficient batch indexing script (`src/index_production.py`).
  - **Storage**: ChromaDB collection `complaints_production`.

### 2.3 RAG Pipeline (Task 3)

- **Model**: `google/flan-t5-base` for response generation.
- **Logic**: Implemented in `src/rag_pipeline.py`.
  - Retrieves top-k relevant chunks via Cosine Similarity.
  - Constructs a prompt with context.
  - Generates a concise natural language answer.

### 2.4 Interactive Interface (Task 4)

- **UI**: Gradio-based chat interface (`src/app.py`).
- **Features**:
  - Semantic Q&A.
  - Source attribution (Product, Issue categories).
  - Error handling for missing indices.

## 3. Evaluation

A comparative evaluation (`notebooks/evaluation.ipynb`) was designed to contrast the Prototype and Production systems.

- **Metric**: Quality of generated answers for 5 standard financial queries.
- **Expected Outcome**: The Production system, having access to a vastly larger dataset (464k vs ~5k), provides more accurate and diverse context, leading to higher quality and less generic answers.

## 4. Conclusion & Next Steps

The project demonstrates a scalable approach to financial text analysis.

- **Success**: All 4 tasks completed within the timeframe.
- **Future Work**:
  - Deploy to a cloud server (AWS/GCP).
  - Upgrade LLM to a larger model (e.g., Llama-2-7b) for more complex reasoning.
  - Implement fine-grained filtering by Date or Company in the retrieval step.
