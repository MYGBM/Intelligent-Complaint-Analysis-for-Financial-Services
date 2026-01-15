# Intelligent Complaint Analysis for Financial Services 🏦

![Project Status](https://img.shields.io/badge/Status-In%20Development-blue)
![Python](https://img.shields.io/badge/Python-3.9+-yellow)
![RAG](https://img.shields.io/badge/AI-RAG%20Pipeline-green)

A Retrieval-Augmented Generation (RAG) system designed to help financial institutions analyze and query customer complaints efficiently. This project processes thousands of complaints, builds a searchable knowledge base, and provides granular answers via an interactive AI chatbot.

---

## 🚀 Features

- **Semantic Search**: Retrieval of relevant complaints using advanced embedding models (`all-MiniLM-L6-v2`).
- **RAG Pipeline**: Generates human-like answers grounded in real complaint data using `FLAN-T5`.
- **Interactive Dashboard**: clean web interface built with **Gradio** for real-time querying.
- **Data Integrity**: Robust cleaning pipelines ensuring high-quality inputs.
- **Traceability**: Every answer cites its source complaint chunks for full transparency.

## 📂 Project Structure

```bash
Intelligent-Complaint-Analysis-for-Financial-Services/
├── .github/workflows/   # CI/CD pipelines
├── data/                # Data storage
│   ├── raw/             # Original datasets (complaints.csv, embeddings.parquet)
│   └── processed/       # Cleaned and prepared data
├── vector_store/        # Persisted ChromaDB databases (Prototype & Production)
├── notebooks/           # Jupyter notebooks for EDA and experimentation
├── src/                 # Source code modules
│   ├── __init__.py
│   ├── eda.py                 # Data loading & preprocessing
│   ├── chunker.py             # Text splitting strategies
│   ├── embedder.py            # Embedding generation
│   ├── vector_store_prototype.py  # ChromaDB management
│   ├── retriever.py           # Semantic search logic
│   ├── generator.py           # LLM response generation
│   └── rag_pipeline.py        # End-to-end orchestration
├── tests/               # Unit tests
├── app.py               # Gradio chat interface application
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

## 🛠️ Technology Stack

| Component      | Technology                               | Reasoning                                                                          |
| -------------- | ---------------------------------------- | ---------------------------------------------------------------------------------- |
| **Language**   | Python 3.9+                              | Standard for AI/ML engineering.                                                    |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, cpu-friendly, and effective for semantic similarity.                         |
| **Vector DB**  | ChromaDB                                 | Lightweight, open-source, and local storage (no external server required).         |
| **LLM**        | `google/flan-t5-base`                    | Instruction-tuned model capable of running locally without heavy GPU requirements. |
| **Chunking**   | `RecursiveCharacterTextSplitter`         | Preserves semantic context by respecting sentence boundaries.                      |
| **UI**         | Gradio                                   | Rapid development of interactive chat interfaces.                                  |

## 🏁 Getting Started

### Prerequisites

- Python 3.9 or higher
- Pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/MYGBM/Intelligent-Complaint-Analysis-for-Financial-Services.git
   cd Intelligent-Complaint-Analysis-for-Financial-Services
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup**
   Ensure your datasets (`complaints.csv` and `complaint_embeddings.parquet`) are placed in `data/raw/`.

### Usage

**1. Run Exploratory Data Analysis (EDA)**
Generate initial insights and data cleaning maps.

```bash
jupyter notebook notebooks/eda.ipynb
```

**2. Build Prototype Vector Store (Optional)**
Embed a sample of 10k complaints to verify the pipeline.

```bash
python src/chunk_embed_index.py
```

**3. Launch the RAG App**
Start the full interactive chatbot using pre-built production embeddings.

```bash
python app.py
```

## 📊 Evaluation

The system is evaluated using a qualitative framework:

- **Relevance**: Does the retrieved chunk answer the question?
- **Accuracy**: Is the generated answer factually correct based on the text?
- **Clarity**: Is the answer easy to understand?

See `src/evaluate.py` for the test suite and results table.

## 🤝 Contribution

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
