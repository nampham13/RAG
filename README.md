# RAG Pipeline with FAISS Vector Search

A modular RAG (Retrieval-Augmented Generation) system that processes PDF documents into FAISS vector indexes for fast similarity search.

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Ollama server running locally (`http://localhost:11434`)
- Required Ollama models: `embeddinggemma:latest` and `bge-m3:latest`

### Ollama Setup

```bash
# Install Ollama (if not already installed)
# Download from: https://ollama.ai/download

# Start Ollama server
ollama serve

# Pull required models
ollama pull embeddinggemma:latest
ollama pull bge-m3:latest

# Verify models are available
ollama list
```

### Installation

```bash
pip install -r requirements.txt
```

### Process PDFs to FAISS Indexes

```powershell
python -m pipeline.rag_pipeline
```

**Output**: Creates FAISS vector indexes and metadata for all PDFs in `data/pdf/`

### Test Search Functionality

```powershell
python demo_faiss_search.py
```

## 📁 Project Structure

├── pipeline/                    # RAG Pipeline package
│   ├── __init**.py             # Package exports
│   ├── rag_pipeline.py         # Main RAG pipeline orchestrator
│   ├── vector_store.py         # FAISS index management
│   ├── summary_generator.py    # Document/batch summaries
│   └── retriever.py            # Vector similarity search
├── demo_faiss_search.py         # Search demo script
├── data/
│   ├── pdf/                     # Source PDF files
│   ├── vectors/                 # FAISS indexes (.faiss, .pkl)
│   └── metadata/                # Document summaries (.json)
├── loaders/                     # PDF loading and parsing
├── chunkers/                    # Text chunking strategies
├── embedders/                   # Ollama embedding providers
└── requirements.txt             # Python dependencies

## 🔍 Usage in Code

```python
from pipeline import RAGPipeline
from pathlib import Path

# Initialize pipeline
pipeline = RAGPipeline()

# Search similar content
results = pipeline.search_similar(
    faiss_file=Path("data/vectors/Doc_vectors.faiss"),
    metadata_map_file=Path("data/vectors/Doc_metadata_map.pkl"),
    query_text="your search query",
    top_k=5
)

# Display results
for result in results:
    print(f"Score: {result['similarity_score']:.4f}")
    print(f"Text: {result['text'][:100]}...")
    print(f"Page: {result['page_number']}")
```

## 🏗️ Architecture

### Core Pipeline

1. **PDF Loading**: Extract text, tables, and metadata from PDFs
2. **Text Chunking**: Split documents into semantically meaningful chunks
3. **Embedding Generation**: Convert chunks to vectors using Ollama models
4. **FAISS Indexing**: Store vectors in efficient binary format for fast search

### Key Components

- **Loaders**: PDF parsing with table extraction and text normalization
- **Chunkers**: Hybrid chunking (semantic + rule-based + fixed-size fallback)
- **Embedders**: Ollama-based embedding with Gemma (768-dim) and BGE-M3 (1024-dim) models
- **FAISS**: Vector similarity search with cosine similarity metric (IndexFlatIP)

## 📊 Output Files

Each processed PDF generates 3 files:

- **`.faiss`**: Binary vector index with normalized embeddings (cosine similarity ready)
- **`.pkl`**: Metadata mapping (page numbers, chunk provenance, similarity scores)
- **`_summary.json`**: Document information (lightweight text format)

## 🎯 Ollama Embedders

### Quick Usage

```python
from embedders.embedder_factory import EmbedderFactory

factory = EmbedderFactory()
gemma_embedder = factory.create_gemma()    # 768 dimensions
bge3_embedder = factory.create_bge_m3()    # 1024 dimensions

embedding = gemma_embedder.embed("Your text here")
print(f"Embedding dimension: {len(embedding)}")
```

### Model Comparison

| Model | Dimensions | Use Case |
|-------|------------|----------|
| EmbeddingGemma | 768 | High-dimensional semantic search |
| BGE-M3 | 1024 | Multilingual efficient retrieval |

### Advanced Usage

```python
from embedders.providers.ollama import OllamaModelSwitcher, OllamaModelType

# Dynamic model switching
switcher = OllamaModelSwitcher()
switcher.switch_to_gemma()
embedder = switcher.current_embedder

# Test connection
if embedder.test_connection():
    embedding = embedder.embed("Sample text")
    print(f"✅ Connected - Dimension: {len(embedding)}")
```

### Model Specifications

**EmbeddingGemma:**

- **Model ID**: `embeddinggemma:latest`
- **Dimensions**: 768
- **Context**: Up to 8192 tokens
- **Best for**: General semantic search, high-quality embeddings

**BGE-M3:**

- **Model ID**: `bge-m3:latest`
- **Dimensions**: 1024
- **Context**: Up to 8192 tokens
- **Best for**: Multilingual content, efficient retrieval

## 🔧 Troubleshooting

### Embedding Issues

**Ollama Connection Failed:**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama server
ollama serve
```

**Model Not Found:**

```bash
# Pull models manually
ollama pull embeddinggemma:latest
ollama pull bge-m3:latest

# List available models
ollama list
```

**Test Embedding Functionality:**

```python
from embedders.embedder_factory import EmbedderFactory

factory = EmbedderFactory()
embedder = factory.create_gemma()

# Test connection
if embedder.test_connection():
    embedding = embedder.embed("Hello world")
    print(f"✅ Success! Dimension: {len(embedding)}")
else:
    print("❌ Connection failed")
```

## ✅ Production Ready

- **Compact Storage**: FAISS binary format (70-90% smaller than JSON)
- **Fast Search**: Optimized vector similarity with cosine similarity metric
- **Modular Design**: Clean separation of concerns with factory patterns
- **Error Handling**: Robust processing with connection testing and fallbacks
- **Normalized Embeddings**: Vectors normalized for optimal cosine similarity performance
"# RAG_chatbot" 
