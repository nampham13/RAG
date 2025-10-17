# RAG System – AI Coding Agent Instructions

## 🎯 System Architecture
Modular RAG pipeline with strict OOP design: `PDF → PDFLoader → PDFDocument → HybridChunker → ChunkSet → OllamaEmbedder → FAISS`

**Key Modules:**
- **`loaders/`** - Raw PDF extraction (text/tables) with factory patterns
- **`chunkers/`** - Multi-strategy text segmentation (semantic/rule-based/fixed-size)
- **`embedders/`** - Ollama-only embeddings (Gemma:768-dim, BGE-M3:1024-dim)
- **`llm/`** - LLM integration with configuration loading and API handlers
- **`pipeline/`** - RAG pipeline package with composition architecture
  - **`rag_pipeline.py`** - Main orchestrator using composition
  - **`vector_store.py`** - FAISS index management with IndexFlatIP
  - **`summary_generator.py`** - Document/batch summaries
  - **`retriever.py`** - Cosine similarity search using normalized vectors
- **`data/`** - FAISS indexes (.faiss), metadata maps (.pkl), summaries (.json)

---

## 🛠️ Essential Developer Workflows

### Core Pipeline Execution
```powershell
# Process all PDFs in data/pdf/ to FAISS indexes
python run_pipeline.py

# Alternative: Direct module execution
python -m pipeline.rag_pipeline

# Test chunking functionality (available demo)
python chunkers/chunk_pdf_demo.py

# Test retrieval system
python test_retrieval.py
```

### Testing & Validation
```powershell
# Run all tests with coverage (configured in pyproject.toml)
python -m pytest -v --cov=loaders

# Run tests by module (note: embedders typo in path)
python -m pytest test/loaders/ -v        # PDF loading tests
python -m pytest test/chunkers/ -v       # Chunking tests
python -m pytest test/embeders/ -v       # Embedding tests (note: "embeders" not "embedders")
python -m pytest test/pipeline/ -v       # Pipeline tests

# Individual chunker tests
python -m pytest chunkers/test_fixed_size_chunker.py -v

# Manual pipeline integration test
python test/pipeline/test_pipeline_manual.py

# Real PDF integration test
python test/pipeline/test_real_pdf.py
```

### Chunk Caching Behavior
The pipeline maintains a cache of processed chunks (`data/cache/processed_chunks.json`):
- **First run**: Generates embeddings for all chunks
- **Subsequent runs**: Skips chunks with identical content hash (no re-embedding)
- **To force re-processing**: Delete `data/cache/processed_chunks.json` before running

### LLM Integration Testing
```powershell
# Test LLM API connections
python -c "from llm.LLM_API import LLMAPI; api = LLMAPI(); print('LLM ready')"
```

### Ollama Setup (Required)
```bash
# Check available models
ollama list

# Required models
ollama pull embeddinggemma:latest
ollama pull bge-m3:latest
```

---

## 🏗️ Critical Code Patterns

### 1. Factory Pattern (Universal)
**Every major class uses factories for common configs:**
```python
# PDF Loading
loader = PDFLoader.create_default()        # Text + tables, normalization enabled
loader = PDFLoader.create_text_only()      # Text only

# Embedding (Ollama-only)
factory = EmbedderFactory()
gemma = factory.create_gemma()             # 768-dim semantic search
bge3 = factory.create_bge_m3()             # 1024-dim multilingual

# Fast model switching
switcher = OllamaModelSwitcher()
switcher.switch_to_gemma()
embedder = switcher.current_embedder
```

### 2. Constructor Injection (No Global Config)
**All configuration via constructor - no YAML dependencies:**
```python
# ✅ Current pattern
loader = PDFLoader(extract_tables=True, min_text_length=15)

# ❌ Deprecated (YAML auto-loading)
# loader = PDFLoader()  # Would load preprocessing.yaml
```

### 3. Configuration Loading Pattern
**Centralized config loading with caching:**
```python
from llm.config_loader import get_config

# Load app configuration (cached singleton)
config = get_config()
model_settings = config.get('llm', {}).get('models', {})
```

### 4. Data Model Normalization
**Raw extraction separated from cleaning:**
```python
# Load raw data
pdf_doc = loader.load("doc.pdf")

# Apply normalization when needed
pdf_doc = pdf_doc.normalize()  # Deduplication, text cleaning, etc.
```

### 5. Modular Chunking Strategies
**HybridChunker orchestrates multiple chunking approaches:**
```python
# Configure chunker with multiple strategies
chunker = HybridChunker(
    max_tokens=200,
    overlap_tokens=20,
    mode=ChunkerMode.AUTO  # Auto-selects best strategy per document section
)

# Available strategies: semantic, rule-based, fixed-size, structural-first
chunk_set = chunker.chunk(pdf_document)
```

### 6. Composition over Inheritance
**RAGPipeline uses composition with specialized classes:**
```python
from pipeline import RAGPipeline, VectorStore, SummaryGenerator, Retriever

class RAGPipeline:
    def __init__(self, ...):
        self.loader = PDFLoader.create_default()
        self.chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
        self.embedder = embedder_factory.create_gemma()
        self.vector_store = VectorStore(self.vectors_dir)      # Separate class
        self.summary_generator = SummaryGenerator(...)         # Separate class
        self.retriever = Retriever(self.embedder)              # Separate class
```

---

## 🔧 Integration Points & Dependencies

### External Services
- **Ollama Server**: `http://localhost:11434` (required for all embeddings)
- **FAISS**: Vector storage and cosine similarity search (IndexFlatIP with normalized vectors)

### PDF Processing Libraries
```python
# Multiple engines for robustness
fitz (PyMuPDF)      # Primary text extraction
pdfplumber          # Table extraction fallback
camelot-py[cv]      # Advanced table parsing
```

### LLM Integration
- **Local LLM**: Ollama-based models via `llm/LLM_LOCAL.py`
- **API LLM**: External API integration via `llm/LLM_API.py`
- **Config Loading**: YAML-based configuration in `config/app.yaml`

### Data Output Structure
Each processed PDF generates three files:
```
data/
├── vectors/
│   ├── Document_vectors_20251015_143022.faiss      # Binary FAISS index
│   └── Document_metadata_map_20251015_143022.pkl   # Chunk metadata (pages, provenance)
└── metadata/
    └── Document_summary_20251015_143022.json       # Human-readable document info
```

### Search & Retrieval
```python
# Direct pipeline usage for search
from pipeline import RAGPipeline
pipeline = RAGPipeline()

# Search against specific FAISS index
results = pipeline.search_similar(
    faiss_file=Path("data/vectors/Doc_vectors_20251015.faiss"),
    metadata_map_file=Path("data/vectors/Doc_metadata_map_20251015.pkl"),
    query_text="your search query",
    top_k=5
)

# Results include cosine similarity scores, text content, and page numbers
for result in results:
    print(f"Score: {result['cosine_similarity']:.4f}")
    print(f"Page: {result['page_number']}")
```

---

## ⚠️ Project-Specific Conventions

### Vietnamese Documentation
- Code comments and docstrings in Vietnamese
- README files in Vietnamese
- Maintain consistency for team collaboration

### Strict Single Responsibility
- **Loaders**: Raw extraction only (no chunking/normalization)
- **Chunkers**: Document → chunks only (no embedding)
- **Embedders**: Chunks → vectors only (Ollama-only)
- **LLM**: Model integration and API handling only
- **VectorStore**: FAISS index management only
- **SummaryGenerator**: Summary creation and persistence only
- **Retriever**: Search operations only
- **RAGPipeline**: Orchestration and composition only

### Table Handling
```python
# Tables include schema in chunk metadata
if chunk.metadata.get("block_type") == "table":
    table_payload = chunk.metadata.get("table_payload")
    headers = table_payload.header
    rows = table_payload.rows
```

### Configuration Management
```python
# Use config_loader for centralized configuration
from llm.config_loader import get_config

config = get_config()  # Cached singleton pattern
llm_config = config.get('llm', {})
```

### Testing Structure
```python
# One test class per module
class TestPDFLoader:
    def test_initialization_default_params(self):
        """Test constructor with defaults"""

    def test_factory_create_default(self):
        """Test factory method patterns"""

# Test file organization mirrors source structure
test/
├── loaders/test_pdf_loader.py
├── chunkers/test_chunkers.py
├── embeders/test_embedders.py
└── pipeline/test_rag_pipeline.py
```

### Testing Structure
```python
# One test class per module
class TestPDFLoader:
    def test_initialization_default_params(self):
        """Test constructor with defaults"""

    def test_factory_create_default(self):
        """Test factory method patterns"""

# Test file organization mirrors source structure
test/
├── loaders/test_pdf_loader.py
├── chunkers/test_chunkers.py
├── embeders/test_embedders.py
└── pipeline/test_rag_pipeline.py
```

---

## 🚨 Common Pitfalls

- **Ollama Connection**: Always test connection before embedding operations
- **Dimension Mismatch**: Gemma (768) ≠ BGE-M3 (1024) - choose based on use case
- **Memory Usage**: FAISS indexes can be large; monitor disk space in `data/vectors/`
- **PDF Encoding**: Use UTF-8 handling; some PDFs have encoding issues
- **Config Loading**: Use `get_config()` from `llm.config_loader` instead of direct YAML loading
- **Model Switching**: Use `OllamaModelSwitcher` for runtime model changes, not recreating embedders

Use these patterns when extending the system or adding new modules.