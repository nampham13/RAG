# Reranker Integration Guide

## Overview

The RAG pipeline now supports optional reranking to improve retrieval accuracy. Reranking re-scores the initial vector search results using different algorithms, potentially improving the relevance of returned documents.

## Available Rerankers

### 1. BM25 Reranker
- **Type**: `"bm25"`
- **Dependencies**: `rank_bm25` (already in requirements.txt)
- **Best for**: Keyword-based queries, exact matching
- **Speed**: Fast (CPU-based)

### 2. BGE Reranker
- **Type**: `"bge"`
- **Dependencies**: `transformers`, `torch` (already in requirements.txt)
- **Best for**: Semantic similarity, complex queries
- **Speed**: Slower (model-based, CPU only)

## Quick Start

### Basic Usage (No Reranking)

```python
from pipeline.rag_pipeline import RAGPipeline

# Initialize without reranking
pipeline = RAGPipeline(output_dir="data")

# Search
results = pipeline.search_similar(
    faiss_file=faiss_file,
    metadata_map_file=metadata_file,
    query_text="your query",
    top_k=5
)
```

### Using BM25 Reranker

```python
from pipeline.rag_pipeline import RAGPipeline

# Initialize with BM25 reranker
pipeline = RAGPipeline(
    output_dir="data",
    reranker_type="bm25"
)

# Search with reranking
results = pipeline.search_similar(
    faiss_file=faiss_file,
    metadata_map_file=metadata_file,
    query_text="your query",
    top_k=5,
    use_reranking=True,
    reranking_top_k=15  # Retrieve 15 candidates, rerank to top 5
)
```

### Using BGE Reranker

```python
from pipeline.rag_pipeline import RAGPipeline

# Initialize with BGE reranker
pipeline = RAGPipeline(
    output_dir="data",
    reranker_type="bge"
)

# Search with reranking
results = pipeline.search_similar(
    faiss_file=faiss_file,
    metadata_map_file=metadata_file,
    query_text="your query",
    top_k=5,
    use_reranking=True,
    reranking_top_k=20
)
```

## Advanced Configuration

### Custom BM25 Parameters

```python
from pipeline.rag_pipeline import RAGPipeline

# Custom BM25 settings
bm25_kwargs = {
    'k1': 1.5,      # Term frequency saturation (default: 1.5)
    'b': 0.75,      # Length normalization (default: 0.75)
    'epsilon': 0.25 # IDF floor value (default: 0.25)
}

pipeline = RAGPipeline(
    output_dir="data",
    reranker_type="bm25",
    reranker_kwargs=bm25_kwargs
)
```

### Custom BGE Model

```python
from pipeline.rag_pipeline import RAGPipeline

# Custom BGE model
bge_kwargs = {
    'model_name': 'BAAI/bge-large-en-v1.5',  # Larger model
    'batch_size': 16,
    'embed_cache': True
}

pipeline = RAGPipeline(
    output_dir="data",
    reranker_type="bge",
    reranker_kwargs=bge_kwargs
)
```

## Using with fetch_retrieval

```python
from pipeline.pipeline_qa import fetch_retrieval

# Without reranking
result = fetch_retrieval(
    query_text="your query",
    top_k=5
)

# With reranking
result = fetch_retrieval(
    query_text="your query",
    top_k=5,
    use_reranking=True,
    reranking_top_k=15
)

# Access results
context = result['context']
sources = result['sources']

for source in sources:
    print(f"Score: {source['similarity_score']:.4f}")
    print(f"Text: {source['text'][:100]}...")
```

## How Reranking Works

1. **Initial Retrieval**: Vector search retrieves `reranking_top_k` candidates (default: `top_k * 3`)
2. **Reranking**: Reranker re-scores all candidates based on query relevance
3. **Final Selection**: Top `top_k` results after reranking are returned

### Example Flow

```
Query: "machine learning algorithms"

Step 1: Vector Search
  → Retrieve 15 candidates (reranking_top_k=15)
  → Based on embedding similarity

Step 2: Reranking
  → BM25/BGE re-scores all 15 candidates
  → Re-orders based on new scores

Step 3: Return Top 5
  → Return best 5 results after reranking
```

## Choosing Reranking Parameters

### reranking_top_k

The number of candidates to retrieve before reranking:

- **Too small**: May miss relevant documents
- **Too large**: Slower, diminishing returns
- **Recommended**: 2-5x the desired `top_k`

```python
# For top_k=5
reranking_top_k=10  # 2x - fast, may miss some candidates
reranking_top_k=15  # 3x - balanced (recommended)
reranking_top_k=25  # 5x - thorough, slower
```

## Performance Considerations

### BM25 Reranker
- **Speed**: Very fast (< 10ms for 20 candidates)
- **Memory**: Low (~1MB)
- **Best for**: Production systems, real-time applications

### BGE Reranker
- **Speed**: Moderate (100-500ms for 20 candidates on CPU)
- **Memory**: Moderate (~500MB for model)
- **Best for**: Offline processing, high-accuracy requirements

## Testing

Run the integration test:

```bash
python test_reranker_integration.py
```

Run the demo:

```bash
python demo_reranker.py
```

## Result Format

Results include reranking scores when reranking is enabled:

```python
{
    'chunk_id': 'chunk_123',
    'text': 'Document text...',
    'page_number': 5,
    'similarity_score': 0.85,      # Rerank score (when reranking enabled)
    'cosine_similarity': 0.78,     # Original vector similarity
    'rerank_score': 0.85,          # Explicit rerank score
    'file_name': 'document.pdf',
    # ... other metadata
}
```

## When to Use Reranking

### Use Reranking When:
- Accuracy is more important than speed
- You have complex semantic queries
- Initial vector search returns marginal results
- You want to combine lexical and semantic search

### Skip Reranking When:
- Speed is critical (< 100ms response time)
- Vector search already returns good results
- Simple keyword queries
- Limited computational resources

## Troubleshooting

### BGE Reranker Not Working

```python
# Check if transformers is installed
pip install transformers torch

# Verify initialization
from rerankers.reranker_factory import RerankerFactory
reranker = RerankerFactory.create_bge()
# Should not raise an error
```

### BM25 Reranker Not Working

```python
# Check if rank_bm25 is installed
pip install rank_bm25

# Verify initialization
from rerankers.reranker_factory import RerankerFactory
reranker = RerankerFactory.create_bm25()
# Should return a BM25Reranker instance
```

### No Improvement with Reranking

- Increase `reranking_top_k` to retrieve more candidates
- Try a different reranker (BM25 vs BGE)
- Verify query quality and document content
- Check if vector search is already optimal

## Examples

See `demo_reranker.py` for comprehensive examples of:
- Basic retrieval without reranking
- BM25 reranking
- BGE reranking
- Advanced configuration
- Performance comparison

## API Reference

### RAGPipeline.__init__

```python
def __init__(
    self,
    output_dir: str = "data",
    pdf_dir: Optional[str | Path] = None,
    model_type: OllamaModelType = OllamaModelType.GEMMA,
    reranker_type: Optional[str] = None,
    reranker_kwargs: Optional[Dict[str, Any]] = None
)
```

### RAGPipeline.search_similar

```python
def search_similar(
    self,
    faiss_file: Path,
    metadata_map_file: Path,
    query_text: str,
    top_k: int = 5,
    use_softmax: bool = True,
    temperature: Optional[float] = None,
    use_reranking: bool = False,
    reranking_top_k: Optional[int] = None
) -> List[Dict[str, Any]]
```

### fetch_retrieval

```python
def fetch_retrieval(
    query_text: str,
    pipeline: Optional[RAGPipeline] = None,
    top_k: int = 5,
    max_chars: int = 4000,
    use_reranking: bool = False,
    reranking_top_k: Optional[int] = None
) -> Dict[str, Any]
```

## Contributing

To add a new reranker:

1. Implement `IReranker` interface in `rerankers/`
2. Add factory method to `RerankerFactory`
3. Update this documentation
4. Add tests to `test_reranker_integration.py`
