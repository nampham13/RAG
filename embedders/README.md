# Embedders Module - Integrated Chunking + Embedding

## 🎯 Tổng quan

Module `embedders` cung cấp hệ thống embedding tích hợp với chunking, hỗ trợ nhiều providers (Ollama, OpenAI, BGE) và cho phép chuyển đổi model nhanh chóng.

**Kiến trúc chính:**
PDF/Text → PDFLoader → PDFDocument → HybridChunker → ChunkSet → ChunkSetEmbedder → EmbeddingResults

## 🚀 Cách sử dụng nhanh

### 1. Pipeline tích hợp (Khuyến nghị)

```python
from embedders import ChunkAndEmbedPipeline, EmbedderFactory

# Tạo embedder với Ollama
factory = EmbedderFactory()
embedder = factory.create_ollama_nomic()

# Tạo pipeline tích hợp
pipeline = ChunkAndEmbedPipeline(embedder=embedder)

# Xử lý PDF end-to-end
results = pipeline.process_pdf("document.pdf")

# Hoặc xử lý text
results = pipeline.process_text("Your text content here")
```

### 2. Chuyển đổi model nhanh

```python
# Kiểm tra models available
available_models = pipeline.get_available_models()
print(f"Available models: {available_models}")

# Chuyển model
pipeline.switch_embedder_model("all-MiniLM-L6-v2")
```

### 3. Sử dụng riêng lẻ

```python
from embedders import EmbedderFactory, EmbedderType, ChunkSetEmbedder
from chunkers import HybridChunker
from loaders import PDFLoader

# Load và chunk
loader = PDFLoader.create_default()
pdf_doc = loader.load("doc.pdf")
pdf_doc = pdf_doc.normalize()

chunker = HybridChunker()
chunk_set = chunker.chunk(pdf_doc)

# Embed
embedder = EmbedderFactory().create_ollama_nomic()
orchestrator = ChunkSetEmbedder(embedder)
results = orchestrator.embed_chunk_set(chunk_set)
```

## 🔧 Providers hỗ trợ

### Ollama (Khuyến nghị cho local)

```python
from embedders import EmbedderFactory

factory = EmbedderFactory()

# Nomic embed (default)
embedder = factory.create_ollama_nomic()

# MiniLM
embedder = factory.create_ollama_minilm()

# Custom base URL
embedder = factory.create_ollama_nomic(base_url="http://192.168.1.100:11434")
```

**Cài đặt Ollama:**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull nomic-embed-text
ollama pull all-MiniLM-L6-v2

# Start server
ollama serve
```

### OpenAI

```python
embedder = factory.create_openai_ada(api_key="your-api-key")
```

### BGE (HuggingFace)

```python
from embedders import EmbeddingProfile, EmbedderType

profile = EmbeddingProfile.create_bge_large()
embedder = factory.create(EmbedderType.BGE, profile)
```

## 📊 Pipeline Features

### ChunkAndEmbedPipeline

- **Tích hợp end-to-end**: PDF → Chunks → Embeddings
- **Model switching**: Đổi model Ollama nhanh chóng
- **Parallel processing**: Xử lý nhiều chunks song song
- **Metadata inclusion**: Bao gồm metadata trong embedding
- **Context awareness**: Thêm context từ chunks lân cận
- **Error handling**: Robust error handling và logging

### ChunkSetEmbedder

- **Batch processing**: Tối ưu cho nhiều chunks
- **Statistics**: Báo cáo chi tiết về quá trình embedding
- **Provider agnostic**: Hoạt động với mọi embedder

## 🏗️ Architecture

### Core Classes

IEmbedder (interface)
├── BaseEmbedder (abstract base)
│   ├── OllamaEmbedder
│   ├── OpenAIEmbedder
│   └── BGEEmbedder

### Data Flow

1. **Input**: PDF file hoặc raw text
2. **Loading**: PDFLoader extract text/tables
3. **Normalization**: Clean và chuẩn hóa data
4. **Chunking**: HybridChunker tạo chunks
5. **Embedding**: Embedder tạo vectors
6. **Output**: Dict[chunk_id → EmbeddingResult]

## ⚙️ Configuration

### EmbeddingProfile

```python
from embedders import EmbeddingProfile

# Ollama profiles
profile = EmbeddingProfile.create_ollama_nomic()
profile = EmbeddingProfile.create_ollama_minilm()

# Custom profile
profile = EmbeddingProfile(
    model_id="custom-model",
    provider="ollama",
    max_tokens=1024,
    dimension=512,
    normalize=True
)
```

### Pipeline Settings

```python
pipeline = ChunkAndEmbedPipeline(
    embedder=embedder,
    include_metadata=True,      # Include chunk metadata
    include_context=False,      # Add adjacent chunk context
    parallel_workers=2          # Parallel embedding workers
)
```

## 📈 Performance Tips

### Ollama Optimization

- Sử dụng models nhỏ hơn cho tốc độ: `all-MiniLM-L6-v2`
- Tăng batch_size nếu có GPU
- Monitor RAM usage với models lớn

### Parallel Processing

```python
# Tăng workers cho nhiều chunks
pipeline = ChunkAndEmbedPipeline(
    embedder=embedder,
    parallel_workers=4  # Sử dụng 4 threads
)
```

### Memory Management

- Xử lý files lớn từng phần
- Clear results sau khi sử dụng
- Monitor embedding dimensions

## 🔍 Monitoring & Debugging

### Pipeline Info

```python
info = pipeline.get_pipeline_info()
print(f"Embedder: {info['embedder']['model']}")
print(f"Chunker: {info['chunker']['type']}")
```

### Embedding Statistics

```python
stats = orchestrator.get_embedding_stats(results)
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Avg processing time: {stats['avg_processing_time']:.3f}s")
```

### Error Handling

```python
for chunk_id, result in results.items():
    if result.status != "success":
        print(f"Failed {chunk_id}: {result.error_message}")
```

## 🎯 Best Practices

1. **Chọn model phù hợp**: Nomic cho chất lượng, MiniLM cho tốc độ
2. **Monitor resources**: Theo dõi CPU/Memory với models lớn
3. **Batch processing**: Sử dụng parallel workers cho nhiều documents
4. **Error recovery**: Handle network issues với Ollama
5. **Caching**: Cache embeddings để tránh re-processing

## 🚨 Troubleshooting

### Ollama Issues

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Check logs
ollama logs
```

### Common Errors

- **Connection refused**: Ollama không chạy
- **Model not found**: Chưa pull model
- **Out of memory**: Giảm batch_size hoặc dùng model nhỏ hơn
- **Timeout**: Tăng timeout hoặc kiểm tra network

## 📝 Examples

Xem `run_embed_demo.py` để có ví dụ đầy đủ về cách sử dụng pipeline.
