# Chunker Models

## Cấu trúc thư mục

```
chunkers/
├── model/                    # Thư mục chứa tất cả data models
│   ├── __init__.py          # Export tất cả models
│   ├── enums.py             # ChunkType, ChunkStrategy enums
│   ├── block_span.py        # BlockSpan model
│   ├── provenance_agg.py    # ProvenanceAgg model
│   ├── score.py             # Score model
│   ├── chunk.py             # Chunk model
│   ├── chunk_set.py         # ChunkSet model
│   └── chunk_stats.py       # ChunkStats model
├── model.py                 # Legacy import (tương thích ngược)
├── base_chunker.py          # Abstract base class
├── hybrid_chunker.py        # Main orchestrator
├── semantic_chunker.py      # Semantic chunking
├── rule_based_chunker.py    # Rule-based chunking
├── fixed_size_chunker.py    # Fixed-size chunking
├── examples.py              # Usage examples
└── __init__.py              # Package exports
```

## Nguyên tắc thiết kế

### Single Responsibility Principle
Mỗi class có một trách nhiệm duy nhất:

- **BlockSpan**: Quản lý vị trí character offset trong source blocks
- **ProvenanceAgg**: Theo dõi nguồn gốc và metadata của chunk
- **Score**: Đo lường và đánh giá chất lượng chunk
- **Chunk**: Lưu trữ nội dung chunk và metadata liên quan
- **ChunkSet**: Quản lý collection của chunks và thống kê tổng thể
- **ChunkStats**: Tính toán và lưu trữ thống kê chunking

### Tách biệt Models và Logic
- Models chỉ chứa data và basic validation
- Business logic được tách riêng vào các chunker classes
- Dễ test và maintain

## Cách sử dụng

### Import từ package
```python
from chunkers import (
    HybridChunker, ChunkType, ChunkSet,
    ChunkerMode
)
```

### Import trực tiếp từ models
```python
from chunkers.model import Chunk, Score, ProvenanceAgg
```

### Import legacy (tương thích ngược)
```python
from chunkers.model import ChunkType, Chunk  # Vẫn hoạt động
```

## Lợi ích của cấu trúc mới

1. **Modularity**: Mỗi model trong file riêng, dễ maintain
2. **Clarity**: Code rõ ràng, dễ hiểu
3. **Testability**: Dễ test từng phần riêng biệt
4. **Extensibility**: Dễ thêm models mới
5. **Backward Compatibility**: Code cũ vẫn hoạt động

## Chạy Examples

```bash
cd RAG
python chunkers/examples.py
```

## Chunking Strategies

### HybridChunker (Main)
- **AUTO**: Tự động chọn strategy tốt nhất
- **SEMANTIC_FIRST**: Ưu tiên semantic chunking
- **STRUCTURAL_FIRST**: Ưu tiên structural chunking
- **SEQUENTIAL**: Thử structural → semantic → fixed size
- **FIXED_SIZE**: Chỉ dùng fixed-size chunking

### Individual Chunkers
- **SemanticChunker**: Dựa trên coherence và discourse markers
- **RuleBasedChunker**: Dựa trên headings, lists, tables
- **FixedSizeChunker**: Dựa trên token count cố định

## Quality Metrics

Mỗi chunk có các metrics đánh giá chất lượng:
- **Coherence Score**: Độ liên kết ngữ nghĩa (0-1)
- **Completeness Score**: Độ hoàn chỉnh thông tin (0-1)
- **Token Ratio**: Tỷ lệ token_count/max_tokens
- **Structural Integrity**: Điểm toàn vẹn cấu trúc (0-1)
- **Overall Score**: Điểm tổng thể có trọng số