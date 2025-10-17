# RAG Pipeline - Quick Start Guide

## 🚀 Chạy Pipeline

### 1. Xử Lý Tất Cả PDFs

```powershell
python rag_pipeline.py
```

**Output**: FAISS indexes + metadata cho tất cả PDFs trong `data/pdf/`

---

## 📂 Output Files (Chỉ Files Cần Thiết)

Mỗi PDF tạo ra **3 files**:

1. **`.faiss`** - Vector index (binary)
2. **`.pkl`** - Metadata map (binary)
3. **`_summary.json`** - Document info (text)

data/
├── vectors/
│   ├── Document_vectors_20251013.faiss       ← Vector index
│   └── Document_metadata_map_20251013.pkl    ← Metadata
└── metadata/
    └── Document_summary_20251013.json        ← Summary

---

## 🔍 Search (Demo)

```powershell
python demo_faiss_search.py
```

**Tự động**:

- Tìm latest FAISS index
- Test với 4 queries
- Hiển thị top 3 results

---

## 💻 Sử Dụng Trong Code

```python
from rag_pipeline import RAGPipeline
from pathlib import Path

# Initialize
pipeline = RAGPipeline()

# Search
results = pipeline.search_similar(
    faiss_file=Path("data/vectors/Doc_vectors.faiss"),
    metadata_map_file=Path("data/vectors/Doc_metadata_map.pkl"),
    query_text="your query here",
    top_k=5
)

# Results
for r in results:
    print(f"{r['similarity_score']:.4f} - {r['text'][:100]}")
    print(f"Page {r['page_number']}")
```

---

## ✅ Hoàn Tất

**3 files/PDF** - Compact, fast, production-ready! 🎯
