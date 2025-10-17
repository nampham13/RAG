"""
Demo: Chunk PDF document using HybridChunker
"""

from loaders.pdf_loader import PDFLoader
from chunkers.hybrid_chunker import HybridChunker

pdf_path = r"C:\Users\ENGUYEHWC\Prototype\Version_03\RAG\data\pdf\Process_Service Management.pdf"

# 1. Load PDF
loader = PDFLoader.create_default()
pdf_doc = loader.load(pdf_path)
pdf_doc = pdf_doc.normalize()  # Chuẩn hóa dữ liệu

# 2. Chunk PDFDocument
chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
chunk_set = chunker.chunk(pdf_doc)

# 3. Print chunk results
print(f"Chunk count: {len(chunk_set.chunks)}")
with open("chunk_output.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunk_set.chunks):
        f.write(f"\n=== Chunk {i+1} ===\n")
        text = getattr(chunk, 'textForEmbedding', getattr(chunk, 'text', ''))
        f.write(f"Tokens: {getattr(chunk, 'tokensEstimate', getattr(chunk, 'token_count', 'N/A'))}\n")
        f.write(f"Content:\n{text}\n\n")
        # Print metadata
        if hasattr(chunk, 'metadata') and chunk.metadata:
            f.write(f"Metadata: {chunk.metadata}\n\n")
        if hasattr(chunk, 'provenance') and chunk.provenance:
            f.write(f"Provenance: {chunk.provenance}\n")
        # Nếu là bảng, xuất chi tiết schema
        if chunk.metadata.get("block_type") == "table" or chunk.metadata.get("group_type") == "table":
            table_payload = chunk.metadata.get("table_payload")
            if table_payload:
                # TableSchema có rows, header, bbox, metadata (không có matrix)
                rows = getattr(table_payload, "rows", [])
                header = getattr(table_payload, "header", [])
                caption = getattr(table_payload, "metadata", {}).get("table_caption", None) if hasattr(table_payload, "metadata") else None
                bbox = getattr(table_payload, "bbox", None)
                
                f.write(f"--- Table Schema ---\n")
                f.write(f"Table Header: {header}\n")
                f.write(f"Table Rows Count: {len(rows)}\n")
                f.write(f"Table Caption: {caption}\n")
                f.write(f"Table BBox: {bbox}\n")
                # Print first 3 rows as sample
                for idx, row in enumerate(rows[:3]):
                    cells = [c.value for c in row.cells] if hasattr(row, "cells") else []
                    f.write(f"  Row {idx+1}: {cells}\n")
                if len(rows) > 3:
                    f.write(f"  ... ({len(rows)-3} more rows)\n")
            else:
                f.write("Table Payload: None\n")
print("Đã xuất chi tiết các chunk ra file chunk_output.txt")
