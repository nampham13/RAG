#!/usr/bin/env python3
"""
Script kiểm tra Vector Store
Kiểm tra FAISS indexes và metadata
"""
import sys
import pickle
from pathlib import Path
import faiss

print("🔍 Checking Vector Store...\n")

# === 1. Check Pipeline Initialization ===
print("1️⃣ Kiểm tra Pipeline...")
try:
    from pipeline import RAGPipeline
    pipeline = RAGPipeline()
    print("✅ Pipeline initialized successfully\n")
except Exception as e:
    print(f"❌ Pipeline error: {e}\n")
    sys.exit(1)

# === 2. Check FAISS Indexes ===
print("2️⃣ Kiểm tra FAISS Indexes...")
vectors_dir = Path('data/vectors')
faiss_files = sorted(vectors_dir.glob('*_vectors_*.faiss'), key=lambda p: p.stat().st_mtime, reverse=True)

if not faiss_files:
    print("❌ Không tìm thấy FAISS index files\n")
    sys.exit(1)

print(f"✅ Tìm thấy {len(faiss_files)} FAISS index files\n")

# === 3. Check Each Index ===
print("3️⃣ Kiểm tra chi tiết từng index...\n")
for idx, faiss_file in enumerate(faiss_files, 1):
    print(f"Index {idx}: {faiss_file.name}")
    print(f"  Size: {faiss_file.stat().st_size / 1024:.2f} KB")
    
    # Load FAISS index
    try:
        faiss_index = faiss.read_index(str(faiss_file))
        print(f"  ✅ FAISS Loaded - Vectors: {faiss_index.ntotal}, Dimensions: {faiss_index.d}")
    except Exception as e:
        print(f"  ❌ Error loading FAISS: {e}")
        continue
    
    # Load metadata
    metadata_file = faiss_file.with_name(faiss_file.name.replace('_vectors_', '_metadata_map_')).with_suffix('.pkl')
    if metadata_file.exists():
        try:
            with open(metadata_file, 'rb') as f:
                metadata_map = pickle.load(f)
            print(f"  ✅ Metadata Loaded - Chunks: {len(metadata_map)}")
            
            # Check first chunk
            if metadata_map:
                first_chunk = metadata_map.get(0, {})
                print(f"    Page: {first_chunk.get('page_number', 'N/A')}")
                print(f"    Text preview: {first_chunk.get('text', '')[:80]}...")
        except Exception as e:
            print(f"  ❌ Error loading metadata: {e}")
    else:
        print(f"  ❌ Metadata file not found: {metadata_file.name}")
    
    print()

# === 4. Test Retrieval ===
print("4️⃣ Kiểm tra Retrieval Function...\n")
try:
    latest_faiss = faiss_files[0]
    latest_metadata = latest_faiss.with_name(latest_faiss.name.replace('_vectors_', '_metadata_map_')).with_suffix('.pkl')
    
    results = pipeline.search_similar(
        faiss_file=latest_faiss,
        metadata_map_file=latest_metadata,
        query_text="RAG pipeline configuration",
        top_k=3
    )
    
    print(f"✅ Search successful - Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Score: {result.get('cosine_similarity', result.get('similarity_score', 'N/A')):.4f}")
        print(f"  Page: {result.get('page_number', 'N/A')}")
        print(f"  Text: {result.get('text', '')[:100]}...")
        print()
        
except Exception as e:
    print(f"❌ Retrieval error: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 60)
print("✅ Vector Store Check Complete!")
