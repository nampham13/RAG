from embedders.embedder_factory import EmbedderFactory
import numpy as np
from pathlib import Path
import pickle
import faiss

# Initialize embedder
factory = EmbedderFactory()
embedder = factory.create_gemma()

# Load FAISS index and metadata
vectors_dir = Path("data/vectors")
faiss_files = list(vectors_dir.glob("*_vectors_*.faiss"))
metadata_files = list(vectors_dir.glob("*_metadata_map_*.pkl"))

if faiss_files and metadata_files:
    faiss_file = faiss_files[0]
    metadata_file = metadata_files[0]
    
    index = faiss.read_index(str(faiss_file))
    with open(metadata_file, 'rb') as f:
        metadata_map = pickle.load(f)
    
    # Test different queries
    queries = [
        "Query Rewriting",
        "Query Rewriting technique",
        "Transforms ambiguous queries",
        "clarify intent first",
        "improve RAG pipeline"
    ]
    
    print("Embedding similarity analysis:")
    print("=" * 80)
    
    for query in queries:
        query_embedding = embedder.embed(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize
        faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = index.search(query_vector, 10)
        
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        
        for i, (dist, idx) in enumerate(zip(distances[0][:6], indices[0][:6]), 1):
            page = metadata_map[idx].get('page_number', '?')
            text_preview = metadata_map[idx].get('text', '')[:50]
            similarity = 1 - dist  # Convert distance to similarity
            has_keyword = any(k in metadata_map[idx].get('text', '').lower() for k in ['query rewriting', 'query routing', 'clarify intent'])
            keyword_marker = " *** HAS KEYWORD ***" if has_keyword else ""
            print(f"  {i}. Page {page}: {similarity:.4f} - {text_preview}...{keyword_marker}")
