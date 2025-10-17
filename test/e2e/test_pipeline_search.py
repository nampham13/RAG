from pathlib import Path
from pipeline.rag_pipeline import RAGPipeline
from embedders.providers.ollama import OllamaModelType
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize pipeline exactly like the test
pipeline = RAGPipeline(
    output_dir="data",
    model_type=OllamaModelType.GEMMA
)

# Find the FAISS and metadata files
vectors_dir = Path("data/vectors")
faiss_files = list(vectors_dir.glob("*_vectors_*.faiss"))
metadata_files = list(vectors_dir.glob("*_metadata_map_*.pkl"))

if not faiss_files or not metadata_files:
    print("No FAISS or metadata files found")
    exit(1)

faiss_file = faiss_files[0]
metadata_file = metadata_files[0]

print(f"Using FAISS file: {faiss_file}")
print(f"Using metadata file: {metadata_file}")

# Test the exact query
query = "Query Rewriting"
print(f"\n=== Testing query: '{query}' ===")

results = pipeline.search_similar(
    faiss_file=faiss_file,
    metadata_map_file=metadata_file,
    query_text=query,
    top_k=10  # Get more results to see if 17 and 18 appear
)

for i, result in enumerate(results, 1):
    print(f"{i}. Page {result['page_number']}: {result['similarity_score']:.4f}")
    if result['page_number'] == 10:
        print(f"   *** PAGE 10 CONTENT ***")
        print(f"   Preview: {result.get('text', '')[:200]}...")
    print()