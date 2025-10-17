from pipeline.rag_pipeline import RAGPipeline
from pathlib import Path

# Initialize pipeline
pipeline = RAGPipeline()

# Find the latest FAISS index
data_dir = Path('data/vectors')
faiss_files = list(data_dir.glob('*vectors*.faiss'))
metadata_files = list(data_dir.glob('*metadata*.pkl'))

if not faiss_files or not metadata_files:
    print('No FAISS index found. Run pipeline first.')
    exit(1)

# Use the latest files
faiss_file = max(faiss_files, key=lambda x: x.stat().st_mtime)
metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)

print(f'Using FAISS file: {faiss_file}')
print(f'Using metadata file: {metadata_file}')

# Test with different queries
queries = [
    'Query Rewriting',
    'Query Rewriting technique',
    '11. Query Rewriting',
    'clarify intent first'
]

for query in queries:
    print(f'\n=== Testing query: "{query}" ===')
    results = pipeline.search_similar(
        faiss_file=faiss_file,
        metadata_map_file=metadata_file,
        query_text=query,
        top_k=10  # Increased from 5 to find more results
    )

    for i, result in enumerate(results, 1):  # Show all results
        print(f'  {i}. Page {result["page_number"]}: {result["similarity_score"]:.4f}')
        if 'Query Rewriting' in result['text'][:200]:
            print('     *** CONTAINS QUERY REWRITING ***')