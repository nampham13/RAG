import sys
sys.path.append('C:/Users/ENGUYEHWC/Prototype/Version_03/RAG')

from pipeline.pipeline_qa import fetch_retrieval

# Test retrieval
result = fetch_retrieval("configuration management", top_k=5)
print("Context length:", len(result.get("context", "")))
print("Number of sources:", len(result.get("sources", [])))

for i, src in enumerate(result.get("sources", [])):
    print(f"\nSource {i+1}:")
    print(f"  File: {src.get('file_name', 'N/A')}")
    print(f"  Page: {src.get('page_number', 'N/A')}")
    print(f"  Score: {src.get('similarity_score', 0):.4f}")
    text = src.get('text', '')
    print(f"  Text length: {len(text)}")
    print(f"  Text preview: {repr(text[:100])}")