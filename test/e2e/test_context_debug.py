from pipeline.pipeline_qa import fetch_retrieval

# Test build_context chi tiết
query = 'Query Rewriting'
result = fetch_retrieval(query, top_k=10, max_chars=1000)

# In ra từng source chi tiết
sources = result.get('sources', [])
print(f"Total sources: {len(sources)}")
print()

for i, source in enumerate(sources[:3], 1):
    page = source.get('page_number', '?')
    score = source.get('similarity_score', 0.0)
    text_len = len(source.get('text', ''))
    has_qr = 'Query Rewriting' in source.get('text', '')
    print(f"{i}. Page {page} - Score: {score:.4f} - Text length: {text_len} - Has Query Rewriting: {has_qr}")