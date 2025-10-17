from pipeline.pipeline_qa import fetch_retrieval

# Test với query chính xác
query = 'Query Rewriting'
print(f'Testing query: "{query}"')

result = fetch_retrieval(query, top_k=10)
sources = result.get('sources', [])

print(f'Số sources: {len(sources)}')
for i, source in enumerate(sources, 1):
    page = source.get('page_number', 'unknown')
    score = source.get('similarity_score', 0)
    text = source.get('text', '')
    has_query_rewriting = 'Query Rewriting' in text
    print(f'{i}. Page {page}: {score:.4f} - Contains Query Rewriting: {has_query_rewriting}')
    if has_query_rewriting:
        # Show the relevant part
        start = text.find('Query Rewriting')
        if start >= 0:
            context = text[max(0, start-50):start+150]
            print(f'   Context: ...{context}...')