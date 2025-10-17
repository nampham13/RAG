from pipeline.pipeline_qa import fetch_retrieval

# Test với query dài hơn
queries = [
    'tell me about 12 method for improve rag',
    'What is the re ranking?',
    'What is the pros and cons of page index?',
    'Why we should be use Contextual Retrieval?',
    'Can reranking improve accuracy by up to 60%?',
    'tell me a fact of re ranking',
    'Hybrid RAG define?'
]

for query in queries:
    print(f'Query: {query}')
    try:
        result = fetch_retrieval(query, top_k=5, max_chars=4000)
        print(f'Context length: {len(result["context"])}')
        print(f'Sources count: {len(result["sources"])}')
        if len(result['context']) > 3000:
            print('WARNING: Context too long!')
    except Exception as e:
        print(f'Error: {e}')