from pipeline.rag_pipeline import RAGPipeline
from pipeline.pipeline_qa import RAGRetrievalService
from pipeline.query_expander import QueryExpander

# Test query expansion effect
expander = QueryExpander()
pipeline = RAGPipeline(output_dir="data")
service = RAGRetrievalService(pipeline)

query = "Query Rewriting"
expanded = expander.expand(query)

print(f"Original query: {query}")
print(f"Expanded query: {expanded}")
print()

# Get results with expanded query
results = service.retrieve(query, top_k=10)

print("Top 10 results with query expansion:")
print("-" * 60)
for i, result in enumerate(results[:10], 1):
    page = result.get('page_number', '?')
    score = result.get('similarity_score', 0)
    has_qr = 'Query Rewriting' in result.get('text', '')
    marker = " *** QUERY REWRITING ***" if has_qr else ""
    print(f"{i}. Page {page}: {score:.4f}{marker}")