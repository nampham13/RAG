from pipeline.rag_pipeline import RAGPipeline
from pipeline.pipeline_qa import RAGRetrievalService

# Debug build_context
pipeline = RAGPipeline(output_dir="data")
service = RAGRetrievalService(pipeline)

query = "Query Rewriting"
results = service.retrieve(query, top_k=10)

print(f"Total results: {len(results)}")
print("\nBuilding context with max_chars=2000:")
context = service.build_context(results, max_chars=2000)

print(f"Context length: {len(context)}")

# Check if Query Rewriting is in context
has_qr = 'Query Rewriting' in context
print(f"Has 'Query Rewriting': {has_qr}")

if has_qr:
    start = context.find('Query Rewriting')
    snippet = context[max(0, start-100):start+200]
    print(f"Snippet: ...{snippet}...")
else:
    print("Query Rewriting NOT found in context!")
    # Show first 500 chars
    print(f"Context preview: {context[:500]}")