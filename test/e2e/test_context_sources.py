from pipeline.rag_pipeline import RAGPipeline
from pipeline.pipeline_qa import RAGRetrievalService

# Debug build_context with new settings
pipeline = RAGPipeline(output_dir="data")
service = RAGRetrievalService(pipeline)

query = "Query Rewriting"
results = service.retrieve(query, top_k=10)

print("Testing build_context with max_chars=2000:")
context = service.build_context(results, max_chars=2000)

print(f"Context length: {len(context)}")
print(f"Has Query Rewriting: {'Query Rewriting' in context}")

# Count how many sources are in context
import re
sources = re.findall(r'\[\d+\]', context)
print(f"Number of sources in context: {len(sources)}")
print(f"Sources: {sources}")