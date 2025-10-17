from pipeline.pipeline_qa import fetch_retrieval

# Test the retrieval with Query Rewriting
print("Testing fetch_retrieval with 'Query Rewriting'...")

result = fetch_retrieval("Query Rewriting", top_k=10)
context = result.get("context", "")
sources = result.get("sources", [])

print(f"Context length: {len(context)} characters")
print(f"Number of sources: {len(sources)}")

# Check if any sources contain "Query Rewriting"
found_query_rewriting = False
for i, source in enumerate(sources, 1):
    page = source.get('page_number', 'unknown')
    score = source.get('similarity_score', 0)
    text_preview = source.get('text', '')[:100]
    print(f"{i}. Page {page}: {score:.4f} - {text_preview}...")

    if 'Query Rewriting' in source.get('text', ''):
        found_query_rewriting = True
        print("   *** FOUND QUERY REWRITING ***")

print(f"\nQuery Rewriting found: {found_query_rewriting}")

# Show context preview
if context:
    print(f"\nContext preview: {context[:500]}...")
else:
    print("\nNo context retrieved")