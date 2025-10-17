from pipeline.rag_pipeline import RAGPipeline
from pipeline.pipeline_qa import RAGRetrievalService

# Check chunks in detail
pipeline = RAGPipeline(output_dir="data")
service = RAGRetrievalService(pipeline)

query = "Query Rewriting"
results = service.retrieve(query, top_k=10)

print("Chunks from retrieval:")
for i, r in enumerate(results[:6], 1):
    page = r.get('page_number', '?')
    score = r.get('similarity_score', 0.0)
    text = r.get('text', '')
    has_qr = 'Query Rewriting' in text
    
    print(f"\n{i}. Page {page} - Score: {score:.4f} - Has Query Rewriting: {has_qr}")
    print(f"   Text length: {len(text)}")
    print(f"   First 100 chars: {text[:100]}...")
    if has_qr:
        pos = text.find('Query Rewriting')
        print(f"   Query Rewriting position: {pos}")
        print(f"   Context: ...{text[max(0,pos-50):pos+100]}...")