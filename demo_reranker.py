"""
Demonstration script for using rerankers in RAG pipeline.
Shows how to use BM25 and BGE rerankers for improved retrieval results.

Usage:
    python demo_reranker.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.rag_pipeline import RAGPipeline
from pipeline.pipeline_qa import fetch_retrieval
from embedders.providers.ollama import OllamaModelType


def print_results(results, title, show_rerank_score=False):
    """Helper function to print search results."""
    print(f"\n{title}")
    print("=" * 80)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results[:5], 1):
        similarity = result.get('similarity_score', 0)
        cosine_sim = result.get('cosine_similarity', 0)
        rerank_score = result.get('rerank_score', None)
        page = result.get('page_number', 'N/A')
        text = result.get('text', '')[:150]
        
        print(f"\n[{i}] Page {page}")
        print(f"    Similarity: {similarity:.4f}")
        if show_rerank_score and rerank_score is not None:
            print(f"    Rerank Score: {rerank_score:.4f}")
            print(f"    Cosine Similarity: {cosine_sim:.4f}")
        print(f"    Text: {text}...")


def demo_basic_retrieval():
    """Demo 1: Basic retrieval without reranking."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Retrieval (No Reranking)")
    print("="*80)
    
    # Initialize pipeline without reranker
    pipeline = RAGPipeline(
        output_dir="data",
        model_type=OllamaModelType.GEMMA,
        reranker_type=None
    )
    
    print(f"\nPipeline Configuration:")
    print(f"  Embedder: {pipeline.embedder.profile.model_id}")
    print(f"  Reranker: {pipeline.reranker_type or 'None'}")
    
    # Example query
    query = "What is configuration management?"
    print(f"\nQuery: '{query}'")
    
    # Use fetch_retrieval for convenience
    result = fetch_retrieval(
        query_text=query,
        pipeline=pipeline,
        top_k=5,
        use_reranking=False
    )
    
    print_results(result.get('sources', []), "Results (Vector Search Only)")


def demo_bm25_reranking():
    """Demo 2: BM25 reranking."""
    print("\n" + "="*80)
    print("DEMO 2: BM25 Reranking")
    print("="*80)
    
    # Initialize pipeline with BM25 reranker
    pipeline = RAGPipeline(
        output_dir="data",
        model_type=OllamaModelType.GEMMA,
        reranker_type="bm25"
    )
    
    print(f"\nPipeline Configuration:")
    print(f"  Embedder: {pipeline.embedder.profile.model_id}")
    print(f"  Reranker: {pipeline.reranker_type}")
    print(f"  Reranker Type: {type(pipeline.reranker).__name__ if pipeline.reranker else 'None'}")
    
    # Example query
    query = "What is configuration management?"
    print(f"\nQuery: '{query}'")
    
    # Retrieve without reranking first
    result_no_rerank = fetch_retrieval(
        query_text=query,
        pipeline=pipeline,
        top_k=5,
        use_reranking=False
    )
    
    print_results(result_no_rerank.get('sources', []), "Results WITHOUT Reranking")
    
    # Retrieve with reranking
    result_with_rerank = fetch_retrieval(
        query_text=query,
        pipeline=pipeline,
        top_k=5,
        use_reranking=True,
        reranking_top_k=15  # Retrieve 15 candidates, rerank to top 5
    )
    
    print_results(result_with_rerank.get('sources', []), "Results WITH BM25 Reranking", show_rerank_score=True)
    
    # Show comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Without reranking: {len(result_no_rerank.get('sources', []))} results")
    print(f"With reranking: {len(result_with_rerank.get('sources', []))} results")
    
    # Check if results are different
    no_rerank_ids = [r.get('chunk_id', '') for r in result_no_rerank.get('sources', [])]
    with_rerank_ids = [r.get('chunk_id', '') for r in result_with_rerank.get('sources', [])]
    
    if no_rerank_ids != with_rerank_ids:
        print("\n✓ Reranking changed the order of results!")
    else:
        print("\n✓ Results are in the same order (possibly no better candidates)")


def demo_bge_reranking():
    """Demo 3: BGE reranking (requires transformers)."""
    print("\n" + "="*80)
    print("DEMO 3: BGE Reranking")
    print("="*80)
    
    try:
        # Initialize pipeline with BGE reranker
        pipeline = RAGPipeline(
            output_dir="data",
            model_type=OllamaModelType.GEMMA,
            reranker_type="bge"
        )
        
        if pipeline.reranker is None:
            print("\n⚠ BGE reranker not available (requires transformers and torch)")
            print("  Install with: pip install transformers torch")
            return
        
        print(f"\nPipeline Configuration:")
        print(f"  Embedder: {pipeline.embedder.profile.model_id}")
        print(f"  Reranker: {pipeline.reranker_type}")
        print(f"  Reranker Type: {type(pipeline.reranker).__name__}")
        
        # Example query
        query = "What is configuration management?"
        print(f"\nQuery: '{query}'")
        
        # Retrieve with BGE reranking
        result_with_rerank = fetch_retrieval(
            query_text=query,
            pipeline=pipeline,
            top_k=5,
            use_reranking=True,
            reranking_top_k=15
        )
        
        print_results(result_with_rerank.get('sources', []), "Results WITH BGE Reranking", show_rerank_score=True)
        
    except Exception as e:
        print(f"\n⚠ Error running BGE reranking: {e}")
        print("  BGE reranker requires transformers and torch packages")


def demo_advanced_usage():
    """Demo 4: Advanced reranking configuration."""
    print("\n" + "="*80)
    print("DEMO 4: Advanced Reranking Configuration")
    print("="*80)
    
    # Custom BM25 parameters
    bm25_kwargs = {
        'k1': 1.5,  # Term frequency saturation
        'b': 0.75,  # Length normalization
    }
    
    # Initialize pipeline with custom BM25 settings
    pipeline = RAGPipeline(
        output_dir="data",
        model_type=OllamaModelType.GEMMA,
        reranker_type="bm25",
        reranker_kwargs=bm25_kwargs
    )
    
    print(f"\nPipeline Configuration:")
    print(f"  Embedder: {pipeline.embedder.profile.model_id}")
    print(f"  Reranker: {pipeline.reranker_type}")
    print(f"  Custom Parameters: k1={bm25_kwargs['k1']}, b={bm25_kwargs['b']}")
    
    # Example query
    query = "machine learning algorithms"
    print(f"\nQuery: '{query}'")
    
    # Retrieve with different reranking_top_k values
    print("\n--- Testing different candidate pool sizes ---")
    
    for candidate_size in [10, 20, 30]:
        result = fetch_retrieval(
            query_text=query,
            pipeline=pipeline,
            top_k=5,
            use_reranking=True,
            reranking_top_k=candidate_size
        )
        
        print(f"\nCandidate pool: {candidate_size} → Top 5 results")
        sources = result.get('sources', [])
        if sources:
            print(f"  Top result score: {sources[0].get('similarity_score', 0):.4f}")
            print(f"  5th result score: {sources[min(4, len(sources)-1)].get('similarity_score', 0):.4f}")
        else:
            print("  No results found")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("RAG Pipeline Reranking Demonstration")
    print("="*80)
    print("\nThis demo shows how to use rerankers to improve retrieval results.")
    print("Make sure you have:")
    print("  1. Run 'python run_pipeline.py' to create FAISS indexes")
    print("  2. Ollama server running (for embeddings)")
    print("  3. rank_bm25 installed (for BM25 reranker)")
    
    # Run demos
    try:
        demo_basic_retrieval()
    except Exception as e:
        print(f"\n✗ Demo 1 failed: {e}")
    
    try:
        demo_bm25_reranking()
    except Exception as e:
        print(f"\n✗ Demo 2 failed: {e}")
    
    try:
        demo_bge_reranking()
    except Exception as e:
        print(f"\n✗ Demo 3 failed: {e}")
    
    try:
        demo_advanced_usage()
    except Exception as e:
        print(f"\n✗ Demo 4 failed: {e}")
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)


if __name__ == "__main__":
    main()
