"""
Test script for reranker integration in RAG pipeline.
Tests both BM25 and BGE rerankers.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.rag_pipeline import RAGPipeline
from embedders.providers.ollama import OllamaModelType


def test_bm25_reranker():
    """Test BM25 reranker integration."""
    print("\n" + "="*80)
    print("Testing BM25 Reranker Integration")
    print("="*80)
    
    try:
        # Initialize pipeline with BM25 reranker
        pipeline = RAGPipeline(
            output_dir="data",
            model_type=OllamaModelType.GEMMA,
            reranker_type="bm25"
        )
        
        print(f"\n✓ Pipeline initialized with BM25 reranker")
        print(f"  Embedder: {pipeline.embedder.profile.model_id}")
        print(f"  Reranker: {pipeline.reranker_type}")
        print(f"  Reranker instance: {type(pipeline.reranker).__name__ if pipeline.reranker else 'None'}")
        
        # Get info
        info = pipeline.get_info()
        print(f"\nPipeline Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Try to find an index to test with
        from pipeline.pipeline_qa import RAGRetrievalService
        service = RAGRetrievalService(pipeline)
        pair = service.get_latest_index_pair()
        
        if pair:
            faiss_file, metadata_file = pair
            print(f"\n✓ Found index files:")
            print(f"  FAISS: {faiss_file.name}")
            print(f"  Metadata: {metadata_file.name}")
            
            # Test retrieval without reranking
            print("\n--- Testing retrieval WITHOUT reranking ---")
            results_no_rerank = pipeline.search_similar(
                faiss_file=faiss_file,
                metadata_map_file=metadata_file,
                query_text="What is configuration management?",
                top_k=5,
                use_reranking=False
            )
            
            print(f"Results: {len(results_no_rerank)} chunks found")
            for i, result in enumerate(results_no_rerank[:3], 1):
                print(f"\n  [{i}] Score: {result.get('similarity_score', 0):.4f}")
                print(f"      Text: {result.get('text', '')[:100]}...")
            
            # Test retrieval WITH reranking
            print("\n--- Testing retrieval WITH BM25 reranking ---")
            results_with_rerank = pipeline.search_similar(
                faiss_file=faiss_file,
                metadata_map_file=metadata_file,
                query_text="What is configuration management?",
                top_k=5,
                use_reranking=True,
                reranking_top_k=15
            )
            
            print(f"Results: {len(results_with_rerank)} chunks found")
            for i, result in enumerate(results_with_rerank[:3], 1):
                has_rerank = 'rerank_score' in result
                print(f"\n  [{i}] Score: {result.get('similarity_score', 0):.4f} {'(reranked)' if has_rerank else ''}")
                if has_rerank:
                    print(f"      Rerank score: {result.get('rerank_score', 0):.4f}")
                print(f"      Text: {result.get('text', '')[:100]}...")
            
            print("\n✓ BM25 reranker test completed successfully!")
            
        else:
            print("\n⚠ No index files found - skipping search test")
            print("  Run 'python run_pipeline.py' to create indexes first")
            
    except Exception as e:
        print(f"\n✗ Error testing BM25 reranker: {e}")
        import traceback
        traceback.print_exc()


def test_bge_reranker():
    """Test BGE reranker integration."""
    print("\n" + "="*80)
    print("Testing BGE Reranker Integration")
    print("="*80)
    
    try:
        # Initialize pipeline with BGE reranker
        pipeline = RAGPipeline(
            output_dir="data",
            model_type=OllamaModelType.GEMMA,
            reranker_type="bge"
        )
        
        print(f"\n✓ Pipeline initialized with BGE reranker")
        print(f"  Embedder: {pipeline.embedder.profile.model_id}")
        print(f"  Reranker: {pipeline.reranker_type}")
        print(f"  Reranker instance: {type(pipeline.reranker).__name__ if pipeline.reranker else 'None'}")
        
        # Get info
        info = pipeline.get_info()
        print(f"\nPipeline Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n✓ BGE reranker test completed successfully!")
        print("  Note: Full search test skipped for BGE (requires transformers)")
        
    except Exception as e:
        print(f"\n⚠ BGE reranker requires transformers and torch packages")
        print(f"  Error: {e}")


def test_no_reranker():
    """Test pipeline without reranker (baseline)."""
    print("\n" + "="*80)
    print("Testing Pipeline WITHOUT Reranker (Baseline)")
    print("="*80)
    
    try:
        # Initialize pipeline without reranker
        pipeline = RAGPipeline(
            output_dir="data",
            model_type=OllamaModelType.GEMMA,
            reranker_type=None
        )
        
        print(f"\n✓ Pipeline initialized without reranker")
        print(f"  Embedder: {pipeline.embedder.profile.model_id}")
        print(f"  Reranker: {pipeline.reranker_type or 'None'}")
        print(f"  Reranker instance: {type(pipeline.reranker).__name__ if pipeline.reranker else 'None'}")
        
        # Get info
        info = pipeline.get_info()
        print(f"\nPipeline Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n✓ Baseline test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error testing baseline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RAG Pipeline Reranker Integration Tests")
    print("="*80)
    
    # Test without reranker first (baseline)
    test_no_reranker()
    
    # Test BM25 reranker
    test_bm25_reranker()
    
    # Test BGE reranker
    test_bge_reranker()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
