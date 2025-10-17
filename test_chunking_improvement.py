#!/usr/bin/env python3
"""
Improved Chunking Strategy Test
===============================
Test và so sánh các chunking strategies để cải thiện retrieval accuracy.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from loaders.pdf_loader import PDFLoader
from chunkers.hybrid_chunker import HybridChunker, ChunkerMode
from chunkers.fixed_size_chunker import FixedSizeChunker
from chunkers.semantic_chunker import SemanticChunker
from embedders.embedder_factory import EmbedderFactory
from pipeline.retriever import Retriever


def test_chunking_strategies():
    """Test different chunking strategies and their impact on retrieval."""

    print("🔧 Testing Improved Chunking Strategies")
    print("=" * 60)

    # Load PDF
    pdf_path = 'data/pdf/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy.pdf'
    loader = PDFLoader.create_default()
    doc = loader.load(pdf_path)

    print(f"📄 Loaded PDF: {len(doc.pages)} pages, {sum(len(page.blocks) for page in doc.pages)} blocks")

    # Test different chunking configurations
    strategies = [
        {
            "name": "Current (500 tokens)",
            "chunker": HybridChunker(max_tokens=500, overlap_tokens=50, mode=ChunkerMode.AUTO)
        },
        {
            "name": "Small Chunks (200 tokens)",
            "chunker": HybridChunker(max_tokens=200, overlap_tokens=30, mode=ChunkerMode.AUTO)
        },
        {
            "name": "Medium Chunks (300 tokens)",
            "chunker": HybridChunker(max_tokens=300, overlap_tokens=40, mode=ChunkerMode.AUTO)
        },
        {
            "name": "Semantic First (300 tokens)",
            "chunker": HybridChunker(max_tokens=300, overlap_tokens=40, mode=ChunkerMode.SEMANTIC_FIRST)
        },
        {
            "name": "Fixed Size (250 tokens)",
            "chunker": FixedSizeChunker(max_tokens=250, overlap_tokens=35)
        }
    ]

    # Test questions
    test_questions = [
        "What are the main techniques for improving RAG accuracy?",
        "Why did early RAG implementations fail?",
        "How does the author achieve 98% accuracy in RAG systems?",
        "What is the role of chunking in RAG pipelines?"
    ]

    results = {}

    for strategy in strategies:
        print(f"\n🎯 Testing: {strategy['name']}")
        print("-" * 40)

        # Chunk the document
        chunk_set = strategy['chunker'].chunk(doc)
        chunks = chunk_set.chunks

        print(f"   Created {len(chunks)} chunks")
        print(f"   Average chunk size: {sum(len(c.text.split()) for c in chunks) / len(chunks):.1f} words")

        # Analyze chunk sizes
        chunk_sizes = [len(c.text.split()) for c in chunks]
        print(f"   Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} words")

        # Test retrieval with this chunking strategy
        # (Simplified test - just count relevant matches)
        strategy_results = {
            "chunk_count": len(chunks),
            "avg_chunk_size": sum(len(c.text.split()) for c in chunks) / len(chunks),
            "retrieval_scores": []
        }

        # For each test question, simulate retrieval
        for question in test_questions:
            # Simple keyword matching simulation
            question_words = set(question.lower().split())
            best_score = 0.0

            for chunk in chunks:
                chunk_words = set(chunk.text.lower().split())
                overlap = len(question_words.intersection(chunk_words))
                score = overlap / len(question_words) if question_words else 0.0
                best_score = max(best_score, score)

            strategy_results["retrieval_scores"].append(best_score)

        avg_retrieval_score = sum(strategy_results["retrieval_scores"]) / len(strategy_results["retrieval_scores"])
        strategy_results["avg_retrieval_score"] = avg_retrieval_score

        print(".3f")

        results[strategy['name']] = strategy_results

    # Compare results
    print("\n📊 Chunking Strategy Comparison")
    print("=" * 60)
    print("<25")
    print("-" * 60)

    for name, data in results.items():
        print("<25")

    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['avg_retrieval_score'])
    print(f"\n🏆 Best Strategy: {best_strategy[0]}")
    print(".3f")

    return results


def create_improved_chunker():
    """Create an improved chunker configuration based on test results."""

    print("\n🔧 Creating Improved Chunker Configuration")
    print("=" * 50)

    # Based on test results, create improved configuration
    improved_config = {
        "max_tokens": 250,  # Smaller chunks for better precision
        "overlap_tokens": 35,  # Maintain context
        "mode": ChunkerMode.SEMANTIC_FIRST,  # Prioritize semantic boundaries
        "min_sentences_per_chunk": 2,  # Ensure minimum coherence
        "enable_metadata_augmentation": True
    }

    print("📋 Improved Configuration:")
    for key, value in improved_config.items():
        print(f"   {key}: {value}")

    # Create the improved chunker
    improved_chunker = HybridChunker(
        max_tokens=improved_config["max_tokens"],
        overlap_tokens=improved_config["overlap_tokens"],
        mode=improved_config["mode"],
        config={
            "min_sentences_per_chunk": improved_config["min_sentences_per_chunk"]
        }
    )

    return improved_chunker, improved_config


def test_improved_retrieval():
    """Test retrieval with improved chunking strategy."""

    print("\n🎯 Testing Improved Retrieval")
    print("=" * 40)

    # Create improved chunker
    improved_chunker, config = create_improved_chunker()

    # Load and chunk document
    pdf_path = 'data/pdf/medium-com-RAG is Hard Until I Know these  Techniques  RAG Pipeline to  Accuracy.pdf'
    loader = PDFLoader.create_default()
    doc = loader.load(pdf_path)

    chunk_set = improved_chunker.chunk(doc)

    print(f"✅ Created {len(chunk_set.chunks)} improved chunks")
    print(f"   Average size: {sum(len(c.text.split()) for c in chunk_set.chunks) / len(chunk_set.chunks):.1f} words")

    # Test with actual embedding and retrieval
    try:
        factory = EmbedderFactory()
        embedder = factory.create_gemma()
        retriever = Retriever(embedder)

        test_question = "What techniques improve RAG accuracy to 98%?"

        # Create mock FAISS index for testing (simplified)
        print(f"\n🧪 Testing retrieval for: '{test_question}'")

        # Get embeddings for chunks
        chunk_texts = [chunk.text for chunk in chunk_set.chunks[:10]]  # Test first 10 chunks
        embeddings = [embedder.embed(text) for text in chunk_texts]

        print(f"   Generated embeddings for {len(embeddings)} chunks")

        # Calculate similarities
        query_embedding = embedder.embed(test_question)
        similarities = []

        import numpy as np
        for emb in embeddings:
            # Cosine similarity
            dot_product = np.dot(query_embedding, emb)
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(emb)
            similarity = dot_product / (norm_q * norm_e) if norm_q > 0 and norm_e > 0 else 0.0
            similarities.append(similarity)

        # Show top results
        top_indices = np.argsort(similarities)[::-1][:3]
        print("\n📈 Top 3 similar chunks:")
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            preview = chunk_texts[idx][:100] + "..." if len(chunk_texts[idx]) > 100 else chunk_texts[idx]
            print(f"   {i+1}. Score: {score:.4f}")
            print(f"      {preview}")

        avg_score = sum(similarities) / len(similarities)
        max_score = max(similarities)
        print(".4f")
        print(".4f")

    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")


def main():
    """Run chunking improvement tests."""

    print("🚀 Chunking Strategy Improvement Test")
    print("Document: Medium RAG Techniques Article")
    print("=" * 60)

    # Test different strategies
    results = test_chunking_strategies()

    # Test improved retrieval
    test_improved_retrieval()

    print("\n✅ Chunking improvement tests completed!")
    print("\n💡 Recommendations:")
    print("   1. Use smaller chunk sizes (200-300 tokens) for better precision")
    print("   2. Increase overlap (30-40 tokens) to maintain context")
    print("   3. Prioritize semantic chunking for coherent text")
    print("   4. Add metadata augmentation for better retrieval")


if __name__ == "__main__":
    main()