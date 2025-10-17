#!/usr/bin/env python3
"""
Test RAG System - Retrieval, Scoring, and LM Integration
========================================================
Tests the complete RAG pipeline with questions about the Medium RAG article.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.pipeline_qa import RAGRetrievalService
from pipeline.rag_pipeline import RAGPipeline
from embedders.embedder_factory import EmbedderFactory

# Import LLM functions
try:
    from llm.LLM_API import call_gemini
    from llm.config_loader import resolve_gemini_settings
except ImportError:
    from LLM_API import call_gemini
    from config_loader import resolve_gemini_settings


def test_retrieval_system():
    """Test the retrieval system with various questions about RAG techniques."""

    print("🔍 Testing RAG Retrieval System")
    print("=" * 50)

    # Initialize pipeline and retriever
    pipeline = RAGPipeline()
    retriever = RAGRetrievalService(pipeline)

    # Test questions about the RAG article
    test_questions = [
        "What are the main techniques for improving RAG accuracy?",
        "Why did early RAG implementations fail?",
        "How does the author achieve 98% accuracy in RAG systems?",
        "What is the role of chunking in RAG pipelines?",
        "How does embedding quality affect RAG performance?",
        "What are common RAG evaluation metrics?",
        "How to handle long documents in RAG systems?",
        "What preprocessing steps are important for RAG?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("-" * 60)

        try:
            # Get retrieval results
            results = retriever.retrieve(question, top_k=3)

            if not results:
                print("❌ No results found")
                continue

            # Display results with scoring
            for j, result in enumerate(results, 1):
                score = result.get('cosine_similarity', 0.0)
                page = result.get('page_number', '?')
                text = result.get('text', '')[:200] + '...' if len(result.get('text', '')) > 200 else result.get('text', '')

                print(f"📄 Result {j}: Score {score:.4f} | Page {page}")
                print(f"   {text}")
                print()

        except Exception as e:
            print(f"❌ Error processing question: {e}")
            continue


def test_embedding_quality():
    """Test embedding generation and quality."""

    print("\n🔗 Testing Embedding Quality")
    print("=" * 50)

    try:
        # Test different embedders
        factory = EmbedderFactory()

        # Test Gemma embedder
        print("🧠 Testing Gemma Embedder (768-dim)")
        gemma_embedder = factory.create_gemma()
        test_text = "RAG systems improve accuracy through better chunking and retrieval techniques."
        embedding = gemma_embedder.embed(test_text)
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
        print(".6f")

        # Test BGE-M3 embedder
        print("\n🧠 Testing BGE-M3 Embedder (1024-dim)")
        bge_embedder = factory.create_bge_m3()
        embedding_bge = bge_embedder.embed(test_text)
        print(f"   Embedding dimension: {len(embedding_bge)}")
        print(f"   Sample values: {embedding_bge[:5]}")
        print(".6f")

        # Test similarity between similar texts
        similar_text = "RAG pipelines achieve higher accuracy with improved chunking and retrieval methods."
        emb1 = gemma_embedder.embed(test_text)
        emb2 = gemma_embedder.embed(similar_text)

        # Calculate cosine similarity
        import numpy as np
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        similarity = dot_product / (norm1 * norm2)

        print(".4f")

    except Exception as e:
        print(f"❌ Embedding test failed: {e}")


def test_language_model():
    """Test language model integration."""

    print("\n🤖 Testing Language Model Integration")
    print("=" * 50)

    try:
        # Test basic connectivity with Gemini
        test_messages = [{"role": "user", "content": "Explain RAG in one sentence."}]
        print(f"🗣️  Test prompt: {test_messages[0]['content']}")

        response = call_gemini(test_messages)
        print(f"🤖 Response: {response[:200]}..." if len(response) > 200 else f"🤖 Response: {response}")

        # Test if API key is configured
        settings = resolve_gemini_settings()
        if not settings.get("api_key"):
            print("⚠️  Warning: Gemini API key not configured")

    except Exception as e:
        print(f"❌ Language model test failed: {e}")
        print("   This might be due to missing API keys or network issues")


def test_scoring_and_ranking():
    """Test scoring and ranking functionality."""

    print("\n📊 Testing Scoring and Ranking")
    print("=" * 50)

    try:
        # Initialize pipeline and retriever
        pipeline = RAGPipeline()
        retriever = RAGRetrievalService(pipeline)

        # Test question with expected high relevance
        question = "What techniques improve RAG accuracy to 98%?"

        results = retriever.retrieve(question, top_k=5)

        if results:
            print("📈 Top 5 results with scores:")
            for i, result in enumerate(results, 1):
                score = result.get('cosine_similarity', 0.0)
                page = result.get('page_number', '?')
                text_preview = result.get('text', '')[:100] + '...' if len(result.get('text', '')) > 100 else result.get('text', '')

                # Score interpretation
                if score > 0.9:
                    quality = "⭐ Excellent"
                elif score > 0.8:
                    quality = "✅ Good"
                elif score > 0.7:
                    quality = "⚠️  Fair"
                else:
                    quality = "❌ Poor"

                print(f"{i}. {quality} | Score: {score:.4f} | Page: {page}")
                print(f"   {text_preview}")
                print()

            # Test score distribution
            scores = [r.get('cosine_similarity', 0.0) for r in results]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)

            print("📊 Score Statistics:")
            print(".4f")
            print(".4f")
            print(".4f")
            print(".4f")

        else:
            print("❌ No results for scoring test")

    except Exception as e:
        print(f"❌ Scoring test failed: {e}")


def main():
    """Run all RAG system tests."""

    print("🚀 Starting RAG System Tests")
    print("Document: Medium RAG Techniques Article")
    print("=" * 60)

    # Test 1: Retrieval System
    test_retrieval_system()

    # Test 2: Embedding Quality
    test_embedding_quality()

    # Test 3: Language Model
    test_language_model()

    # Test 4: Scoring and Ranking
    test_scoring_and_ranking()

    print("\n✅ All tests completed!")
    print("Check the results above for system performance analysis.")


if __name__ == "__main__":
    main()