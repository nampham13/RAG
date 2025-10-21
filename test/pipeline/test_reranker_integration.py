"""
Unit tests for reranker integration in RAG pipeline.
Tests the integration of BM25 and BGE rerankers with the retrieval system.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from pipeline.rag_pipeline import RAGPipeline
from pipeline.retriever import Retriever
from rerankers.reranker_factory import RerankerFactory
from rerankers.i_reranker import IReranker
from embedders.providers.ollama import OllamaModelType


class TestRerankerFactory:
    """Test RerankerFactory creation methods."""
    
    def test_create_bm25_default(self):
        """Test creating BM25 reranker with default settings."""
        reranker = RerankerFactory.create_bm25()
        assert reranker is not None
        assert hasattr(reranker, 'rerank')
        
    def test_create_bm25_with_kwargs(self):
        """Test creating BM25 reranker with custom parameters."""
        bm25_kwargs = {'k1': 1.2, 'b': 0.75}
        reranker = RerankerFactory.create_bm25(bm25_kwargs=bm25_kwargs)
        assert reranker is not None
        assert reranker.bm25_kwargs == bm25_kwargs
        
    def test_create_from_config_bm25(self):
        """Test creating reranker from config string."""
        reranker = RerankerFactory.create_from_config("bm25")
        assert reranker is not None
        
    def test_create_from_config_invalid(self):
        """Test creating reranker with invalid type."""
        reranker = RerankerFactory.create_from_config("invalid_type")
        assert reranker is None


class TestRetrieverWithReranker:
    """Test Retriever class with reranker integration."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = Mock()
        embedder.embed.return_value = [0.1] * 768
        embedder.dimension = 768
        return embedder
    
    @pytest.fixture
    def mock_reranker(self):
        """Create mock reranker."""
        reranker = Mock(spec=IReranker)
        
        def mock_rerank(query, candidates, top_k=5):
            # Simple mock: reverse order and add rerank scores
            results = []
            for i, cand in enumerate(candidates[:top_k]):
                result = cand.copy()
                result['score'] = 1.0 - (i * 0.1)  # Decreasing scores
                results.append(result)
            return results
        
        reranker.rerank = Mock(side_effect=mock_rerank)
        return reranker
    
    def test_retriever_init_without_reranker(self, mock_embedder):
        """Test Retriever initialization without reranker."""
        retriever = Retriever(mock_embedder)
        assert retriever.embedder == mock_embedder
        assert retriever.reranker is None
        
    def test_retriever_init_with_reranker(self, mock_embedder, mock_reranker):
        """Test Retriever initialization with reranker."""
        retriever = Retriever(mock_embedder, reranker=mock_reranker)
        assert retriever.embedder == mock_embedder
        assert retriever.reranker == mock_reranker
    
    @patch('faiss.read_index')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_search_without_reranking(self, mock_pickle_load, mock_open, mock_faiss_read, 
                                     mock_embedder, mock_reranker):
        """Test search without reranking enabled."""
        # Setup mocks
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # similarities
            np.array([[0, 1, 2]])  # indices
        )
        mock_faiss_read.return_value = mock_index
        
        mock_metadata = {
            0: {'chunk_id': 'c1', 'text': 'text 1', 'page_number': 1},
            1: {'chunk_id': 'c2', 'text': 'text 2', 'page_number': 2},
            2: {'chunk_id': 'c3', 'text': 'text 3', 'page_number': 3}
        }
        mock_pickle_load.return_value = mock_metadata
        
        # Create retriever and search
        retriever = Retriever(mock_embedder, reranker=mock_reranker)
        results = retriever.search_similar(
            Path("test.faiss"),
            Path("test.pkl"),
            "query",
            top_k=3,
            use_reranking=False
        )
        
        # Verify results
        assert len(results) == 3
        assert results[0]['similarity_score'] == 0.9
        assert 'rerank_score' not in results[0]
        
        # Verify reranker was not called
        mock_reranker.rerank.assert_not_called()
    
    @patch('faiss.read_index')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_search_with_reranking(self, mock_pickle_load, mock_open, mock_faiss_read,
                                   mock_embedder, mock_reranker):
        """Test search with reranking enabled."""
        # Setup mocks
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7, 0.6, 0.5]]),  # similarities
            np.array([[0, 1, 2, 3, 4]])  # indices
        )
        mock_faiss_read.return_value = mock_index
        
        mock_metadata = {
            i: {'chunk_id': f'c{i}', 'text': f'text {i}', 'page_number': i}
            for i in range(5)
        }
        mock_pickle_load.return_value = mock_metadata
        
        # Create retriever and search
        retriever = Retriever(mock_embedder, reranker=mock_reranker)
        results = retriever.search_similar(
            Path("test.faiss"),
            Path("test.pkl"),
            "query",
            top_k=3,
            use_reranking=True,
            reranking_top_k=5
        )
        
        # Verify results
        assert len(results) == 3
        assert 'rerank_score' in results[0]
        
        # Verify reranker was called
        mock_reranker.rerank.assert_called_once()


class TestRAGPipelineWithReranker:
    """Test RAGPipeline with reranker integration."""
    
    @patch('pipeline.rag_pipeline.OllamaModelSwitcher')
    @patch('pipeline.rag_pipeline.PDFLoader')
    def test_pipeline_init_no_reranker(self, mock_loader, mock_switcher):
        """Test pipeline initialization without reranker."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.profile.model_id = "test_model"
        mock_embedder.dimension = 768
        mock_embedder.test_connection.return_value = True
        
        mock_switcher_instance = Mock()
        mock_switcher_instance.switch_to_gemma.return_value = mock_embedder
        mock_switcher.return_value = mock_switcher_instance
        
        mock_loader.create_default.return_value = Mock()
        
        # Create pipeline
        pipeline = RAGPipeline(
            output_dir="/tmp/test_data",
            reranker_type=None
        )
        
        assert pipeline.reranker is None
        assert pipeline.reranker_type is None
    
    @patch('pipeline.rag_pipeline.RerankerFactory')
    @patch('pipeline.rag_pipeline.OllamaModelSwitcher')
    @patch('pipeline.rag_pipeline.PDFLoader')
    def test_pipeline_init_with_bm25_reranker(self, mock_loader, mock_switcher, mock_reranker_factory):
        """Test pipeline initialization with BM25 reranker."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.profile.model_id = "test_model"
        mock_embedder.dimension = 768
        mock_embedder.test_connection.return_value = True
        
        mock_switcher_instance = Mock()
        mock_switcher_instance.switch_to_gemma.return_value = mock_embedder
        mock_switcher.return_value = mock_switcher_instance
        
        mock_loader.create_default.return_value = Mock()
        
        mock_reranker = Mock(spec=IReranker)
        mock_reranker_factory.create_from_config.return_value = mock_reranker
        
        # Create pipeline
        pipeline = RAGPipeline(
            output_dir="/tmp/test_data",
            reranker_type="bm25"
        )
        
        assert pipeline.reranker is not None
        assert pipeline.reranker_type == "bm25"
        mock_reranker_factory.create_from_config.assert_called_once_with("bm25")
    
    @patch('pipeline.rag_pipeline.RerankerFactory')
    @patch('pipeline.rag_pipeline.OllamaModelSwitcher')
    @patch('pipeline.rag_pipeline.PDFLoader')
    def test_pipeline_init_with_reranker_kwargs(self, mock_loader, mock_switcher, mock_reranker_factory):
        """Test pipeline initialization with custom reranker kwargs."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.profile.model_id = "test_model"
        mock_embedder.dimension = 768
        mock_embedder.test_connection.return_value = True
        
        mock_switcher_instance = Mock()
        mock_switcher_instance.switch_to_gemma.return_value = mock_embedder
        mock_switcher.return_value = mock_switcher_instance
        
        mock_loader.create_default.return_value = Mock()
        
        mock_reranker = Mock(spec=IReranker)
        mock_reranker_factory.create_from_config.return_value = mock_reranker
        
        # Create pipeline with custom kwargs
        custom_kwargs = {'k1': 1.2, 'b': 0.5}
        pipeline = RAGPipeline(
            output_dir="/tmp/test_data",
            reranker_type="bm25",
            reranker_kwargs=custom_kwargs
        )
        
        assert pipeline.reranker is not None
        mock_reranker_factory.create_from_config.assert_called_once_with("bm25", **custom_kwargs)
    
    @patch('pipeline.rag_pipeline.OllamaModelSwitcher')
    @patch('pipeline.rag_pipeline.PDFLoader')
    def test_pipeline_get_info_with_reranker(self, mock_loader, mock_switcher):
        """Test pipeline get_info includes reranker information."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.profile.model_id = "test_model"
        mock_embedder.dimension = 768
        mock_embedder.test_connection.return_value = True
        
        mock_switcher_instance = Mock()
        mock_switcher_instance.switch_to_gemma.return_value = mock_embedder
        mock_switcher.return_value = mock_switcher_instance
        
        mock_loader.create_default.return_value = Mock()
        
        # Create pipeline
        pipeline = RAGPipeline(
            output_dir="/tmp/test_data",
            reranker_type="bm25"
        )
        
        info = pipeline.get_info()
        assert 'reranker' in info
        assert info['reranker'] == "bm25"


class TestBM25RerankerIntegration:
    """Integration tests for BM25 reranker."""
    
    def test_bm25_rerank_basic(self):
        """Test basic BM25 reranking functionality."""
        reranker = RerankerFactory.create_bm25()
        
        # Sample candidates
        candidates = [
            {'id': '1', 'text': 'machine learning algorithms for data science'},
            {'id': '2', 'text': 'deep learning neural networks'},
            {'id': '3', 'text': 'machine learning applications'},
            {'id': '4', 'text': 'statistical methods for analysis'},
        ]
        
        query = "machine learning"
        
        # Rerank
        results = reranker.rerank(query, candidates, top_k=3)
        
        # Verify
        assert len(results) == 3
        assert all('score' in r for r in results)
        assert all('text' in r for r in results)
        
        # Results should be sorted by score (descending)
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
