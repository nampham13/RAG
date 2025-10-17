import pytest
from unittest.mock import patch, MagicMock, Mock
import requests

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embedders.embedder_factory import EmbedderFactory
from embedders.embedder_type import EmbedderType
from embedders.model.embedding_profile import EmbeddingProfile
from embedders.providers.ollama.gemma_embedder import GemmaEmbedder
from embedders.providers.ollama.bge3_embedder import BGE3Embedder
from embedders.providers.ollama_embedder import OllamaEmbedder


class TestEmbedderFactory:
    """Test chi tiết cho EmbedderFactory"""

    def test_initialization(self):
        """Test constructor"""
        factory = EmbedderFactory()
        assert isinstance(factory._registry, dict)

    def test_create_gemma(self):
        """Test create_gemma method"""
        factory = EmbedderFactory()
        embedder = factory.create_gemma()

        assert isinstance(embedder, GemmaEmbedder)
        assert embedder.base_url == "http://localhost:11434"

    def test_create_bge_m3(self):
        """Test create_bge_m3 method"""
        factory = EmbedderFactory()
        embedder = factory.create_bge_m3()

        assert isinstance(embedder, BGE3Embedder)
        assert embedder.base_url == "http://localhost:11434"

    def test_create_ollama_nomic(self):
        """Test create_ollama_nomic method"""
        factory = EmbedderFactory()
        embedder = factory.create_ollama_nomic()

        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.base_url == "http://localhost:11434"

    def test_create_with_custom_base_url(self):
        """Test create methods với custom base_url"""
        factory = EmbedderFactory()
        custom_url = "http://custom:8080"

        gemma = factory.create_gemma(base_url=custom_url)
        bge3 = factory.create_bge_m3(base_url=custom_url)
        nomic = factory.create_ollama_nomic(base_url=custom_url)

        assert gemma.base_url == custom_url
        assert bge3.base_url == custom_url
        assert nomic.base_url == custom_url


class TestGemmaEmbedder:
    """Test chi tiết cho GemmaEmbedder"""

    @patch('requests.post')
    def test_initialization(self, mock_post):
        """Test constructor"""
        embedder = GemmaEmbedder.create_default()
        assert embedder.profile.model_id == "embeddinggemma:latest"
        assert embedder.profile.dimension == 768
        assert embedder.profile.max_tokens == 2048

    @patch('embedders.providers.ollama.base_ollama_embedder.requests.post')
    def test_embed_success(self, mock_post):
        """Test embed method thành công"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_post.return_value = mock_response

        embedder = GemmaEmbedder.create_default()
        result = embedder.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 768
        mock_post.assert_called_once()

    @patch('embedders.providers.ollama.base_ollama_embedder.requests.post')
    def test_embed_connection_error(self, mock_post):
        """Test embed method với connection error"""
        mock_post.side_effect = requests.exceptions.ConnectionError()

        embedder = GemmaEmbedder.create_default()
        result = embedder.embed("test text")

        # The method catches exceptions and returns empty list
        assert result == []

    @patch('embedders.providers.ollama.base_ollama_embedder.requests.get')
    def test_test_connection_success(self, mock_get):
        """Test test_connection method thành công"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        embedder = GemmaEmbedder.create_default()
        result = embedder.test_connection()

        assert result is True

    @patch('embedders.providers.ollama.base_ollama_embedder.requests.get')
    def test_test_connection_failure(self, mock_get):
        """Test test_connection method thất bại"""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        embedder = GemmaEmbedder.create_default()
        result = embedder.test_connection()

        assert result is False


class TestBGE3Embedder:
    """Test chi tiết cho BGE3Embedder"""

    @patch('requests.post')
    def test_initialization(self, mock_post):
        """Test constructor"""
        embedder = BGE3Embedder.create_default()
        assert embedder.profile.model_id == "bge-m3:latest"
        assert embedder.profile.dimension == 1024
        assert embedder.profile.max_tokens == 8192

    @patch('embedders.providers.ollama.base_ollama_embedder.requests.post')
    def test_embed_success(self, mock_post):
        """Test embed method thành công"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 1024}
        mock_post.return_value = mock_response

        embedder = BGE3Embedder.create_default()
        result = embedder.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 1024
        mock_post.assert_called_once()

    @patch('embedders.providers.ollama.base_ollama_embedder.requests.post')
    def test_embed_invalid_response(self, mock_post):
        """Test embed method với invalid response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Model not found"}
        mock_post.return_value = mock_response

        embedder = BGE3Embedder.create_default()
        result = embedder.embed("test text")

        # The method returns empty list when no embedding is found
        assert result == []

    @patch('embedders.providers.ollama.base_ollama_embedder.requests.get')
    def test_test_connection_success(self, mock_get):
        """Test test_connection method thành công"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        embedder = BGE3Embedder.create_default()
        result = embedder.test_connection()

        assert result is True


class TestOllamaEmbedder:
    """Test chi tiết cho base OllamaEmbedder"""

    @patch('embedders.providers.ollama_embedder.requests.post')
    def test_create_default(self, mock_post):
        """Test create_default factory method"""
        embedder = OllamaEmbedder.create_default()
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.base_url == "http://localhost:11434"

    @patch('embedders.providers.ollama_embedder.requests.post')
    def test_embed_batch(self, mock_post):
        """Test embed_batch method"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_post.return_value = mock_response

        embedder = OllamaEmbedder.create_default()
        texts = ["text1", "text2", "text3"]
        results = embedder.embed_batch(texts)

        assert len(results) == 3
        assert all(len(emb) == 768 for emb in results)
        assert mock_post.call_count == 3

    @patch('embedders.providers.ollama_embedder.requests.post')
    def test_embed_batch_partial_failure(self, mock_post):
        """Test embed_batch với partial failure"""
        def side_effect(*args, **kwargs):
            # Check the request data for the text content
            request_data = kwargs.get('json', {})
            if request_data.get('prompt') == "text2":
                raise requests.exceptions.ConnectionError()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embedding": [0.1] * 768}
            return mock_response

        mock_post.side_effect = side_effect

        embedder = OllamaEmbedder.create_default()
        texts = ["text1", "text2", "text3"]

        with pytest.raises(requests.exceptions.ConnectionError):
            embedder.embed_batch(texts)