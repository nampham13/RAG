"""
Ollama Embedding Provider
========================
Implementation cho Ollama embedding models.
"""

import logging
import requests
from typing import List, Optional, Dict, Any

from .base_embedder import BaseEmbedder
from ..model.embedding_profile import EmbeddingProfile

logger = logging.getLogger(__name__)


class OllamaEmbedder(BaseEmbedder):
    """
    Generic Ollama embedding provider.
    Single Responsibility: Tạo embeddings sử dụng Ollama API.
    
    Note: Đây là generic class. Sử dụng GemmaEmbedder hoặc BGE3Embedder 
    cho các models cụ thể với config có sẵn.
    """
    
    # Default config cho generic Ollama embedder
    DEFAULT_MODEL_ID = "nomic-embed-text"
    DEFAULT_DIMENSION = 768
    DEFAULT_MAX_TOKENS = 8192

    def __init__(self,
                 profile: EmbeddingProfile,
                 base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama embedder.

        Args:
            profile: Embedding profile (REQUIRED)
            base_url: Ollama server URL
        """
        super().__init__(profile)
        self.base_url = base_url.rstrip('/')
        self._test_connection()

    @classmethod
    def create_default(cls, base_url: str = "http://localhost:11434") -> 'OllamaEmbedder':
        """
        Factory method để tạo Ollama embedder với config mặc định (nomic).

        Args:
            base_url: Ollama server URL

        Returns:
            OllamaEmbedder: Configured Ollama embedder
        """
        profile = EmbeddingProfile(
            model_id=cls.DEFAULT_MODEL_ID,
            provider="ollama",
            max_tokens=cls.DEFAULT_MAX_TOKENS,
            dimension=cls.DEFAULT_DIMENSION,
            normalize=True
        )
        return cls(profile=profile, base_url=base_url)

    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Ollama server at {self.base_url}")
            else:
                logger.warning(f"Ollama server responded with status {response.status_code}")
        except Exception as e:
            logger.warning(f"Cannot connect to Ollama server: {e}")

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Ollama API.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector
        """
        try:
            payload = {
                "model": self.profile.model_id,
                "prompt": text
            }

            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")

            result = response.json()
            embedding = result.get("embedding", [])

            if not embedding:
                raise RuntimeError("No embedding returned from Ollama")

            return embedding

        except Exception as e:
            logger.error(f"Error generating Ollama embedding: {e}")
            raise

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding cho text.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        return self._generate_embedding(text)
    
    def test_connection(self) -> bool:
        """
        Test connection to Ollama server.
        
        Returns:
            bool: True if connected successfully
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        """
        Kiểm tra xem Ollama embedder có available không.

        Returns:
            bool: True if available
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                return self.profile.model_id in model_names
            return False
        except:
            return False

    def get_available_models(self) -> List[str]:
        """
        Lấy danh sách models available từ Ollama.

        Returns:
            List[str]: Model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name") for m in models]
            return []
        except:
            return []

    def switch_model(self, model_name: str) -> bool:
        """
        Đổi model nhanh chóng.

        Args:
            model_name: Tên model mới

        Returns:
            bool: True nếu thành công
        """
        if model_name in self.get_available_models():
            self.profile.model_id = model_name
            logger.info(f"Switched to Ollama model: {model_name}")
            return True
        else:
            logger.error(f"Model {model_name} not available in Ollama")
            return False

    @property
    def dimension(self) -> int:
        """
        Lấy dimension của embedding.

        Returns:
            int: Embedding dimension
        """
        if self.profile.dimension:
            return self.profile.dimension

        # Known Ollama model dimensions
        model_dims = {
            "nomic-embed-text": 768,
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "paraphrase-MiniLM-L3-v2": 384,
            "paraphrase-MiniLM-L6-v2": 384,
            "distiluse-base-multilingual-cased-v1": 512,
            "distiluse-base-multilingual-cased-v2": 512,
            "bge-m3:latest": 1024,
            "embeddinggemma:latest": 2048,  # Conservative estimate
        }

        return model_dims.get(self.profile.model_id, 768)

    def estimate_tokens(self, text: str) -> int:
        """
        Better token estimation cho Ollama models.

        Args:
            text: Text to estimate

        Returns:
            int: Estimated token count
        """
        # Ollama thường dùng tokenizer tương tự như các model gốc
        # Ước lượng đơn giản: ~4 chars per token
        return len(text) // 4