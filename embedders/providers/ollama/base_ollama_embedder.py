"""
Base Ollama Embedding Provider
==============================
Base class cho tất cả Ollama embedding providers.
"""

import logging
import requests
from typing import List, Optional, Dict, Any

from ..base_embedder import BaseEmbedder
from ...model.embedding_profile import EmbeddingProfile

logger = logging.getLogger(__name__)


from abc import ABC, abstractmethod

class BaseOllamaEmbedder(BaseEmbedder, ABC):
    """
    Base class cho tất cả Ollama embedding providers.
    Single Responsibility: Cung cấp common functionality cho Ollama embedders.
    """

    def __init__(self,
                 profile: Optional[EmbeddingProfile] = None,
                 base_url: str = "http://localhost:11434"):
        """
        Initialize base Ollama embedder.

        Args:
            profile: Embedding profile, nếu None sẽ dùng default
            base_url: Ollama server URL
        """
        if profile is None:
            # Subclass phải override để cung cấp default profile
            raise ValueError("profile must be provided for BaseOllamaEmbedder")

        super().__init__(profile)
        self.base_url = base_url.rstrip('/')
        self._test_connection()

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
            return []
    def _process_batch(self, requests) -> List:
        """
        Batch processing cho Ollama - gọi từng request.

        Args:
            requests: List of EmbedRequests

        Returns:
            List[EmbeddingResult]: Results
        """
        # Ollama thường không hỗ trợ batch embedding tốt, nên xử lý từng cái
        return [self.embed_single(req) for req in requests]

    @abstractmethod
    def embed_single(self, req) -> List[float]:
        """
        Abstract method để embed một request.
        Subclass phải implement method này.

        Args:
            req: EmbedRequest

        Returns:
            List[float]: Embedding vector
        """
        pass

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

    def embed(self, text: str) -> List[float]:
        """
        Tạo embedding cho một text string.

        Args:
            text: Input text để embed

        Returns:
            List[float]: Embedding vector
        """
        return self._generate_embedding(text)

    def test_connection(self) -> bool:
        """
        Test connection tới Ollama server.

        Returns:
            bool: True nếu connection thành công
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False