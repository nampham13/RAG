"""
Embedder Interface
==================
Abstract interface cho tất cả embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List


class IEmbedder(ABC):
    """
    Interface cho embedding providers.
    Single Responsibility: Define contract cho embedding operations.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Lấy dimension của embedding vectors.

        Returns:
            int: Embedding dimension
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Tạo embedding cho một text string.

        Args:
            text: Input text để embed

        Returns:
            List[float]: Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embeddings cho một batch của texts.

        Args:
            texts: List của input texts

        Returns:
            List[List[float]]: List của embedding vectors
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connection tới embedding service.

        Returns:
            bool: True nếu connection thành công
        """
        pass