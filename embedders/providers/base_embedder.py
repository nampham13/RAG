"""
Base Embedder
=============
Base class cho tất cả embedding providers.
"""

import logging
from typing import List, Optional
from abc import ABC

from ..i_embedder import IEmbedder
from ..model.embedding_profile import EmbeddingProfile

logger = logging.getLogger(__name__)


class BaseEmbedder(IEmbedder, ABC):
    """
    Base class cho tất cả embedding providers.
    Single Responsibility: Cung cấp common functionality cho embedders.
    """

    def __init__(self, profile: EmbeddingProfile):
        """
        Initialize base embedder.

        Args:
            profile: Embedding profile configuration
        """
        self.profile = profile
        self._validate_profile()

    def _validate_profile(self):
        """Validate embedding profile"""
        if not self.profile.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.profile.provider:
            raise ValueError("provider cannot be empty")

    @property
    def dimension(self) -> int:
        """
        Lấy dimension của embedding vectors.

        Returns:
            int: Embedding dimension từ profile hoặc default
        """
        return self.profile.dimension or 768  # Default fallback

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Default implementation: embed từng text một.

        Args:
            texts: List của input texts

        Returns:
            List[List[float]]: List của embedding vectors
        """
        return [self.embed(text) for text in texts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.profile.model_id}, dim={self.dimension})"