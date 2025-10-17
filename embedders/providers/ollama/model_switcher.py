"""
Ollama Model Switcher
====================
Utility để switch giữa các Ollama models một cách dễ dàng.
"""

import logging
from typing import Optional
from enum import Enum

from .gemma_embedder import GemmaEmbedder
from .bge3_embedder import BGE3Embedder
from ..base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class OllamaModelType(Enum):
    """Enum cho các Ollama models hỗ trợ."""
    GEMMA = "embeddinggemma:latest"
    BGE_M3 = "bge-m3:latest"


class OllamaModelSwitcher:
    """
    Model switcher cho Ollama embedders.
    Single Responsibility: Quản lý việc switch giữa các Ollama models.
    
    Example:
        >>> switcher = OllamaModelSwitcher()
        >>> gemma = switcher.get_embedder(OllamaModelType.GEMMA)
        >>> bge3 = switcher.get_embedder(OllamaModelType.BGE_M3)
        >>> switcher.switch_to_gemma()
        >>> current = switcher.current_embedder
    """

    def __init__(self, 
                 default_model: OllamaModelType = OllamaModelType.BGE_M3,
                 base_url: str = "http://localhost:11434"):
        """
        Initialize model switcher.

        Args:
            default_model: Model mặc định khi khởi tạo
            base_url: Ollama server URL
        """
        self.base_url = base_url
        self._embedders = {}  # Cache embedders
        self._current_model = default_model
        self._initialize_embedders()

    def _initialize_embedders(self) -> None:
        """Initialize tất cả embedders và cache lại."""
        self._embedders[OllamaModelType.GEMMA] = GemmaEmbedder.create_default(self.base_url)
        self._embedders[OllamaModelType.BGE_M3] = BGE3Embedder.create_default(self.base_url)
        logger.info(f"Initialized Ollama embedders: {list(OllamaModelType)}")

    @property
    def current_embedder(self) -> BaseEmbedder:
        """
        Lấy embedder hiện tại.

        Returns:
            BaseEmbedder: Current embedder instance
        """
        return self._embedders[self._current_model]

    @property
    def current_model_name(self) -> str:
        """
        Lấy tên model hiện tại.

        Returns:
            str: Model name
        """
        return self._current_model.value

    def get_embedder(self, model_type: OllamaModelType) -> BaseEmbedder:
        """
        Lấy embedder theo model type.

        Args:
            model_type: Loại model cần lấy

        Returns:
            BaseEmbedder: Embedder instance
        """
        return self._embedders[model_type]

    def switch_to(self, model_type: OllamaModelType) -> BaseEmbedder:
        """
        Switch sang model type khác.

        Args:
            model_type: Model type muốn switch tới

        Returns:
            BaseEmbedder: Embedder instance sau khi switch
        """
        old_model = self._current_model
        self._current_model = model_type
        logger.info(f"Switched from {old_model.value} to {model_type.value}")
        return self.current_embedder

    def switch_to_gemma(self) -> GemmaEmbedder:
        """
        Switch sang Gemma model.

        Returns:
            GemmaEmbedder: Gemma embedder instance
        """
        return self.switch_to(OllamaModelType.GEMMA)  # type: ignore

        # Hoặc ép kiểu rõ ràng:
        # return self.switch_to(OllamaModelType.GEMMA) as GemmaEmbedder

    def switch_to_bge_m3(self) -> BGE3Embedder:
        """
        Switch sang BGE-M3 model.

        Returns:
            BGE3Embedder: BGE-M3 embedder instance
        """
        return self.switch_to(OllamaModelType.BGE_M3)  # type: ignore

    def get_available_models(self) -> list:
        """
        Lấy danh sách models có sẵn.

        Returns:
            list: List của OllamaModelType
        """
        return list(OllamaModelType)

    def embed_with_model(self, text: str, model_type: OllamaModelType) -> list:
        """
        Embed text với model cụ thể mà không switch.

        Args:
            text: Text cần embed
            model_type: Model type để sử dụng

        Returns:
            list: Embedding vector
        """
        embedder = self.get_embedder(model_type)
        return embedder.embed(text)

    def compare_models(self, text: str) -> dict:
        """
        So sánh embedding từ cả 2 models.

        Args:
            text: Text để embed

        Returns:
            dict: Dictionary với embeddings từ mỗi model
        """
        results = {}
        for model_type in OllamaModelType:
            embedder = self.get_embedder(model_type)
            embedding = embedder.embed(text)
            results[model_type.value] = {
                "embedding": embedding,
                "dimension": len(embedding),
                "model": model_type.value
            }
        return results

    def __repr__(self) -> str:
        return f"OllamaModelSwitcher(current={self.current_model_name}, available={len(self._embedders)})"