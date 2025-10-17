"""
Embedding Factory
=================
Creates embedder instances from configuration.
"""

from typing import Dict, Optional, Any

from .embedder_type import EmbedderType
from .model.embedding_profile import EmbeddingProfile
from .i_embedder import IEmbedder
from .providers.ollama_embedder import OllamaEmbedder
from .providers.ollama.gemma_embedder import GemmaEmbedder
from .providers.ollama.bge3_embedder import BGE3Embedder


class EmbedderFactory:
    """
    Factory to build embedders based on profile and type.
    Single Responsibility: centralize embedder instantiation logic.
    """

    def __init__(self, registry: Dict[EmbedderType, type] | None = None):
        self._registry = registry or {}

    def create(self, embedder_type: EmbedderType, profile: EmbeddingProfile, **kwargs: Any) -> IEmbedder:
        """Create the requested embedder."""
        if embedder_type == EmbedderType.OLLAMA:
            # Default to GemmaEmbedder for OLLAMA type
            return GemmaEmbedder(
                profile=profile,
                base_url=kwargs.get('base_url', 'http://localhost:11434')
            ) # pyright: ignore[reportAbstractUsage]
        raise ValueError(f"Unsupported embedder type: {embedder_type!r}")

    def create_ollama_nomic(self, base_url: str = "http://localhost:11434", **kwargs: Any) -> OllamaEmbedder:
        """
        Factory method cho Ollama Nomic embedder.
        
        Note: Config nằm trong OllamaEmbedder class.

        Args:
            base_url: Ollama server URL
            **kwargs: Additional arguments

        Returns:
            OllamaEmbedder: Ollama Nomic embedder
        """
        return OllamaEmbedder.create_default(base_url=base_url)

    def create_bge_m3(self, base_url: str = "http://localhost:11434", **kwargs: Any) -> BGE3Embedder:
        """
        Factory method cho Ollama BGE-M3 embedder.
        
        Note: Config nằm trong BGE3Embedder class (MODEL_ID, DIMENSION, MAX_TOKENS).

        Args:
            base_url: Ollama server URL
            **kwargs: Additional arguments

        Returns:
            BGE3Embedder: Ollama BGE-M3 embedder
        """
        return BGE3Embedder.create_default(base_url=base_url)
    
    def create_gemma(self, base_url: str = "http://localhost:11434", **kwargs: Any) -> GemmaEmbedder:
        """
        Factory method cho Ollama Embedding Gemma embedder.
        
        Note: Config nằm trong GemmaEmbedder class (MODEL_ID, DIMENSION, MAX_TOKENS).

        Args:
            base_url: Ollama server URL
            **kwargs: Additional arguments

        Returns:
            GemmaEmbedder: Ollama Embedding Gemma embedder
        """
        return GemmaEmbedder.create_default(base_url=base_url)