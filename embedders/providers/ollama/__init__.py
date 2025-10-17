"""
Ollama Embedding Providers
==========================
Collection of Ollama-based embedding providers.
"""

from .base_ollama_embedder import BaseOllamaEmbedder
from .gemma_embedder import GemmaEmbedder
from .bge3_embedder import BGE3Embedder
from .model_switcher import OllamaModelSwitcher, OllamaModelType

__all__ = [
    "BaseOllamaEmbedder",
    "GemmaEmbedder",
    "BGE3Embedder",
    "OllamaModelSwitcher",
    "OllamaModelType"
]