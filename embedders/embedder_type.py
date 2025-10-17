"""
Embedder Type Enum
==================
Enum cho các loại embedder providers.
"""

from enum import Enum


class EmbedderType(Enum):
    """Enum cho các embedding providers."""
    OLLAMA = "ollama"