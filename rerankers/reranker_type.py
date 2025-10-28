"""
Reranker Type Enum
==================
Enum cho các loại reranker providers.
"""

from enum import Enum


class RerankerType(Enum):
    """Enum cho các reranking providers."""
    BGE = "bge"
    JINA = "jina"