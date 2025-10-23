"""
Rerankers Providers Package
==========================
Concrete implementations của rerankers.
"""

# Import BGE provider
try:
    from .bge_reranker import BGEReranker
    _bge_available = True
except ImportError:
    _bge_available = False
    BGEReranker = None

# Import Jina provider
try:
    from .jina_reranker import JinaReranker
    _jina_available = True
except ImportError:
    _jina_available = False
    JinaReranker = None

__all__ = []

if _bge_available:
    __all__.append('BGEReranker')
if _jina_available:
    __all__.append('JinaReranker')