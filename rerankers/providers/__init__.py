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

try:
    from .L6_reranker import L6Reranker
    _l6_available = True
except ImportError:
    _l6_available = False
    L6Reranker = None

__all__ = []

if _bge_available:
    __all__.append('BGEReranker')
if _jina_available:
    __all__.append('JinaReranker')
if _l6_available:
    __all__.append('L6Reranker')