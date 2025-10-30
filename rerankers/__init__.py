"""
Rerankers Package
=================
Provider-agnostic reranking orchestration.
"""

# Import core components
from .i_reranker import IReranker
from .model.reranker_profile import RerankerProfile

# Conditional imports for components with heavy dependencies
try:
    from .reranker_factory import RerankerFactory
    from .reranker_type import RerankerType
    _factory_available = True
except ImportError:
    _factory_available = False
    RerankerFactory = None
    RerankerType = None

# Import providers conditionally
try:
    from .providers.bge_reranker import BGEReranker
    _bge_available = True
except ImportError:
    _bge_available = False
    BGEReranker = None

try:
    from .providers.jina_reranker import JinaReranker
    _jina_available = True
except ImportError:
    _jina_available = False
    JinaReranker = None

try:
    from .providers.L6_reranker import L6Reranker
    _l6_available = True
except ImportError:
    _l6_available = False
    L6Reranker = None

__all__ = ['IReranker', 'RerankerProfile']

if _factory_available:
    __all__.extend(['RerankerFactory', 'RerankerType'])
if _bge_available:
    __all__.append('BGEReranker')
if _jina_available:
    __all__.append('JinaReranker')
if _l6_available:
    __all__.append('L6Reranker')