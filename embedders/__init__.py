"""
Embedders Package
=================
Provider-agnostic embedding orchestration với integrated chunking.
"""

# Import core components
from .i_embedder import IEmbedder
from .model.embedding_profile import EmbeddingProfile

# Conditional imports for components with heavy dependencies
try:
    from .embedder_factory import EmbedderFactory
    from .embedder_type import EmbedderType
    _factory_available = True
except ImportError:
    _factory_available = False
    EmbedderFactory = None
    EmbedderType = None

try:
    from .chunk_and_embed_pipeline import ChunkAndEmbedPipeline
    _pipeline_available = True
except ImportError:
    _pipeline_available = False
    ChunkAndEmbedPipeline = None

# Import providers conditionally
try:
    from .providers import OllamaEmbedder, GemmaEmbedder, BGE3Embedder
    _providers_available = True
except ImportError:
    _providers_available = False
    OllamaEmbedder = None
    GemmaEmbedder = None
    BGE3Embedder = None

__all__ = ['IEmbedder', 'EmbeddingProfile']

if _factory_available:
    __all__.extend(['EmbedderFactory', 'EmbedderType'])
if _pipeline_available:
    __all__.append('ChunkAndEmbedPipeline')
if _providers_available:
    __all__.extend(['OllamaEmbedder', 'GemmaEmbedder', 'BGE3Embedder'])