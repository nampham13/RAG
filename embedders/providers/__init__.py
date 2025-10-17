"""
Embedders Providers Package
==========================
Concrete implementations của embedders.
"""

# Import Ollama providers
try:
    from .ollama_embedder import OllamaEmbedder
    _ollama_available = True
except ImportError:
    _ollama_available = False
    OllamaEmbedder = None

# Import new Ollama specialized embedders
try:
    from .ollama.gemma_embedder import GemmaEmbedder
    from .ollama.bge3_embedder import BGE3Embedder
    _ollama_specialized_available = True
except ImportError:
    _ollama_specialized_available = False
    GemmaEmbedder = None
    BGE3Embedder = None

__all__ = []

if _ollama_available:
    __all__.append('OllamaEmbedder')
if _ollama_specialized_available:
    __all__.extend(['GemmaEmbedder', 'BGE3Embedder'])