"""
BGE-M3 Embedding Provider
=========================
Implementation cho BGE-M3 embedding model từ Ollama.
"""

from typing import Optional, List

from .base_ollama_embedder import BaseOllamaEmbedder
from ...model.embedding_profile import EmbeddingProfile


class BGE3Embedder(BaseOllamaEmbedder):
    """
    BGE-M3 embedding provider cho Ollama.
    Single Responsibility: Tạo embeddings sử dụng BGE-M3 model.
    
    Config:
        - Model: bge-m3:latest
        - Dimension: 1024
        - Max tokens: 8192
        - Provider: ollama
    """
    
    # Class-level constants (config nằm trong class)
    MODEL_ID = "bge-m3:latest"
    DIMENSION = 1024
    MAX_TOKENS = 8192
    PROVIDER = "ollama"

    def __init__(self,
                 profile: Optional[EmbeddingProfile] = None,
                 base_url: str = "http://localhost:11434"):
        """
        Initialize BGE-M3 embedder.

        Args:
            profile: Embedding profile, nếu None sẽ tạo từ class constants
            base_url: Ollama server URL
        """
        if profile is None:
            profile = self._create_profile()

        super().__init__(profile, base_url)

    @classmethod
    def _create_profile(cls) -> EmbeddingProfile:
        """
        Tạo EmbeddingProfile từ class constants.
        Config nằm TRONG class, không phụ thuộc EmbeddingProfile.
        """
        return EmbeddingProfile(
            model_id=cls.MODEL_ID,
            provider=cls.PROVIDER,
            max_tokens=cls.MAX_TOKENS,
            dimension=cls.DIMENSION,
            normalize=True
        )

    @property
    def dimension(self) -> int:
        """
        Lấy dimension của BGE-M3 embedding.

        Returns:
            int: Embedding dimension (1024)
        """
        return self.DIMENSION

    @classmethod
    def create_default(cls, base_url: str = "http://localhost:11434") -> 'BGE3Embedder':
        """
        Factory method để tạo BGE-M3 embedder với cấu hình mặc định.

        Args:
            base_url: Ollama server URL

        Returns:
            BGE3Embedder: Configured BGE-M3 embedder
        """
        return cls(profile=None, base_url=base_url)
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding cho text sử dụng BGE-M3 model.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        return self._generate_embedding(text)
    
    def embed_single(self, req) -> List[float]:
        """
        Embed a single EmbedRequest.
        
        Args:
            req: EmbedRequest object
            
        Returns:
            List[float]: Embedding vector
        """
        return self.embed(req.text)
    
    def get_config(self) -> dict:
        """
        Lấy config của BGE-M3 embedder.
        
        Returns:
            dict: Configuration dictionary
        """
        return {
            "model_id": self.MODEL_ID,
            "provider": self.PROVIDER,
            "max_tokens": self.MAX_TOKENS,
            "dimension": self.DIMENSION,
            "normalize": True,
            "base_url": self.base_url
        }