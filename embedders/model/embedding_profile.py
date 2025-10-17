"""
Embedding Configuration Profile
==============================
Cấu hình cho embedding models và providers.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Local base class for embedding profile models
class LoaderBaseModel:
    pass


@dataclass
class EmbeddingProfile(LoaderBaseModel):
    """
    Profile cấu hình cho embedding model.
    Single Responsibility: Quản lý cấu hình cho embedding provider.
    
    Note: Chỉ chứa các thông số THỰC SỰ được sử dụng trong code.
    Các thông số như batch_size, device, precision không cần thiết cho Ollama.
    """
    model_id: str                                    # ID của model (e.g., "nomic-embed-text")
    provider: str                                    # Provider name (ollama, openai, etc)
    max_tokens: int = 512                            # Max tokens cho input
    dimension: Optional[int] = None                  # Expected embedding dimension
    normalize: bool = True                           # Có normalize vector không

    def __post_init__(self):
        """Validate profile after initialization"""
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.provider:
            raise ValueError("provider cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("dimension must be positive if specified")

    # NOTE: Factory methods đã bị LOẠI BỎ
    # Config giờ nằm TRONG từng embedder class (GemmaEmbedder, BGE3Embedder, etc.)
    # Ví dụ: GemmaEmbedder.MODEL_ID, GemmaEmbedder.DIMENSION, GemmaEmbedder.MAX_TOKENS

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Lấy dictionary config để truyền cho provider.
        Chỉ chứa các thông số thực sự cần thiết.
        """
        config = {
            "model_id": self.model_id,
            "provider": self.provider,
            "max_tokens": self.max_tokens,
            "normalize": self.normalize,
        }

        if self.dimension:
            config["dimension"] = self.dimension

        return config