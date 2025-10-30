"""
Reranker Configuration Profile
==============================
Cấu hình cho reranker models và providers.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RerankerProfile:
    """
    Profile cấu hình cho reranker model.
    Single Responsibility: Quản lý cấu hình cho reranker provider.
    """
    model_id: str                                    # ID của model
    provider: str                                    # Provider name (bge, jina, etc)
    batch_size: int = 8                              # Batch size cho scoring
    max_length: int = 512                            # Max sequence length
    normalize: bool = True                           # Normalize scores không
    use_fp16: bool = False                           # Sử dụng FP16 precision

    def __post_init__(self):
        """Validate profile after initialization"""
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.provider:
            raise ValueError("provider cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    def get_config_dict(self) -> dict:
        """
        Lấy dictionary config để truyền cho provider.
        """
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "normalize": self.normalize,
            "use_fp16": self.use_fp16,
        }