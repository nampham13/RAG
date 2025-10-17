"""
Embedding Request Model
======================
Đại diện cho một yêu cầu embedding từ chunk.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from loaders.model.base import LoaderBaseModel


@dataclass
class EmbedRequest(LoaderBaseModel):
    """
    Yêu cầu embedding cho một chunk.
    Single Responsibility: Đóng gói thông tin cần thiết để tạo embedding.
    """
    chunk_id: str                                    # ID của chunk
    text: str                                        # Nội dung cần embedding
    chunk_type: str = "text"                         # Loại chunk (text, table, etc)
    priority: int = 1                                # Độ ưu tiên (1=cao, 5=thấp)
    context: Optional[str] = None                    # Context bổ sung nếu có
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata bổ sung

    def __post_init__(self):
        """Validate request after initialization"""
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not self.text:
            raise ValueError("text cannot be empty")
        if self.priority < 1 or self.priority > 5:
            raise ValueError("priority must be between 1 and 5")

    def get_embedding_text(self) -> str:
        """
        Lấy text cuối cùng để embedding, có thể kết hợp với context.
        """
        if self.context:
            return f"{self.context}\n\n{self.text}"
        return self.text

    def normalize(self) -> 'EmbedRequest':
        """
        Chuẩn hóa request (remove extra whitespace, etc.)
        """
        # REMOVED: Clean text using cleantext library (was corrupting text)
        # from cleantext import clean

        # normalized_text = clean(self.text) if self.text else ""
        # normalized_context = clean(self.context) if self.context else None
        
        # Simple normalization without clean-text
        normalized_text = self.text.strip() if self.text else ""
        normalized_context = self.context.strip() if self.context else None

        return EmbedRequest(
            chunk_id=self.chunk_id,
            text=normalized_text,
            chunk_type=self.chunk_type,
            priority=self.priority,
            context=normalized_context,
            metadata=self.metadata.copy()
        )