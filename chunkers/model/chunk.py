"""
Chunk Model
===========
Đại diện cho một đoạn văn bản sẵn sàng để embedding.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .enums import ChunkType, ChunkStrategy
from .provenance_agg import ProvenanceAgg
from .score import Score
from loaders.model.base import LoaderBaseModel


@dataclass
class Chunk(LoaderBaseModel):
    """
    Đại diện cho một đoạn văn bản sẵn sàng để embedding.
    Single Responsibility: Lưu trữ nội dung chunk và metadata liên quan.
    """
    chunk_id: str                                    # Unique ID của chunk
    text: str                                        # Nội dung text
    token_count: int = 0                             # Số tokens (ước lượng)
    char_count: int = 0                              # Số ký tự
    chunk_type: ChunkType = ChunkType.HYBRID         # Loại chunk
    strategy: Optional[ChunkStrategy] = None         # Chiến lược đã dùng
    provenance: Optional[ProvenanceAgg] = None       # Thông tin nguồn gốc
    score: Optional[Score] = None                    # Metrics chất lượng
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata bổ sung

    # Thông tin bổ sung cho chunking context
    previous_chunk_id: Optional[str] = None          # ID của chunk trước
    next_chunk_id: Optional[str] = None              # ID của chunk sau
    section_title: Optional[str] = None              # Tiêu đề section (nếu có)

    def __post_init__(self):
        """Tự động tính toán các metrics cơ bản nếu chưa có"""
        if not self.char_count and self.text:
            self.char_count = len(self.text)
        if not self.token_count and self.text:
            # Ước lượng token count đơn giản: ~4 chars = 1 token
            self.token_count = len(self.text) // 4

    def is_valid(self, max_tokens: int = 512) -> bool:
        """Kiểm tra chunk có hợp lệ không"""
        return (
            bool(self.text.strip()) and
            self.token_count > 0 and
            self.token_count <= max_tokens and
            self.provenance is not None
        )

    def add_context(self, before: str = "", after: str = ""):
        """Thêm context từ chunks xung quanh (nếu cần)"""
        if before:
            self.metadata['context_before'] = before[:100]  # Giới hạn 100 chars
        if after:
            self.metadata['context_after'] = after[:100]
    
    @property
    def textForEmbedding(self) -> str:
        """
        Get text to be used for embedding.
        For table chunks, returns schema-aware embedding_text.
        For regular chunks, returns chunk.text.
        """
        # For table chunks, prefer embedding_text from metadata
        if self.metadata.get('group_type') == 'table':
            embedding_text = self.metadata.get('embedding_text')
            if embedding_text:
                return embedding_text
        
        # Fallback to regular text
        return self.text
    
    @property
    def tokensEstimate(self) -> int:
        """Alias for token_count for backward compatibility."""
        return self.token_count

    def to_dict(self) -> dict:
        """Chuyển đổi sang dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'token_count': self.token_count,
            'char_count': self.char_count,
            'chunk_type': self.chunk_type.value,
            'strategy': self.strategy.value if self.strategy else None,
            'provenance': self.provenance.to_dict() if self.provenance else None,
            'score': self.score.to_dict() if self.score else None,
            'metadata': self.metadata,
            'previous_chunk_id': self.previous_chunk_id,
            'next_chunk_id': self.next_chunk_id,
            'section_title': self.section_title
        }