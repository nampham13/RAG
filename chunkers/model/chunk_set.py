"""
Chunk Set Model
===============
Tập hợp các chunks cho một document.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .chunk import Chunk
from .score import Score
from loaders.model.base import LoaderBaseModel


@dataclass
class ChunkSet(LoaderBaseModel):
    """
    Tập hợp các chunks cho một document.
    Single Responsibility: Quản lý collection của chunks và thống kê tổng thể.
    """
    doc_id: str                                      # Document ID
    chunks: List[Chunk] = field(default_factory=list)  # Danh sách chunks
    file_path: Optional[str] = None                  # Đường dẫn file nguồn
    total_tokens: int = 0                            # Tổng số tokens
    total_chars: int = 0                             # Tổng số ký tự
    chunk_strategy: Optional[str] = None             # Chiến lược chunking chính
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata

    def add_chunk(self, chunk: Chunk):
        """Thêm một chunk vào set"""
        self.chunks.append(chunk)
        self.total_tokens += chunk.token_count
        self.total_chars += chunk.char_count

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Tìm chunk theo ID"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_chunks_by_page(self, page_number: int) -> List[Chunk]:
        """Lấy tất cả chunks từ một page"""
        result = []
        for chunk in self.chunks:
            if chunk.provenance and page_number in chunk.provenance.page_numbers:
                result.append(chunk)
        return result

    def get_chunks_by_type(self, chunk_type) -> List[Chunk]:
        """Lấy tất cả chunks theo loại"""
        from .enums import ChunkType
        return [chunk for chunk in self.chunks if chunk.chunk_type == chunk_type]

    def link_chunks(self):
        """Liên kết các chunks theo thứ tự (previous/next)"""
        for i in range(len(self.chunks)):
            if i > 0:
                self.chunks[i].previous_chunk_id = self.chunks[i-1].chunk_id
            if i < len(self.chunks) - 1:
                self.chunks[i].next_chunk_id = self.chunks[i+1].chunk_id

    def validate_chunks(self, max_tokens: int = 512) -> List[str]:
        """
        Validate tất cả chunks, trả về list các chunk_id không hợp lệ
        """
        invalid_ids = []
        for chunk in self.chunks:
            if not chunk.is_valid(max_tokens):
                invalid_ids.append(chunk.chunk_id)
        return invalid_ids

    def get_stats(self):
        """Tạo statistics object cho chunk set"""
        from .chunk_stats import ChunkStats
        return ChunkStats.from_chunk_set(self)

    def to_dict(self) -> dict:
        """Chuyển đổi sang dictionary"""
        return {
            'doc_id': self.doc_id,
            'file_path': self.file_path,
            'chunk_count': len(self.chunks),
            'chunks': [chunk.to_dict() for chunk in self.chunks],
            'total_tokens': self.total_tokens,
            'total_chars': self.total_chars,
            'chunk_strategy': self.chunk_strategy,
            'metadata': self.metadata
        }