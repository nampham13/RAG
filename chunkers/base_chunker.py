"""
Base Chunker Interface
======================
Định nghĩa abstract base class cho tất cả các chunker strategies.
Tuân thủ Single Responsibility Principle và Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .model import ChunkSet, Chunk
from loaders.model.document import PDFDocument
from loaders.model.block import Block


class BaseChunker(ABC):
    """
    Abstract base class cho tất cả chunking strategies.
    Single Responsibility: Định nghĩa interface chung cho chunkers.
    """
    
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50):
        """
        Args:
            max_tokens: Số tokens tối đa cho mỗi chunk
            overlap_tokens: Số tokens overlap giữa các chunks
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
    
    @abstractmethod
    def chunk(self, document: PDFDocument) -> ChunkSet:
        """
        Chunk toàn bộ document thành ChunkSet.
        
        Args:
            document: PDFDocument đã được normalized
            
        Returns:
            ChunkSet chứa tất cả chunks
        """
        pass
    
    @abstractmethod
    def chunk_blocks(self, blocks: List[Block], doc_id: str) -> List[Chunk]:
        """
        Chunk một list blocks thành list chunks.
        
        Args:
            blocks: Danh sách blocks cần chunk
            doc_id: Document ID
            
        Returns:
            List các Chunk objects
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate chunker configuration"""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """
        Ước lượng số tokens trong text.
        Mặc định: ~4 chars = 1 token (có thể override cho chính xác hơn)
        """
        return len(text) // 4
