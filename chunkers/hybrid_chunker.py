"""
Hybrid Chunker
==============
Main orchestrator - kết hợp nhiều chunking strategies.
Single Responsibility: Chọn và điều phối các chunking strategies phù hợp.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from .base_chunker import BaseChunker
from .semantic_chunker import SemanticChunker
from .rule_based_chunker import RuleBasedChunker
from .fixed_size_chunker import FixedSizeChunker
from .model import (
    Chunk, ChunkSet, ChunkType, ChunkStrategy,
    ProvenanceAgg, BlockSpan, Score
)
from loaders.model.document import PDFDocument
from loaders.model.block import Block


class ChunkerMode(Enum):
    """Chế độ hoạt động của HybridChunker"""
    AUTO = "auto"              # Tự động chọn strategy tốt nhất
    SEMANTIC_FIRST = "semantic_first"    # Ưu tiên semantic
    STRUCTURAL_FIRST = "structural_first"  # Ưu tiên structural
    FIXED_SIZE = "fixed_size"  # Chỉ dùng fixed size
    SEQUENTIAL = "sequential"  # Thử lần lượt: structural -> semantic -> fixed


class HybridChunker(BaseChunker):
    """
    Main chunker orchestrator.
    Kết hợp nhiều strategies để tạo chunks tối ưu:
    1. Phân tích document structure
    2. Chọn strategy phù hợp cho từng phần
    3. Apply strategy và merge results
    4. Fallback to fixed-size nếu cần
    """
    
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        mode: ChunkerMode = ChunkerMode.AUTO,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            max_tokens: Số tokens tối đa cho mỗi chunk
            overlap_tokens: Số tokens overlap giữa các chunks
            mode: Chế độ hoạt động
            config: Cấu hình bổ sung
        """
        super().__init__(max_tokens, overlap_tokens)
        self.mode = mode
        self.config = config or {}
        self.validate_config()
        
        # Initialize chunkers
        self.semantic_chunker = SemanticChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_sentences_per_chunk=self.config.get('min_sentences_per_chunk', 3)
        )
        
        self.rule_based_chunker = RuleBasedChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens
        )
        
        self.fixed_size_chunker = FixedSizeChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens
        )

    def __str__(self) -> str:
        return f"HybridChunker (max_tokens={self.max_tokens}, overlap_tokens={self.overlap_tokens}, mode={self.mode.value})"
    
    def chunk(self, document: PDFDocument) -> ChunkSet:
        """
        Chunk toàn bộ document sử dụng hybrid strategy.
        
        Args:
            document: PDFDocument đã được normalized
            
        Returns:
            ChunkSet chứa tất cả chunks
        """
        # Initialize chunk set
        chunk_set = ChunkSet(
            doc_id=document.meta.get('doc_id', 'unknown'),
            file_path=document.file_path,
            chunk_strategy='hybrid'
        )
        
        # Set file_path for all sub-chunkers
        self.semantic_chunker._current_file_path = document.file_path
        self.rule_based_chunker._current_file_path = document.file_path  
        self.fixed_size_chunker._current_file_path = document.file_path
        
        # Collect all blocks
        all_blocks = []
        for page in document.pages:
            for block in page.blocks:
                if isinstance(block, Block) and block.text.strip():
                    all_blocks.append(block)
        
        if not all_blocks:
            return chunk_set
        
        # Chunk based on mode
        if self.mode == ChunkerMode.FIXED_SIZE:
            chunks = self.fixed_size_chunker.chunk_blocks(all_blocks, chunk_set.doc_id)
        elif self.mode == ChunkerMode.SEMANTIC_FIRST:
            chunks = self._chunk_semantic_first(all_blocks, chunk_set.doc_id)
        elif self.mode == ChunkerMode.STRUCTURAL_FIRST:
            chunks = self._chunk_structural_first(all_blocks, chunk_set.doc_id)
        elif self.mode == ChunkerMode.SEQUENTIAL:
            chunks = self._chunk_sequential(all_blocks, chunk_set.doc_id)
        else:  # AUTO
            chunks = self._chunk_auto(all_blocks, chunk_set.doc_id, document)
        
        # Add chunks to set
        for chunk in chunks:
            chunk_set.add_chunk(chunk)
        
        # Link chunks
        chunk_set.link_chunks()
        
        # Update metadata
        chunk_set.metadata['mode'] = self.mode.value
        chunk_set.metadata['strategies_used'] = self._get_strategies_used(chunks)
        
        return chunk_set
    
    def chunk_blocks(self, blocks: List[Block], doc_id: str) -> List[Chunk]:
        """
        Chunk một list blocks (interface method).
        Sử dụng auto mode by default.
        
        Args:
            blocks: Danh sách blocks cần chunk
            doc_id: Document ID
            
        Returns:
            List các Chunk objects
        """
        return self._chunk_auto(blocks, doc_id)
    
    def _chunk_auto(
        self, 
        blocks: List[Block], 
        doc_id: str,
        document: Optional[PDFDocument] = None
    ) -> List[Chunk]:
        """
        Auto mode: Phân tích và chọn strategy tốt nhất cho từng phần.
        
        Strategy:
        1. Phân tích document characteristics
        2. Nếu có structure rõ ràng -> structural chunking
        3. Nếu text liên tục, coherent -> semantic chunking
        4. Fallback -> fixed size
        """
        # Analyze document characteristics
        has_structure = self._has_clear_structure(blocks)
        is_narrative = self._is_narrative_text(blocks)
        avg_block_tokens = sum(self.estimate_tokens(b.text) for b in blocks) / len(blocks)
        
        chunks = []
        
        # Strategy selection
        if has_structure and avg_block_tokens < self.max_tokens:
            # Use structural chunking
            try:
                chunks = self.rule_based_chunker.chunk_blocks(blocks, doc_id)
                # Mark chunks as hybrid
                for chunk in chunks:
                    chunk.chunk_type = ChunkType.HYBRID
            except Exception as e:
                print(f"Structural chunking failed: {e}, falling back to semantic")
                chunks = []
        
        if not chunks and is_narrative:
            # Use semantic chunking
            try:
                chunks = self.semantic_chunker.chunk_blocks(blocks, doc_id)
                # Mark chunks as hybrid
                for chunk in chunks:
                    chunk.chunk_type = ChunkType.HYBRID
            except Exception as e:
                print(f"Semantic chunking failed: {e}, falling back to fixed size")
                chunks = []
        
        # Fallback to fixed size
        if not chunks:
            chunks = self.fixed_size_chunker.chunk_blocks(blocks, doc_id)
            # Mark chunks as hybrid
            for chunk in chunks:
                chunk.chunk_type = ChunkType.HYBRID
        
        return chunks
    
    def _chunk_semantic_first(self, blocks: List[Block], doc_id: str) -> List[Chunk]:
        """Semantic-first mode: Thử semantic trước, fallback fixed size"""
        try:
            chunks = self.semantic_chunker.chunk_blocks(blocks, doc_id)
            if chunks:
                return chunks
        except Exception as e:
            print(f"Semantic chunking failed: {e}")
        
        # Fallback
        return self.fixed_size_chunker.chunk_blocks(blocks, doc_id)
    
    def _chunk_structural_first(self, blocks: List[Block], doc_id: str) -> List[Chunk]:
        """Structural-first mode: Thử structural trước, fallback fixed size"""
        try:
            chunks = self.rule_based_chunker.chunk_blocks(blocks, doc_id)
            if chunks:
                return chunks
        except Exception as e:
            print(f"Structural chunking failed: {e}")
        
        # Fallback
        return self.fixed_size_chunker.chunk_blocks(blocks, doc_id)
    
    def _chunk_sequential(self, blocks: List[Block], doc_id: str) -> List[Chunk]:
        """Sequential mode: Thử structural -> semantic -> fixed size"""
        # Try structural
        try:
            chunks = self.rule_based_chunker.chunk_blocks(blocks, doc_id)
            if chunks and self._validate_chunks(chunks):
                return chunks
        except Exception as e:
            print(f"Structural chunking failed: {e}")
        
        # Try semantic
        try:
            chunks = self.semantic_chunker.chunk_blocks(blocks, doc_id)
            if chunks and self._validate_chunks(chunks):
                return chunks
        except Exception as e:
            print(f"Semantic chunking failed: {e}")
        
        # Fallback to fixed size
        return self.fixed_size_chunker.chunk_blocks(blocks, doc_id)
    
    def _has_clear_structure(self, blocks: List[Block]) -> bool:
        """
        Kiểm tra xem blocks có cấu trúc rõ ràng không.
        (headings, lists, tables)
        """
        structure_indicators = 0
        total_blocks = len(blocks)
        
        for block in blocks:
            text = block.text.strip()
            
            # Check for headings (numbered or capitalized)
            if text and (
                text[0].isupper() and len(text.split()) <= 10 or
                any(text.startswith(str(i) + '.') for i in range(1, 10))
            ):
                structure_indicators += 1
            
            # Check for lists
            if text.startswith(('•', '-', '*', '◦')) or \
               any(text.startswith(f"{i}.") or text.startswith(f"{i})") for i in range(1, 20)):
                structure_indicators += 1
            
            # Check metadata
            if block.metadata and block.metadata.get('type') in ['heading', 'list', 'table']:
                structure_indicators += 1
        
        # If >30% blocks have structure indicators
        return structure_indicators / total_blocks > 0.3 if total_blocks > 0 else False
    
    def _is_narrative_text(self, blocks: List[Block]) -> bool:
        """
        Kiểm tra xem blocks có phải narrative text không.
        (paragraphs liên tục, ít structure)
        """
        if not blocks:
            return False
        
        # Check average block length (narrative text thường có blocks dài)
        avg_length = sum(len(b.text.split()) for b in blocks) / len(blocks)
        
        # Check for discourse markers
        narrative_markers = {
            'however', 'therefore', 'moreover', 'furthermore',
            'in addition', 'as a result', 'consequently',
            'tuy nhiên', 'do đó', 'vì vậy', 'ngoài ra'
        }
        
        marker_count = 0
        for block in blocks:
            text_lower = block.text.lower()
            if any(marker in text_lower for marker in narrative_markers):
                marker_count += 1
        
        # Narrative if average length > 50 words and has discourse markers
        return avg_length > 50 and marker_count / len(blocks) > 0.1
    
    def _validate_chunks(self, chunks: List[Chunk]) -> bool:
        """Validate chunks quality"""
        if not chunks:
            return False
        
        # Check if all chunks are valid
        for chunk in chunks:
            if not chunk.is_valid(self.max_tokens):
                return False
        
        # Check if coverage is reasonable (>80% of input)
        total_chunk_chars = sum(c.char_count for c in chunks)
        if total_chunk_chars < 100:  # Too small
            return False
        
        return True
    
    def _get_strategies_used(self, chunks: List[Chunk]) -> Dict[str, int]:
        """Đếm số lần mỗi strategy được sử dụng"""
        strategies = {}
        for chunk in chunks:
            if chunk.strategy:
                key = chunk.strategy.value
                strategies[key] = strategies.get(key, 0) + 1
        return strategies
    
    def set_mode(self, mode: ChunkerMode):
        """Thay đổi mode"""
        self.mode = mode
    
    def get_stats(self, chunk_set: ChunkSet) -> Dict[str, Any]:
        """Lấy statistics của chunk set"""
        stats = chunk_set.get_stats()
        return {
            'summary': stats.summary(),
            'details': stats.to_dict()
        }
