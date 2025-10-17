"""
Block-Aware Chunker - Tận dụng merged blocks từ PDFLoader để chunking với source tracking.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
from datetime import datetime

from loaders.model.document import PDFDocument
from loaders.model.page import PDFPage


@dataclass
class ChunkSource:
    """Thông tin nguồn của chunk để trích xuất."""
    file_path: str
    page_number: int
    block_indices: List[int]  # Indices của các blocks gốc tạo thành chunk
    bbox: Optional[Tuple[float, float, float, float]] = None  # Combined bbox
    chunk_id: str = ""
    created_at: str = ""


@dataclass
class Chunk:
    """Chunk với source tracking đầy đủ."""
    text: str
    chunk_id: str
    source: ChunkSource
    metadata: Dict[str, Any]
    word_count: int = 0
    char_count: int = 0
    
    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.text.split()) if self.text else 0
        if not self.char_count:
            self.char_count = len(self.text)


class BlockAwareChunker:
    """
    Chunker tận dụng merged blocks từ PDFLoader.
    
    Ưu điểm:
    - Sử dụng blocks đã được merge chuẩn từ loader
    - Source tracking chi tiết (file, page, block indices)
    - Chunk boundaries respect block boundaries 
    - Metadata preservation từ PDF
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 respect_block_boundaries: bool = True,
                 min_chunk_size: int = 100,
                 include_metadata: bool = True):
        """
        Initialize BlockAwareChunker.
        
        Args:
            chunk_size: Target size của chunk (characters)
            chunk_overlap: Overlap giữa chunks (characters)
            respect_block_boundaries: Có tôn trọng block boundaries không
            min_chunk_size: Size tối thiểu của chunk
            include_metadata: Có include metadata trong chunks không
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_block_boundaries = respect_block_boundaries
        self.min_chunk_size = min_chunk_size
        self.include_metadata = include_metadata
    
    def chunk_document(self, document: PDFDocument) -> List[Chunk]:
        """
        Chunk toàn bộ document sử dụng merged blocks.
        
        Args:
            document: PDFDocument đã được load và merge blocks
            
        Returns:
            List các Chunks với source tracking
        """
        all_chunks = []
        
        for page in document.pages:
            file_path = getattr(document, 'file_path', '') or (document.pages[0].source.get('file_path', '') if document.pages else '')
            page_chunks = self.chunk_page(page, file_path)
            all_chunks.extend(page_chunks)
        
        return all_chunks
    
    def chunk_page(self, page: PDFPage, file_path: str) -> List[Chunk]:
        """
        Chunk một page sử dụng merged blocks.
        
        Args:
            page: PDFPage với merged blocks
            file_path: Path của file PDF
            
        Returns:
            List chunks từ page này
        """
        if not page.blocks:
            return []
        
        chunks = []
        current_chunk_text = ""
        current_block_indices = []
        current_bbox = None
        
        for block_idx, block in enumerate(page.blocks):
            if not isinstance(block, (tuple, list)) or len(block) < 5:
                continue
            
            block_text = str(block[4]).strip()
            if not block_text:
                continue
            
            # Tính bbox của block nếu có
            block_bbox = None
            if len(block) >= 4 and isinstance(block[0], (int, float)):
                try:
                    block_bbox = (float(block[0]), float(block[1]), float(block[2]), float(block[3]))
                except (ValueError, TypeError):
                    block_bbox = None
            
            # Check xem có nên bắt đầu chunk mới không
            potential_text = current_chunk_text + (" " if current_chunk_text else "") + block_text
            
            if (len(potential_text) > self.chunk_size and current_chunk_text):
                # Tạo chunk với text hiện tại
                if len(current_chunk_text) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        current_chunk_text,
                        file_path,
                        page.page_number,
                        current_block_indices.copy(),
                        current_bbox
                    )
                    chunks.append(chunk)
                
                # Bắt đầu chunk mới với overlap
                if self.chunk_overlap > 0 and current_chunk_text:
                    overlap_text = current_chunk_text[-self.chunk_overlap:]
                    current_chunk_text = overlap_text + " " + block_text
                else:
                    current_chunk_text = block_text
                
                current_block_indices = [block_idx]
                current_bbox = block_bbox
            else:
                # Thêm vào chunk hiện tại
                if current_chunk_text:
                    current_chunk_text += " " + block_text
                else:
                    current_chunk_text = block_text
                
                current_block_indices.append(block_idx)
                
                # Update combined bbox
                if block_bbox:
                    if current_bbox:
                        current_bbox = self._combine_bboxes(current_bbox, block_bbox)
                    else:
                        current_bbox = block_bbox
        
        # Tạo chunk cuối cùng
        if current_chunk_text and len(current_chunk_text) >= self.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk_text,
                file_path,
                page.page_number,
                current_block_indices,
                current_bbox
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, 
                     text: str, 
                     file_path: str, 
                     page_number: int, 
                     block_indices: List[int],
                     bbox: Optional[Tuple[float, float, float, float]]) -> Chunk:
        """Tạo Chunk object với source tracking."""
        
        # Generate unique chunk ID
        chunk_content = f"{file_path}_{page_number}_{block_indices}_{text[:100]}"
        chunk_id = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()[:12]
        
        # Create source tracking
        source = ChunkSource(
            file_path=file_path,
            page_number=page_number,
            block_indices=block_indices,
            bbox=bbox,
            chunk_id=chunk_id,
            created_at=datetime.now().isoformat()
        )
        
        # Create metadata
        metadata = {
            'chunk_method': 'block_aware',
            'total_blocks': len(block_indices),
            'source_page': page_number,
            'has_bbox': bbox is not None
        }
        
        if self.include_metadata and bbox:
            metadata.update({
                'bbox_area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'bbox_width': bbox[2] - bbox[0],
                'bbox_height': bbox[3] - bbox[1]
            })
        
        return Chunk(
            text=text,
            chunk_id=chunk_id,
            source=source,
            metadata=metadata
        )
    
    def _combine_bboxes(self, 
                       bbox1: Tuple[float, float, float, float], 
                       bbox2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Combine hai bboxes thành bbox bao phủ."""
        return (
            min(bbox1[0], bbox2[0]),  # min x
            min(bbox1[1], bbox2[1]),  # min y
            max(bbox1[2], bbox2[2]),  # max x
            max(bbox1[3], bbox2[3])   # max y
        )
    
    def get_chunk_source_info(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Trích xuất thông tin source chi tiết của chunk.
        
        Returns:
            Dict với thông tin source để citation/reference
        """
        return {
            'file_path': chunk.source.file_path,
            'page_number': chunk.source.page_number,
            'block_indices': chunk.source.block_indices,
            'chunk_id': chunk.chunk_id,
            'bbox': chunk.source.bbox,
            'created_at': chunk.source.created_at,
            'word_count': chunk.word_count,
            'char_count': chunk.char_count,
            'citation': f"{chunk.source.file_path.split('/')[-1]}, Page {chunk.source.page_number}, Blocks {'-'.join(map(str, chunk.source.block_indices))}"
        }
    
    def find_chunks_by_source(self, chunks: List[Chunk], file_path: str, page_number: Optional[int] = None) -> List[Chunk]:
        """
        Tìm chunks theo source (file và page).
        
        Args:
            chunks: List tất cả chunks
            file_path: Path của file cần tìm
            page_number: Page number (optional)
            
        Returns:
            List chunks matching criteria
        """
        result = []
        for chunk in chunks:
            if chunk.source.file_path == file_path:
                if page_number is None or chunk.source.page_number == page_number:
                    result.append(chunk)
        return result
    
    @classmethod
    def create_from_merged_loader(cls, chunk_size: int = 1000) -> 'BlockAwareChunker':
        """
        Factory method để tạo chunker tối ưu cho merged blocks từ PDFLoader.
        """
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=200,
            respect_block_boundaries=True,  # Tôn trọng merged blocks
            min_chunk_size=100,
            include_metadata=True
        )
    
    @classmethod 
    def create_citation_focused(cls) -> 'BlockAwareChunker':
        """
        Factory method để tạo chunker focus vào citation/source tracking.
        """
        return cls(
            chunk_size=800,  # Smaller chunks for better precision
            chunk_overlap=150,
            respect_block_boundaries=True,
            min_chunk_size=50,
            include_metadata=True
        )