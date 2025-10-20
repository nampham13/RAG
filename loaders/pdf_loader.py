import os
import gc
import logging
import hashlib
from dataclasses import asdict
from typing import Any, Optional, List, Dict, Tuple
import re
import tempfile
import contextlib
import fitz  # PyMuPDF
import pdfplumber

from .model.page import PDFPage
from .model.document import PDFDocument
from .model.table import TableSchema, TableRow, TableCell
from .model.block import Block
from .model.text import Text

# Configure logging - clear any existing handlers to prevent duplicate logs
logger = logging.getLogger("PDFLoader")
if logger.handlers:
    logger.handlers.clear()


class PDFLoader:
    """
    PDF Loader class - chỉ chịu trách nhiệm load và parse PDF thành structured data.
    KHÔNG bao gồm normalize hay chunking logic.
    
    Tất cả các thuộc tính cấu hình được triển khai trực tiếp trong class theo chuẩn OOP.
    """
    
    def __init__(
        self,
        extract_text: bool = True,
        extract_tables: bool = True,
        tables_engine: str = "auto",
        min_repeated_text_threshold: int = 3,
        min_text_length: int = 10,
        repeated_block_threshold: int = 3,
        enable_repeated_block_filter: bool = True,
        enable_position_filter: bool = True,
        enable_page_number_filter: bool = True,
        enable_empty_filter: bool = True,
        enable_bbox_filter: bool = True,
        min_bbox_area: float = 10.0,
        enable_block_merging: bool = True,
        min_block_length: int = 50,
        enable_block_normalization: bool = True,
        enable_sentence_segmentation: bool = True
    ) -> None:
        """
        Khởi tạo PDFLoader với các thuộc tính cấu hình.
        
        Args:
            extract_text: Có trích xuất text không
            extract_tables: Có trích xuất bảng không  
            tables_engine: Engine để trích xuất bảng ('auto', 'camelot', 'pdfplumber')
            min_repeated_text_threshold: Ngưỡng tối thiểu để phát hiện text lặp lại
            min_text_length: Độ dài text tối thiểu cho block hợp lệ
            repeated_block_threshold: Block lặp lại >= ngưỡng này sẽ bị lọc
            enable_repeated_block_filter: Bật bộ lọc block lặp lại
            enable_position_filter: Bật bộ lọc vị trí
            enable_page_number_filter: Bật bộ lọc số trang
            enable_empty_filter: Bật bộ lọc block rỗng
            enable_bbox_filter: Bật bộ lọc bbox
            min_bbox_area: Diện tích bbox tối thiểu
            enable_block_merging: Bật tính năng merge blocks phân mảnh
            min_block_length: Độ dài tối thiểu để block được coi là "short"
            enable_block_normalization: Bật normalization (stable_id, hash)
            enable_sentence_segmentation: Bật sentence segmentation (requires spaCy)
        """
        # Core extraction settings
        self.extract_text: bool = extract_text
        self.extract_tables: bool = extract_tables
        self.tables_engine: str = tables_engine.lower()
        
        # Block filtering settings
        self.min_repeated_text_threshold: int = min_repeated_text_threshold
        self.min_text_length: int = min_text_length
        self.repeated_block_threshold: int = repeated_block_threshold
        
        # Filter enablement flags
        self.enable_repeated_block_filter: bool = enable_repeated_block_filter
        self.enable_position_filter: bool = enable_position_filter
        self.enable_page_number_filter: bool = enable_page_number_filter
        self.enable_empty_filter: bool = enable_empty_filter
        self.enable_bbox_filter: bool = enable_bbox_filter
        
        # Bbox settings
        self.min_bbox_area: float = min_bbox_area
        
        # Block merging settings
        self.enable_block_merging: bool = enable_block_merging
        self.min_block_length: int = min_block_length
        
        # Block normalization settings
        self.enable_block_normalization: bool = enable_block_normalization
        self.enable_sentence_segmentation: bool = enable_sentence_segmentation
        
        # Validate settings
        self._validate_config()
        
        logger.info(
            "PDFLoader initialized: extract_text=%s, extract_tables=%s, engine=%s, "
            "normalization=%s, sentence_segmentation=%s", 
            self.extract_text, self.extract_tables, self.tables_engine,
            self.enable_block_normalization, self.enable_sentence_segmentation
        )
    
    def _validate_config(self) -> None:
        """Validate cấu hình đầu vào."""
        if self.min_repeated_text_threshold < 1:
            raise ValueError("min_repeated_text_threshold must be >= 1")
        if self.min_text_length < 0:
            raise ValueError("min_text_length must be >= 0")
        if self.repeated_block_threshold < 1:
            raise ValueError("repeated_block_threshold must be >= 1")
        if self.min_bbox_area < 0:
            raise ValueError("min_bbox_area must be >= 0")
        if self.tables_engine not in ('auto', 'camelot', 'pdfplumber'):
            logger.warning("Unknown tables_engine '%s', falling back to 'auto'", self.tables_engine)
            self.tables_engine = 'auto'

    @classmethod
    def create_default(cls) -> 'PDFLoader':
        """
        Factory method để tạo PDFLoader với cấu hình mặc định.
        Equivalent với config cũ từ YAML.
        Enables block normalization and sentence segmentation for chunking.
        """
        return cls(
            extract_text=True,
            extract_tables=True,
            tables_engine="auto",
            min_repeated_text_threshold=3,
            min_text_length=10,
            repeated_block_threshold=3,
            enable_repeated_block_filter=True,
            enable_position_filter=True,
            enable_page_number_filter=True,
            enable_empty_filter=True,
            enable_bbox_filter=True,
            min_bbox_area=10.0,
            enable_block_merging=True,
            min_block_length=50,
            enable_block_normalization=True,
            enable_sentence_segmentation=True
        )
    
    @classmethod
    def create_text_only(cls) -> 'PDFLoader':
        """
        Factory method để tạo PDFLoader chỉ trích xuất text
        """
        return cls(
            extract_text=True,
            extract_tables=False,
            tables_engine="auto",
            min_repeated_text_threshold=3,
            min_text_length=10,
            enable_block_merging=True,
            min_block_length=50,
            enable_block_normalization=True,
            enable_sentence_segmentation=True
        )
    
    @classmethod 
    def create_tables_only(cls) -> 'PDFLoader':
        """
        Factory method để tạo PDFLoader chỉ trích xuất bảng, không có text.
        """
        return cls(
            extract_text=False,
            extract_tables=True,
            tables_engine="auto",
            min_repeated_text_threshold=3,
            enable_block_merging=False,  # Tables only không cần merge blocks
            min_block_length=50
        )

    def load(self, file_path: str) -> PDFDocument:
        """
        Wrapper cho load_pdf để tương thích với pipeline.
        """
        return self.load_pdf(file_path)

    def _file_sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    def _extract_tables_for_page(self, file_path: str, page_num_1based: int, plumber_pdf: Optional[Any]) -> List[Dict[str, Any]]:
        """Extract tables for a page with fallback strategy and error handling."""
        camelot_module = None
        try:
            import camelot as _camelot
            camelot_module = _camelot
        except Exception:
            camelot_module = None
        
        try:
            # Try to extract tables with proper error suppression
            tables = TableSchema.extract_tables_for_page(
                file_path=file_path,
                page_num_1based=page_num_1based,
                plumber_pdf=plumber_pdf,
                tables_engine=self.tables_engine,
                camelot_module=camelot_module,
            )
            return tables if tables else []
        except Exception as e:
            # Log warning but don't raise - continue processing without tables
            logger.debug(f"Table extraction failed for page {page_num_1based}: {e}")
            return []

    def _extract_table_caption(self, page: Any, table_bbox: Optional[tuple], text_blocks: List[Any], next_page_blocks: Optional[List[Any]] = None, page_height: Optional[float] = None) -> Optional[str]:
        """
        Extract table caption from text blocks.
        Delegates to TableCaptionExtractor in table_utils for the actual extraction logic.
        
        Args:
            page: pdfplumber page object (not used, kept for API compatibility)
            table_bbox: (x0, y0, x1, y1) of table
            text_blocks: List of text blocks from current page
            next_page_blocks: List of text blocks from next page (optional)
            page_height: Page height for cross-page detection
            
        Returns:
            Caption text or None if not found
        """
        from .normalizers.table_utils import TableCaptionExtractor
        
        return TableCaptionExtractor.extract_caption(
            table_bbox=table_bbox,
            text_blocks=text_blocks,
            next_page_blocks=next_page_blocks,
            page_height=page_height
        )

    def _deduplicate_text_and_tables(self, blocks: List[Block]) -> List[Block]:
        """
        Remove text blocks that overlap with table blocks (prefer tables for structured data).
        Uses text similarity to detect duplicates.
        """
        from .model.block import TableBlock
        import difflib
        
        text_blocks = []
        table_blocks = []
        
        for block in blocks:
            if isinstance(block, TableBlock):
                table_blocks.append(block)
            else:
                text_blocks.append(block)
        
        if not table_blocks:
            return blocks
        
        logger.debug(f"Deduplication: {len(text_blocks)} text blocks, {len(table_blocks)} table blocks")
        
        # Keep non-duplicate text blocks
        deduplicated_blocks = []
        
        for text_block in text_blocks:
            is_duplicate = False
            text_content = text_block.text.lower().strip()
            # Remove extra whitespace for better comparison
            text_normalized = ' '.join(text_content.split())
            
            # Check if table content is contained within text block
            for table_block in table_blocks:
                table_content = table_block.text.lower().strip()
                table_normalized = ' '.join(table_content.split())
                
                # Skip very short tables
                if len(table_normalized) < 50:
                    continue
                
                # Method 1: Substring containment (fast check)
                # Check if TEXT content is contained WITHIN table (not the reverse!)
                if text_normalized in table_normalized:
                    is_duplicate = True
                    logger.debug(f"Removing duplicate text block (text contained in table)")
                    break
                
                # Method 2: SequenceMatcher for partial matches
                # Calculate how much of TEXT appears in table (check if table "covers" the text)
                # Use find_longest_match to see if significant portion of text is in table
                matcher = difflib.SequenceMatcher(None, text_normalized, table_normalized)
                match = matcher.find_longest_match(0, len(text_normalized), 0, len(table_normalized))
                
                # If >60% of text block appears consecutively in table, it's a duplicate
                coverage_ratio = match.size / len(text_normalized) if len(text_normalized) > 0 else 0
                
                if coverage_ratio > 0.60:
                    is_duplicate = True
                    logger.debug(f"Removing duplicate text block (coverage={coverage_ratio:.2f} in table)")
                    break
            
            if not is_duplicate:
                deduplicated_blocks.append(text_block)
        
        # Add all table blocks
        deduplicated_blocks.extend(table_blocks)
        
        return deduplicated_blocks

    def _open_documents(self, file_path: str) -> Tuple[Optional[Any], Optional[Any], List[str]]:
        """Open PDF documents with proper error handling."""
        file_warnings: List[str] = []
        doc = None
        plumber_pdf = None
        
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            file_warnings.append(f'fitz open failed: {e}')
            return None, None, file_warnings
        
        if self.extract_tables:
            try:
                plumber_pdf = pdfplumber.open(file_path)
            except Exception as e:
                file_warnings.append(f'pdfplumber open failed: {e}')
                plumber_pdf = None
                # Don't close doc here - let caller handle cleanup
        
        return doc, plumber_pdf, file_warnings

    def assign_table_captions(self, table_objs: List[TableSchema], file_path: str) -> None:
        """
        Gán caption cho bảng bằng cách sử dụng pdfplumber để trích xuất text với thông tin chi tiết.
        Tìm text có pattern "Table X.X" trong vùng gần bảng, ưu tiên text có font size lớn.
        Đảm bảo mỗi caption chỉ được dùng 1 lần trên cùng 1 trang.
        """
        from collections import defaultdict
        
        table_regex = re.compile(r"table\s*\d+(\.\d+)*", re.IGNORECASE)
        
        # Track captions đã được sử dụng trên mỗi trang
        used_captions = defaultdict(set)
        
        # Mở PDF với pdfplumber để lấy chi tiết text
        try:
            with pdfplumber.open(file_path) as pdf:
                for t in table_objs:
                    if not hasattr(t, 'bbox') or not t.bbox or not hasattr(t, 'page_number'):
                        continue
                    page_num = t.page_number
                    bbox = t.bbox
                    if not bbox or len(bbox) != 4:
                        continue
                    
                    # pdfplumber page index is 0-based
                    if page_num - 1 >= len(pdf.pages):
                        continue
                    page = pdf.pages[page_num - 1]
                    
                    x0, y0, x1, y1 = bbox
                    
                    # Lấy tất cả text objects trong trang với thông tin chi tiết
                    words = page.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3)
                    
                    # Ghép các words cùng dòng để tạo thành text hoàn chỉnh
                    lines = defaultdict(list)
                    for word in words:
                        y_key = round(word['top'] / 2) * 2  # Group by 2-pixel increments
                        lines[y_key].append(word)
                    
                    # Tìm các dòng có pattern "Table X.X"
                    table_title_candidates = []
                    for y_key, line_words in lines.items():
                        # Sắp xếp words theo x
                        line_words.sort(key=lambda w: w['x0'])
                        # Ghép text
                        line_text = ' '.join(w.get('text', '') for w in line_words)
                        
                        if table_regex.search(line_text):
                            # Lấy bbox của cả dòng
                            wx0 = min(w['x0'] for w in line_words)
                            wy0 = min(w['top'] for w in line_words)
                            wx1 = max(w['x1'] for w in line_words)
                            wy1 = max(w['bottom'] for w in line_words)
                            
                            # Tính khoảng cách đến bảng (kiểm tra cả phía trên và phía dưới)
                            position = None
                            dist_y = None
                            if wy1 <= y0:  # Phía trên bảng
                                dist_y = y0 - wy1
                                position = 'above'
                            elif wy0 >= y1:  # Phía dưới bảng
                                dist_y = wy0 - y1
                                position = 'below'
                            else:  # Chồng lấn với bảng
                                continue
                            
                            dist_x = 0
                            if wx1 < x0:
                                dist_x = x0 - wx1
                            elif wx0 > x1:
                                dist_x = wx0 - x1
                            
                            font_size = max(w.get('height', 10) for w in line_words)
                            table_title_candidates.append({
                                'text': line_text.strip(),
                                'x0': wx0, 'y0': wy0, 'x1': wx1, 'y1': wy1,
                                'dist_y': dist_y,
                                'dist_x': dist_x,
                                'font_size': font_size,
                                'position': position
                            })
                    
                    if not table_title_candidates:
                        # Fallback: tìm dòng text có font size lớn gần bảng (có thể là tiêu đề không chuẩn)
                        fallback_candidates = []
                        for y_key, line_words in lines.items():
                            line_words.sort(key=lambda w: w['x0'])
                            line_text = ' '.join(w.get('text', '') for w in line_words).strip()
                            
                            if not line_text or len(line_text) < 5:
                                continue
                            
                            wx0 = min(w['x0'] for w in line_words)
                            wy0 = min(w['top'] for w in line_words)
                            wx1 = max(w['x1'] for w in line_words)
                            wy1 = max(w['bottom'] for w in line_words)
                            
                            position = None
                            dist_y = None
                            if wy1 <= y0:
                                dist_y = y0 - wy1
                                position = 'above'
                            elif wy0 >= y1:
                                dist_y = wy0 - y1
                                position = 'below'
                            else:
                                continue
                            
                            dist_x = 0
                            if wx1 < x0:
                                dist_x = x0 - wx1
                            elif wx0 > x1:
                                dist_x = wx0 - x1
                            
                            font_size = max(w.get('height', 10) for w in line_words)
                            if dist_y < 50 and dist_x < 100 and font_size >= 9:
                                fallback_candidates.append({
                                    'text': line_text,
                                    'dist_y': dist_y,
                                    'dist_x': dist_x,
                                    'font_size': font_size,
                                    'position': position
                                })
                        
                        if fallback_candidates:
                            below = [c for c in fallback_candidates if c['position'] == 'below']
                            above = [c for c in fallback_candidates if c['position'] == 'above']
                            
                            if below:
                                below.sort(key=lambda x: (x['dist_y'], -x['font_size']))
                                caption = below[0]['text']
                            elif above:
                                above.sort(key=lambda x: (x['dist_y'], -x['font_size']))
                                caption = above[0]['text']
                            else:
                                continue
                            
                            if caption:
                                if not hasattr(t, 'metadata') or t.metadata is None:
                                    t.metadata = {}
                                t.metadata['table_caption'] = caption
                        continue
                    
                    # Chọn candidate tốt nhất: ưu tiên phía dưới, sau đó gần nhất, rồi font_size lớn
                    # Loại bỏ caption đã được sử dụng trên cùng trang này
                    below_candidates = [c for c in table_title_candidates 
                                      if c['position'] == 'below' and c['text'] not in used_captions[page_num]]
                    above_candidates = [c for c in table_title_candidates 
                                      if c['position'] == 'above' and c['text'] not in used_captions[page_num]]
                    
                    best_candidate = None
                    if below_candidates:
                        below_candidates.sort(key=lambda x: (x['dist_y'], x['dist_x'], -x['font_size']))
                        best_candidate = below_candidates[0]
                    elif above_candidates:
                        above_candidates.sort(key=lambda x: (x['dist_y'], x['dist_x'], -x['font_size']))
                        best_candidate = above_candidates[0]
                    
                    if best_candidate:
                        caption = best_candidate['text']
                        if caption:
                            if not hasattr(t, 'metadata') or t.metadata is None:
                                t.metadata = {}
                            t.metadata['table_caption'] = caption
                            # Mark caption as used on this page
                            used_captions[page_num].add(caption)
        except Exception as e:
            logger.warning(f"Caption assignment failed: {e}")

    def load_pdf(self, file_path: str) -> PDFDocument:
        """
        Load PDF with proper resource management and cleanup.
        Ensures all file handles are properly closed even on error.
        """
        pages: List[PDFPage] = []
        doc_id = os.path.basename(file_path)
        doc = None
        plumber_pdf = None
        file_warnings: List[str] = []
        
        try:
            # Extract metadata
            doc_title, page_labels, meta, num_pages = PDFDocument.extract_metadata(file_path)
            
            # Prepare metadata
            if meta is None:
                meta_dict = {}
            else:
                meta_dict = {str(k): str(v) for k, v in meta.items()}
            
            meta_dict['doc_id'] = doc_id
            meta_dict['doc_title'] = doc_title or ''
            meta_dict['file_path'] = file_path
            meta_dict['num_pages'] = str(num_pages)
            meta = meta_dict
            
            # Open documents
            doc, plumber_pdf, open_warnings = self._open_documents(file_path)
            file_warnings.extend(open_warnings)
            
            if doc is None:
                logger.warning(f"Failed to open PDF: {file_path}")
                return PDFDocument(
                    file_path=file_path,
                    num_pages=num_pages,
                    meta=meta,
                    pages=[],
                    warnings=file_warnings
                )
            
            # Extract all blocks for the document
            try:
                all_blocks = Text.collect_all_blocks(doc)
                block_hash_counter = Text.build_block_hash_counter(all_blocks)
            except Exception as e:
                logger.warning(f"Failed to collect blocks: {e}")
                file_warnings.append(f"Block collection failed: {e}")
                all_blocks = None
                block_hash_counter = None
            
            # Process each page
            for page_idx in range(doc.page_count):
                try:
                    page = doc.load_page(page_idx)
                    
                    # Keep original blocks for caption extraction
                    original_blocks = all_blocks[page_idx] if all_blocks and page_idx < len(all_blocks) else []
                    
                    # Extract text blocks
                    blocks = []
                    if self.extract_text and all_blocks is not None and block_hash_counter is not None:
                        text_config = {
                            'enable_block_merging': self.enable_block_merging,
                            'min_block_length': self.min_block_length,
                            'enable_repeated_block_filter': self.enable_repeated_block_filter,
                            'repeated_block_threshold': self.repeated_block_threshold,
                            'min_text_length': self.min_text_length
                        }
                        
                        blocks = Text.extract_text_blocks_for_page(
                            doc=doc,
                            page_idx=page_idx,
                            all_blocks=all_blocks,
                            block_hash_counter=block_hash_counter,
                            config=text_config
                        )
                        
                        # Normalize blocks
                        if self.enable_block_normalization and blocks:
                            for block in blocks:
                                try:
                                    block.set_stable_id_and_hash(doc_id, page_idx + 1)
                                    
                                    if self.enable_sentence_segmentation:
                                        block.normalize()
                                except Exception as e:
                                    logger.debug(f"Block normalization failed on page {page_idx+1}: {e}")
                    
                    page_meta = {
                        "file_path": file_path,
                        "page_number": page_idx + 1,
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "page_label": page_labels.get(page_idx) if isinstance(page_labels, dict) else None,
                        "page_size": {"width": float(page.rect.width), "height": float(page.rect.height)},
                    }
                    
                    # Extract tables
                    tables = []
                    if self.extract_tables:
                        try:
                            raw_tables = self._extract_tables_for_page(file_path, page_idx + 1, plumber_pdf)
                            
                            # Clean tables
                            from loaders.normalizers.table_utils import clean_table, filter_duplicate_tables
                            tables = []
                            for raw_table in raw_tables:
                                matrix = raw_table.get('matrix', raw_table) if isinstance(raw_table, dict) else raw_table
                                bbox = raw_table.get('bbox') if isinstance(raw_table, dict) else None
                                
                                cleaned_matrix = clean_table(matrix)
                                if cleaned_matrix:
                                    tables.append({'matrix': cleaned_matrix, 'bbox': bbox})
                            
                            # Filter duplicates
                            if len(tables) > 1:
                                matrices_only = [t['matrix'] for t in tables]
                                unique_matrices = filter_duplicate_tables(matrices_only)
                                unique_tables = []
                                for unique_mat in unique_matrices:
                                    for table_dict in tables:
                                        if table_dict['matrix'] == unique_mat:
                                            unique_tables.append(table_dict)
                                            break
                                tables = unique_tables
                        except Exception as e:
                            logger.debug(f"Table extraction error on page {page_idx+1}: {e}")
                    
                    # Convert tables to TableBlocks
                    if tables:
                        from .model.block import TableBlock
                        page_height = page_meta.get('page_size', {}).get('height')
                        next_page_blocks = None
                        if page_idx + 1 < doc.page_count:
                            next_page_blocks = all_blocks[page_idx + 1] if all_blocks and page_idx + 1 < len(all_blocks) else None
                        
                        for table_dict in tables:
                            try:
                                matrix = table_dict.get('matrix', [])
                                bbox = table_dict.get('bbox')
                                
                                if not matrix:
                                    continue
                                
                                table_obj = TableSchema.from_matrix(matrix, file_path=file_path, page_number=page_idx + 1, bbox=bbox)
                                
                                # Extract caption
                                caption = self._extract_table_caption(page, bbox, original_blocks, next_page_blocks, page_height)
                                if caption:
                                    if table_obj.metadata is None:
                                        table_obj.metadata = {}
                                    table_obj.metadata['table_caption'] = caption
                                
                                # Create TableBlock
                                table_text_lines = []
                                if table_obj.header:
                                    table_text_lines.append(" | ".join(str(h) for h in table_obj.header))
                                for row in table_obj.rows:
                                    if hasattr(row, "cells"):
                                        row_text = " | ".join(str(c.value) for c in row.cells)
                                        table_text_lines.append(row_text)
                                table_text = "\n".join(table_text_lines)
                                
                                table_block = TableBlock(
                                    text=table_text,
                                    bbox=bbox,
                                    table=table_obj,
                                    metadata={
                                        'doc_id': doc_id,
                                        'page_number': page_idx + 1,
                                        'block_type': 'table',
                                        'table_schema': table_obj if table_obj else None
                                    }
                                )
                                blocks.append(table_block)
                            except Exception as e:
                                logger.debug(f"Table block creation failed on page {page_idx+1}: {e}")
                    
                    # Remove text blocks that overlap with tables (prefer structured table data)
                    if tables:
                        blocks = self._deduplicate_text_and_tables(blocks)
                    
                    pages.append(PDFPage(
                        page_number=page_idx + 1,
                        text="",
                        blocks=blocks,
                        tables=[],
                        warnings=[],
                        source=page_meta,
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_idx+1}: {e}")
                    file_warnings.append(f"Page {page_idx+1} processing error: {e}")
                    pages.append(PDFPage(
                        page_number=page_idx + 1,
                        text="",
                        blocks=[],
                        tables=[],
                        warnings=[str(e)],
                        source={},
                    ))
            
            # Return document
            if not isinstance(meta, dict) or 'doc_id' not in meta:
                if not isinstance(meta, dict):
                    meta = {}
                meta['doc_id'] = doc_id
                meta['file_path'] = file_path
            
            return PDFDocument(
                file_path=file_path,
                num_pages=num_pages if num_pages else doc.page_count,
                meta=meta,
                pages=pages,
                warnings=file_warnings,
                repeated_block_hashes=set()
            )
        
        except Exception as e:
            logger.error(f"Critical error loading PDF {file_path}: {e}")
            return PDFDocument(
                file_path=file_path,
                num_pages=0,
                meta={'doc_id': doc_id, 'file_path': file_path, 'error': str(e)},
                pages=[],
                warnings=[f"Critical error: {e}"]
            )
        
        finally:
            # Ensure proper cleanup of resources
            try:
                if plumber_pdf is not None:
                    plumber_pdf.close()
            except Exception as e:
                logger.debug(f"Error closing pdfplumber: {e}")
            
            try:
                if doc is not None:
                    doc.close()
            except Exception as e:
                logger.debug(f"Error closing fitz document: {e}")
            
            # Force garbage collection to prevent memory leaks
            gc.collect()

    def load_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Load tất cả PDF files trong một directory."""
        pdf_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith('.pdf')
        ]
        return [asdict(self.load_pdf(f)) for f in pdf_files]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Xuất cấu hình hiện tại của loader.
        Useful for debugging và logging.
        """
        return {
            'extract_text': self.extract_text,
            'extract_tables': self.extract_tables,
            'tables_engine': self.tables_engine,
            'min_repeated_text_threshold': self.min_repeated_text_threshold,
            'min_text_length': self.min_text_length,
            'repeated_block_threshold': self.repeated_block_threshold,
            'enable_repeated_block_filter': self.enable_repeated_block_filter,
            'enable_position_filter': self.enable_position_filter,
            'enable_page_number_filter': self.enable_page_number_filter,
            'enable_empty_filter': self.enable_empty_filter,
            'enable_bbox_filter': self.enable_bbox_filter,
            'min_bbox_area': self.min_bbox_area
        }
    
    def update_config(self, **kwargs) -> None:
        """
        Cập nhật cấu hình loader runtime.
        
        Args:
            **kwargs: Các thuộc tính cần cập nhật và giá trị mới
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info("Updated %s to %s", key, value)
            else:
                logger.warning("Unknown config parameter: %s", key)
        
        # Re-validate after update
        self._validate_config()
    
    def enable_all_filters(self) -> None:
        """Bật tất cả các bộ lọc."""
        self.enable_repeated_block_filter = True
        self.enable_position_filter = True
        self.enable_page_number_filter = True
        self.enable_empty_filter = True
        self.enable_bbox_filter = True
        logger.info("All filters enabled")
    
    def disable_all_filters(self) -> None:
        """Tắt tất cả các bộ lọc."""
        self.enable_repeated_block_filter = False
        self.enable_position_filter = False
        self.enable_page_number_filter = False
        self.enable_empty_filter = False
        self.enable_bbox_filter = False
        logger.info("All filters disabled")
    
    def __repr__(self) -> str:
        """String representation of the loader configuration."""
        return (
            f"PDFLoader(extract_text={self.extract_text}, "
            f"extract_tables={self.extract_tables}, "
            f"tables_engine='{self.tables_engine}', "
            f"min_repeated_text_threshold={self.min_repeated_text_threshold})"
        )

