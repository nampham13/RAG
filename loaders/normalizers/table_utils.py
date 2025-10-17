"""
Utilities để lọc và xử lý bảng trích xuất từ PDF.
"""
from typing import List, Optional
import hashlib

def is_header_footer_table(table: List[List[str]], threshold: int = 3) -> bool:
    """
    Kiểm tra xem bảng có phải là header/footer lặp lại hay không.
    Dựa vào:
    - Số hàng nhỏ (<=3)
    - Nhiều ô trống
    - Chứa các keyword như "Classification", "Owner", "Company", "Version"
    - Pattern đặc trưng của header/footer tables
    """
    if not table or len(table) > threshold:
        return False
    
    # Flatten all cells
    all_text = ' '.join(' '.join(str(cell) if cell is not None else "" for cell in row) for row in table).lower()
    
    # Check for common header/footer keywords (expanded list)
    header_keywords = [
        'classification', 'owner', 'company', 'version', 'isms', 'qms',
        'risk management', 'pr_rsk', 'internal', 'committee', 'xxxx-xxxx',
        'process', 'document', 'page', 'revision'
    ]
    keyword_count = sum(1 for kw in header_keywords if kw in all_text)
    
    # Check for specific header patterns
    has_risk_management = 'risk management' in all_text
    has_version_pattern = 'version:' in all_text or 'version 5.0' in all_text
    has_isms_pattern = 'isms/' in all_text or 'pr_' in all_text
    
    # If contains specific header patterns, likely header/footer
    if (has_risk_management and (has_version_pattern or has_isms_pattern)) and len(table) <= 3:
        return True
    
    # If contains multiple keywords and is very short (<=3 rows), likely a header/footer
    # But allow tables that have meaningful content beyond just header info
    if keyword_count >= 2 and len(table) <= 3:
        # Additional check: if table has mostly empty cells or very short content, it's likely header/footer
        total_cells = sum(len(row) for row in table)
        non_empty_cells = sum(1 for row in table for cell in row if str(cell).strip())
        
        # If more than 50% cells are empty and contains header keywords, likely header/footer
        if non_empty_cells / max(total_cells, 1) < 0.5:
            return True
        
        # If all non-empty cells are very short (classification markings), likely header/footer
        non_empty_content = [str(cell).strip() for row in table for cell in row if str(cell).strip()]
        if all(len(content) <= 50 for content in non_empty_content) and any('xxxx-xxxx' in content.lower() for content in non_empty_content):
            return True
    
    return False

def remove_empty_columns(table: List[List[str]]) -> List[List[str]]:
    """
    Loại bỏ các cột hoàn toàn trống khỏi bảng.
    """
    if not table or not table[0]:
        return table
    
    max_cols = max(len(row) for row in table)
    
    # Identify non-empty columns
    non_empty_cols = []
    for col_idx in range(max_cols):
        has_content = False
        for row in table:
            if col_idx < len(row):
                cell_value = str(row[col_idx]).strip() if row[col_idx] is not None else ""
                if cell_value:
                    has_content = True
                    break
        if has_content:
            non_empty_cols.append(col_idx)
    
    if not non_empty_cols:
        return table
    
    # Rebuild table with only non-empty columns
    result = []
    for row in table:
        new_row = [row[col_idx] if col_idx < len(row) else '' for col_idx in non_empty_cols]
        result.append(new_row)
    
    return result

def remove_empty_rows(table: List[List[str]]) -> List[List[str]]:
    """
    Loại bỏ các hàng hoàn toàn trống khỏi bảng.
    """
    if not table:
        return table
    
    result = []
    for row in table:
        # Handle None values and convert to strings
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cleaned_row):  # Keep row if it has any non-empty content
            result.append(cleaned_row)
    
    return result

def is_table_fragment(table: List[List[str]]) -> bool:
    """
    Kiểm tra xem bảng có phải là fragment (phần bị cắt) của bảng lớn không.
    Dựa vào:
    - Có nhiều ô trống (>50%)
    - Chỉ có 1-2 rows
    - Có vẻ như header hoặc footer không đầy đủ
    """
    if not table or len(table) > 3:  # Only check small tables
        return False
    
    total_cells = sum(len(row) for row in table)
    non_empty_cells = sum(1 for row in table for cell in row if str(cell).strip())
    
    # If more than 60% cells are empty, likely a fragment
    if non_empty_cells / max(total_cells, 1) < 0.4:
        return True
    
    # Check for fragment patterns
    all_text = ' '.join(' '.join(str(cell).strip() for cell in row) for row in table).lower()
    
    # Common fragment patterns
    fragment_patterns = [
        'employees', 'criteria', 'level', 'score',  # Common trailing words
        'very high', 'high', 'medium', 'low', 'significant'  # Risk levels without context
    ]
    
    # If table has very few words and matches fragment patterns, likely a fragment
    words = [w for w in all_text.split() if w.strip()]
    if len(words) <= 5 and any(pattern in all_text for pattern in fragment_patterns):
        return True
    
    return False

def compute_table_hash(table: List[List[str]]) -> str:
    """
    Tính hash của bảng để phát hiện các bảng trùng lặp.
    """
    content = '|'.join(','.join(row) for row in table)
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def filter_duplicate_tables(tables: List[List[List[str]]]) -> List[List[List[str]]]:
    """
    Loại bỏ các bảng trùng lặp dựa trên hash.
    """
    if not tables:
        return tables
    
    seen_hashes = set()
    result = []
    
    for table in tables:
        table_hash = compute_table_hash(table)
        if table_hash not in seen_hashes:
            seen_hashes.add(table_hash)
            result.append(table)
    
    return result

def clean_table(table: List[List[str]]) -> Optional[List[List[str]]]:
    """
    Làm sạch bảng: loại bỏ hàng/cột trống, kiểm tra header/footer, kiểm tra fragments.
    Trả về None nếu bảng nên bị loại bỏ.
    """
    if not table:
        return None
    
    # Check if it's a header/footer
    if is_header_footer_table(table):
        return None
    
    # Check if it's a table fragment
    if is_table_fragment(table):
        return None
    
    # Remove empty rows and columns
    table = remove_empty_rows(table)
    table = remove_empty_columns(table)
    
    # After cleaning, check if table still has content
    if not table or len(table) < 2:  # Need at least header + 1 data row
        return None
    
    # Check if table has at least 2 columns (meaningful table)
    max_cols = max(len(row) for row in table)
    if max_cols < 2:
        return None
    
    # Check if table has meaningful content (not just empty strings)
    total_cells = sum(len(row) for row in table)
    non_empty_cells = sum(1 for row in table for cell in row if str(cell).strip())
    
    # If less than 30% cells have content, likely not a meaningful table
    if non_empty_cells / max(total_cells, 1) < 0.3:
        return None
    
    # Additional check: if table is very small (1-2 rows, 2 cols) and contains only short strings, likely not meaningful
    if len(table) <= 2 and max_cols == 2:
        all_cells = [cell for row in table for cell in row]
        if all(len(str(cell).strip()) <= 20 for cell in all_cells):
            # Check if it looks like classification markings or similar
            cell_text = ' '.join(str(cell).lower() for cell in all_cells)
            if any(pattern in cell_text for pattern in ['xxxx-xxxx', 'classification', 'internal']):
                return None
    
    return table

def clean_tables(tables: List[List[List[str]]]) -> List[List[List[str]]]:
    """
    Làm sạch danh sách bảng: loại bỏ header/footer, hàng/cột trống, bảng trùng lặp.
    """
    if not tables:
        return []
    
    # Clean each table
    cleaned = []
    for table in tables:
        clean = clean_table(table)
        if clean is not None:
            cleaned.append(clean)
    
    # Remove duplicates
    cleaned = filter_duplicate_tables(cleaned)
    
    return cleaned


# ========== TABLE CAPTION EXTRACTION ==========

import re
import logging
from typing import Any, Tuple

logger = logging.getLogger("TableCaptionExtractor")


class TableCaptionExtractor:
    """
    Extracts table captions from PDF text blocks.
    Handles single-page and cross-page caption detection.
    """
    
    # Regex pattern for "Table X.X" format
    TABLE_PATTERN = re.compile(r"table\s*\d+(\.\d+)*", re.IGNORECASE)
    
    # Skip patterns for filtering out non-caption text
    SKIP_PATTERNS = ['classification', 'owner', 'company', 'version', 'isms', 'qms', 'page', 'revision']
    
    # Caption keywords for scoring
    CAPTION_KEYWORDS = [
        'table', 'figure', 'chart', 'diagram', 'summary', 'list', 
        'matrix', 'assessment', 'level', 'criteria', 'role', 'description'
    ]
    
    @staticmethod
    def extract_caption(
        table_bbox: Optional[tuple],
        text_blocks: List[Any],
        next_page_blocks: Optional[List[Any]] = None,
        page_height: Optional[float] = None
    ) -> Optional[str]:
        """
        Extract table caption from text blocks.
        
        Args:
            table_bbox: (x0, y0, x1, y1) of the table
            text_blocks: List of text blocks from current page
            next_page_blocks: List of text blocks from next page (optional)
            page_height: Height of current page for cross-page detection
            
        Returns:
            Caption text or None if not found
        """
        if not table_bbox or not text_blocks:
            return None
            
        table_x0, table_y0, table_x1, table_y1 = table_bbox
        
        # Find candidate blocks above table
        above_blocks = TableCaptionExtractor._find_blocks_above(
            table_bbox, text_blocks, max_distance=100
        )
        
        # Find candidate blocks below table
        below_blocks = TableCaptionExtractor._find_blocks_below(
            table_bbox, text_blocks, max_distance=800
        )
        
        # Check next page for large tables or tables at bottom
        next_page_candidates = []
        if next_page_blocks and page_height:
            table_height = table_y1 - table_y0
            is_large_table = table_height > 400
            is_at_bottom = table_y1 > (page_height * 0.85)
            
            if is_large_table or is_at_bottom:
                if is_large_table:
                    logger.debug(f"Large table (height={table_height:.1f}), checking next page")
                if is_at_bottom:
                    logger.debug(f"Table at bottom (y1={table_y1:.1f}), checking next page")
                    
                next_page_candidates = TableCaptionExtractor._find_blocks_in_next_page(
                    table_bbox, next_page_blocks, page_height, max_distance=800
                )
        
        # Combine all candidates
        all_candidates = above_blocks + below_blocks + next_page_candidates
        
        if not all_candidates:
            return None
        
        # Score and select best caption
        scored_candidates = TableCaptionExtractor._score_candidates(all_candidates)
        
        if not scored_candidates:
            return None
        
        # Sort by score (highest first), then by proximity
        scored_candidates.sort(key=lambda x: (-x[0], x[1]))
        
        best_candidate = scored_candidates[0]
        caption_text = best_candidate[2]
        
        logger.debug(f"Selected caption with score {best_candidate[0]}: '{caption_text[:50]}...'")
        
        # Validate caption
        if not TableCaptionExtractor._is_valid_caption(caption_text):
            return None
        
        return caption_text
    
    @staticmethod
    def _find_blocks_above(
        table_bbox: tuple,
        text_blocks: List[Any],
        max_distance: float
    ) -> List[Tuple[float, str]]:
        """Find text blocks above table within max_distance."""
        table_x0, table_y0, table_x1, table_y1 = table_bbox
        candidates = []
        
        for block in text_blocks:
            if isinstance(block, (list, tuple)) and len(block) >= 5:
                x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                
                # Check if above and within distance
                if y1 < table_y0 and (table_y0 - y1) < max_distance:
                    # Check x-range overlap
                    if x1 > table_x0 and x0 < table_x1:
                        candidates.append((y1, text.strip()))
        
        return candidates
    
    @staticmethod
    def _find_blocks_below(
        table_bbox: tuple,
        text_blocks: List[Any],
        max_distance: float
    ) -> List[Tuple[float, str]]:
        """Find text blocks below table within max_distance."""
        table_x0, table_y0, table_x1, table_y1 = table_bbox
        candidates = []
        
        for block in text_blocks:
            if isinstance(block, (list, tuple)) and len(block) >= 5:
                x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                
                # Check if below and within distance
                if y0 > table_y1 and (y0 - table_y1) < max_distance:
                    # Check x-range overlap
                    if x1 > table_x0 and x0 < table_x1:
                        candidates.append((y0, text.strip()))
                        logger.debug(f"Found below block: y={y0:.1f}, text='{text.strip()[:50]}...'")
        
        return candidates
    
    @staticmethod
    def _find_blocks_in_next_page(
        table_bbox: tuple,
        next_page_blocks: List[Any],
        page_height: float,
        max_distance: float
    ) -> List[Tuple[float, str]]:
        """Find text blocks in next page for large/bottom tables."""
        table_x0, table_y0, table_x1, table_y1 = table_bbox
        candidates = []
        
        for block in next_page_blocks:
            if isinstance(block, (list, tuple)) and len(block) >= 5:
                x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                
                # Look within first max_distance units of next page
                if y0 < max_distance:
                    # Check x-range overlap
                    if x1 > table_x0 and x0 < table_x1:
                        # Use virtual y position for sorting
                        virtual_y = page_height + y0
                        candidates.append((virtual_y, text.strip()))
                        logger.debug(f"Found next page block: y={y0:.1f}, virtual_y={virtual_y:.1f}, text='{text.strip()[:50]}...'")
        
        return candidates
    
    @staticmethod
    def _score_candidates(candidates: List[Tuple[float, str]]) -> List[Tuple[int, float, str]]:
        """Score caption candidates based on various criteria."""
        scored = []
        
        for pos, text in candidates:
            if not text.strip():
                continue
            
            score = 0
            text_lower = text.lower()
            
            # HIGH PRIORITY: Matches "Table X.X" pattern
            if TableCaptionExtractor.TABLE_PATTERN.search(text):
                score += 50
            
            # Contains caption keywords
            if any(keyword in text_lower for keyword in TableCaptionExtractor.CAPTION_KEYWORDS):
                score += 10
            
            # Reasonable length
            text_len = len(text)
            if 15 <= text_len <= 200:
                score += 5
            elif text_len > 200:
                score -= 5
            
            # Starts with capital or number
            if text.strip()[0].isupper() or text.strip()[0].isdigit():
                score += 3
            
            # Not a bullet point or fragment
            if not text.strip().startswith(('•', '-', '○', '*')) and len(text.strip()) > 10:
                score += 2
            
            scored.append((score, pos, text))
        
        return scored
    
    @staticmethod
    def _is_valid_caption(text: str) -> bool:
        """Validate if text is a valid caption."""
        # Too short
        if len(text) < 10:
            logger.debug(f"  → Rejected: too short ({len(text)} chars)")
            return False
        
        # Contains skip patterns
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in TableCaptionExtractor.SKIP_PATTERNS):
            logger.debug(f"  → Rejected: contains skip pattern")
            return False
        
        # Short lines without "Table X.X" pattern
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) <= 2 and all(len(line) < 30 for line in lines):
            if not TableCaptionExtractor.TABLE_PATTERN.search(text):
                logger.debug(f"  → Rejected: short lines ({len(lines)} lines)")
                return False
        
        # Starts with digit but not "Table X.X"
        if text.strip()[0].isdigit() and not TableCaptionExtractor.TABLE_PATTERN.search(text):
            logger.debug(f"  → Rejected: starts with digit but not Table pattern")
            return False
        
        # Bullet points or lists without context
        if text.strip().startswith(('•', '-', '○', '*')) and len(text.strip()) < 50:
            logger.debug(f"  → Rejected: bullet point without context")
            return False
        
        # Must have caption keywords OR reasonable length
        if any(keyword in text_lower for keyword in TableCaptionExtractor.CAPTION_KEYWORDS):
            return True
        
        if 15 < len(text) < 200:
            return True
        
        logger.debug(f"  → Rejected: no caption keywords and not reasonable length")
        return False
