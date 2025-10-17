"""
Utilities để lọc và xử lý block trích xuất từ PDF.
"""
from typing import List, Optional, Tuple, Set
import re
import hashlib

def compute_block_hash(text: str) -> str:
    """
    Tính hash của block text để phát hiện block lặp lại.
    """
    if not text:
        return ""
    # Normalize trước khi hash để tăng khả năng match
    normalized = text.strip().lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def is_repeated_block(text: str, block_hash_counter: dict, threshold: int = 3) -> bool:
    """
    Kiểm tra block có lặp lại nhiều lần trong document hay không.
    
    Args:
        text: Nội dung block
        block_hash_counter: Dict đếm số lần xuất hiện của mỗi hash
        threshold: Ngưỡng số lần lặp lại để coi là header/footer (mặc định 3)
    
    Returns:
        True nếu block lặp lại >= threshold lần
    """
    if not text or len(text.strip()) < 5:
        return False
    
    block_hash = compute_block_hash(text)
    if not block_hash:
        return False
    
    # Count occurrences
    count = block_hash_counter.get(block_hash, 0)
    
    # Consider as repeated if appears >= threshold times
    return count >= threshold

def is_header_footer_block(text: str, bbox: Optional[Tuple] = None, page_height: float = 792.0) -> bool:
    """
    Kiểm tra xem block có phải là header/footer dựa vào vị trí và độ ngắn.
    
    Args:
        text: Nội dung block
        bbox: Bounding box (x0, y0, x1, y1)
        page_height: Chiều cao trang (mặc định 792 cho Letter size)
    
    Returns:
        True nếu là header/footer
    """
    if not text or len(text.strip()) < 5:
        return False
    
    # Check if text is very short (likely metadata)
    text_stripped = text.strip()
    if len(text_stripped) < 20 and len(text_stripped.split()) <= 5:
        # Check if in header/footer position
        if bbox and isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
            y0, y1 = bbox[1], bbox[3]
            # Top 10% or bottom 10% of page
            if y0 < page_height * 0.1 or y1 > page_height * 0.9:
                return True
    
    return False

def is_page_number_block(text: str) -> bool:
    """
    Kiểm tra xem block có phải là số trang hay không.
    Pattern: "Page X/Y", "Page X of Y", "X/Y", etc.
    """
    if not text or len(text.strip()) > 30:
        return False
    
    text_stripped = text.strip()
    
    # Pattern matching for page numbers
    page_patterns = [
        r'^\s*page\s+\d+\s*/\s*\d+\s*$',
        r'^\s*page\s+\d+\s+of\s+\d+\s*$',
        r'^\s*\d+\s*/\s*\d+\s*$',
        r'^\s*\[\s*\d+\s*\]\s*$',
    ]
    
    for pattern in page_patterns:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return True
    
    return False

def is_empty_or_whitespace_block(text: str, min_length: int = 3) -> bool:
    """
    Kiểm tra xem block có rỗng hoặc chỉ chứa whitespace hay không.
    """
    if not text:
        return True
    
    text_stripped = text.strip()
    
    # Check if only whitespace/newline
    if not text_stripped:
        return True
    
    # Check if too short after cleaning
    if len(text_stripped) < min_length:
        return True
    
    return False

def is_bbox_too_small(bbox: Optional[Tuple], min_area: float = 10.0, min_width: float = 5.0, min_height: float = 3.0) -> bool:
    """
    Kiểm tra xem bbox có quá nhỏ (noise) hay không.
    """
    if not bbox or not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
        return False
    
    try:
        x0, y0, x1, y1 = bbox[:4]
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        area = width * height
        
        if area < min_area or width < min_width or height < min_height:
            return True
    except (ValueError, TypeError):
        return False
    
    return False

def normalize_whitespace(text: str) -> str:
    """
    Chuẩn hóa whitespace: multiple spaces -> single space, multiple newlines -> max 2 newlines.
    """
    if not text:
        return text
    
    # Normalize multiple spaces to single space (không áp dụng cho newline)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize multiple newlines to max 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def should_filter_block(
    text: str, 
    bbox: Optional[Tuple], 
    config: Optional[dict] = None,
    block_hash_counter: Optional[dict] = None,
    page_height: float = 792.0
) -> bool:
    """
    Tổng hợp tất cả các điều kiện lọc block.
    Trả về True nếu block nên bị loại bỏ.
    
    Args:
        text: Nội dung block
        bbox: Bounding box
        config: Cấu hình filter
        block_hash_counter: Dict đếm số lần xuất hiện của block (để phát hiện lặp)
        page_height: Chiều cao trang
    """
    if config is None:
        config = {}
    
    # Get config values
    enable_repeated_filter = config.get('enable_repeated_block_filter', True)
    enable_position_filter = config.get('enable_position_filter', True)
    enable_page_number_filter = config.get('enable_page_number_filter', True)
    enable_empty_filter = config.get('enable_empty_filter', True)
    enable_bbox_filter = config.get('enable_bbox_filter', True)
    
    min_text_length = config.get('min_text_length', 3)
    min_bbox_area = config.get('min_bbox_area', 10.0)
    repeated_threshold = config.get('repeated_block_threshold', 3)
    
    # Apply filters
    # 1. Check for repeated blocks (header/footer pattern)
    if enable_repeated_filter and block_hash_counter:
        if is_repeated_block(text, block_hash_counter, threshold=repeated_threshold):
            return True
    
    # 2. Check for position-based header/footer
    if enable_position_filter:
        if is_header_footer_block(text, bbox, page_height):
            return True
    
    # 3. Check for page numbers
    if enable_page_number_filter and is_page_number_block(text):
        return True
    
    # 4. Check for empty/whitespace
    if enable_empty_filter and is_empty_or_whitespace_block(text, min_length=min_text_length):
        return True
    
    # 5. Check for bbox too small
    if enable_bbox_filter and is_bbox_too_small(bbox, min_area=min_bbox_area):
        return True
    
    return False


# ========== BLOCK MERGING FUNCTIONALITY ==========

def should_merge_blocks(block1, block2, config: Optional[dict] = None) -> Tuple[bool, str]:
    """
    Xác định liệu 2 blocks có nên merge không.
    
    Args:
        block1: Block đầu tiên (tuple format)
        block2: Block thứ hai (tuple format)  
        config: Configuration cho merging
    
    Returns:
        (should_merge: bool, reason: str)
    """
    if config is None:
        config = {}
    
    # Config values
    min_block_length = config.get('min_block_length', 50)
    sentence_endings = config.get('sentence_endings', ('.', '!', '?', ':', ';'))
    list_markers = config.get('list_markers', ('•', '-', '○', '*'))
    
    # Validate blocks
    if not _is_valid_block(block1) or not _is_valid_block(block2):
        return False, "invalid_block"
    
    text1 = str(block1[4]).strip()
    text2 = str(block2[4]).strip()
    
    if not text1 or not text2:
        return False, "empty_text"
    
    # Rule 1: Incomplete sentence continuation
    if (not text1.endswith(sentence_endings) and 
        text2 and not text2[0].isupper() and 
        len(text1) < 100 and len(text2) < 100):
        return True, "incomplete_sentence"
    
    # Rule 2: List item continuation  
    if (_is_list_item(text1, list_markers) and not text1.endswith('.') and
        not _is_list_item(text2, list_markers) and len(text2) < 80):
        return True, "list_continuation"
    
    # Rule 3: Very short blocks (<20 chars) should merge
    if len(text1) < 20 and len(text2) < 100:
        return True, "very_short_block"
    
    # Rule 4: Page headers/numbers
    if _is_page_header(text1) and len(text2) < 100:
        return True, "page_header"
    
    # Rule 5: Section headers without content
    if (_is_section_header(text1) and 
        not _is_section_header(text2) and
        len(text2) < 200):
        return True, "section_header"
    
    return False, "no_merge_needed"


def merge_blocks_list(blocks: List, config: Optional[dict] = None) -> List:
    """
    Merge blocks theo các rules đã định.
    
    Args:
        blocks: List các blocks gốc
        config: Configuration cho merging
        
    Returns:
        List các blocks đã được merge
    """
    if not blocks:
        return blocks
    
    merged_blocks = []
    i = 0
    
    while i < len(blocks):
        current_block = blocks[i]
        
        # Tìm tất cả blocks liên tiếp có thể merge
        merge_chain = [current_block]
        j = i + 1
        
        while j < len(blocks):
            should_merge, reason = should_merge_blocks(
                merge_chain[-1], blocks[j], config
            )
            
            if should_merge:
                merge_chain.append(blocks[j])
                j += 1
            else:
                break
        
        # Merge chain thành 1 block
        if len(merge_chain) > 1:
            merged_block = _merge_block_chain(merge_chain, config)
            merged_blocks.append(merged_block)
        else:
            merged_blocks.append(current_block)
        
        i = j if j > i + 1 else i + 1
    
    return merged_blocks


def _merge_block_chain(blocks: List, config: Optional[dict] = None):
    """Merge một chain các blocks thành 1 block."""
    if not blocks:
        return None
    
    if len(blocks) == 1:
        return blocks[0]
    
    if config is None:
        config = {}
    
    list_markers = config.get('list_markers', ('•', '-', '○', '*'))
    
    # Lấy metadata từ block đầu tiên
    base_block = list(blocks[0])  # Copy
    
    # Merge text content
    merged_text = ""
    for i, block in enumerate(blocks):
        text = str(block[4]).strip()
        
        if i == 0:
            merged_text = text
        else:
            # Smart joining
            if (merged_text and 
                not merged_text.endswith((' ', '\n')) and 
                not text.startswith(list_markers)):
                # Thêm space nếu cần
                if not merged_text.endswith(('-', '•')):
                    merged_text += " "
            
            # Xử lý đặc biệt cho list items
            if text.startswith(list_markers):
                if not merged_text.endswith('\n'):
                    merged_text += "\n"
            
            merged_text += text
    
    # Update merged text
    base_block[4] = merged_text
    
    return tuple(base_block)


def _is_valid_block(block) -> bool:
    """Check if block có format hợp lệ."""
    return (isinstance(block, (tuple, list)) and 
            len(block) >= 5)


def _is_list_item(text: str, list_markers: tuple) -> bool:
    """Check if text là list item."""
    text = text.strip()
    return any(text.startswith(marker) for marker in list_markers)


def _is_page_header(text: str) -> bool:
    """Check if text là page header/number."""
    text = text.strip()
    patterns = [
        r'^Page \d+/\d+$',
        r'^\d+/\d+$',
        r'^ISMS.*Classification.*Internal$'
    ]
    return any(re.match(pattern, text, re.IGNORECASE) for pattern in patterns)


def _is_section_header(text: str) -> bool:
    """Check if text là section header."""
    text = text.strip()
    patterns = [
        r'^\d+\.\s*[A-Z][A-Z\s]*$',  # "2. PROCESS"
        r'^\d+\.\d+\s*[A-Z][A-Z\s]*$',  # "2.1 PROCESS CHARACTERISTICS"
        r'^[A-Z][A-Z\s]{5,}$'  # "RISK APPROACHES"
    ]
    return any(re.match(pattern, text) for pattern in patterns)


def analyze_block_improvement(original_blocks: List, merged_blocks: List, config: Optional[dict] = None) -> dict:
    """Phân tích cải thiện sau khi merge."""
    if config is None:
        config = {}
    
    min_block_length = config.get('min_block_length', 50)
    sentence_endings = config.get('sentence_endings', ('.', '!', '?', ':', ';'))
    
    def count_short_blocks(blocks):
        return sum(1 for block in blocks 
                  if _is_valid_block(block) and 
                  len(str(block[4]).strip()) < min_block_length)
    
    def count_incomplete_sentences(blocks):
        return sum(1 for block in blocks
                  if _is_valid_block(block) and
                  not str(block[4]).strip().endswith(sentence_endings))
    
    original_short = count_short_blocks(original_blocks)
    merged_short = count_short_blocks(merged_blocks)
    
    original_incomplete = count_incomplete_sentences(original_blocks)
    merged_incomplete = count_incomplete_sentences(merged_blocks)
    
    return {
        "original_blocks": len(original_blocks),
        "merged_blocks": len(merged_blocks),
        "blocks_reduced": len(original_blocks) - len(merged_blocks),
        "reduction_percentage": (len(original_blocks) - len(merged_blocks)) / len(original_blocks) * 100 if original_blocks else 0,
        "original_short_blocks": original_short,
        "merged_short_blocks": merged_short,
        "short_blocks_improved": original_short - merged_short,
        "original_incomplete": original_incomplete,
        "merged_incomplete": merged_incomplete,
        "incomplete_improved": original_incomplete - merged_incomplete
    }
