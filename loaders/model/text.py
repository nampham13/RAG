"""
Text model and extraction utilities.
Handles text block extraction and processing for PDF pages.
"""
import logging
from typing import List, Any, Optional, Dict
from collections import Counter
from .block import Block

logger = logging.getLogger("Text")


class Text:
    """
    Text extraction and processing for PDF documents.
    Similar to TableSchema, provides static methods for text extraction.
    """
    
    @staticmethod
    def extract_text_blocks_for_page(
        doc: Any,
        page_idx: int,
        all_blocks: List[List[Any]],
        block_hash_counter: Counter,
        config: Optional[Dict[str, Any]] = None
    ) -> List[Block]:
        """
        Extract and process text blocks for a single page.
        
        Args:
            doc: PyMuPDF document object
            page_idx: Zero-based page index
            all_blocks: All blocks from entire document (for context)
            block_hash_counter: Counter of block hashes for duplicate detection
            config: Configuration dict with filtering/merging options
            
        Returns:
            List of Block objects for this page
        """
        if config is None:
            config = {}
        
        # Get blocks for this page
        blocks = all_blocks[page_idx] if all_blocks and page_idx < len(all_blocks) else []
        if not blocks:
            return []
        
        # Keep original blocks for caption extraction
        original_blocks = blocks.copy()
        
        # Apply block merging if enabled
        enable_block_merging = config.get('enable_block_merging', False)
        if enable_block_merging:
            from loaders.normalizers.text_utils import merge_blocks_list
            merge_config = {
                'min_block_length': config.get('min_block_length', 10),
                'sentence_endings': ('.', '!', '?', ':', ';'),
                'list_markers': ('•', '-', '○', '*')
            }
            original_count = len(blocks)
            blocks = merge_blocks_list(blocks, merge_config)
            if len(blocks) != original_count:
                logger.debug(f"Page {page_idx+1}: Merged {original_count} blocks into {len(blocks)} blocks")
        
        # Convert to Block objects
        page = doc.load_page(page_idx)
        page_height = float(page.rect.height)
        
        block_objects = []
        for block_tuple in blocks:
            # block_tuple: (x0, y0, x1, y1, text, block_no, block_type)
            if len(block_tuple) < 5:
                continue
                
            bbox = block_tuple[:4]
            text = block_tuple[4]
            
            # Apply filtering if enabled
            enable_repeated_block_filter = config.get('enable_repeated_block_filter', False)
            repeated_block_threshold = config.get('repeated_block_threshold', 3)
            min_text_length = config.get('min_text_length', 1)
            
            if Text.should_filter_block(
                text=text,
                bbox=bbox,
                page_height=page_height,
                block_hash_counter=block_hash_counter,
                enable_repeated_block_filter=enable_repeated_block_filter,
                repeated_block_threshold=repeated_block_threshold,
                min_text_length=min_text_length
            ):
                continue
            
            # Create Block object
            block = Block(
                text=text,
                bbox=bbox,
                text_source="pymupdf",
                metadata={}
            )
            block_objects.append(block)
        
        return block_objects
    
    @staticmethod
    def should_filter_block(
        text: str,
        bbox: Optional[tuple],
        page_height: float,
        block_hash_counter: Counter,
        enable_repeated_block_filter: bool = False,
        repeated_block_threshold: int = 3,
        min_text_length: int = 1
    ) -> bool:
        """
        Determine if a block should be filtered out.
        
        Args:
            text: Block text content
            bbox: Block bounding box (x0, y0, x1, y1)
            page_height: Page height for header/footer detection
            block_hash_counter: Counter for repeated block detection
            enable_repeated_block_filter: Whether to filter repeated blocks
            repeated_block_threshold: Threshold for repeated block count
            min_text_length: Minimum text length to keep
            
        Returns:
            True if block should be filtered out, False otherwise
        """
        from loaders.normalizers.text_utils import (
            is_repeated_block,
            is_header_footer_block,
            is_toc_or_noise_block
        )
        
        # Filter short blocks
        if not text or len(text.strip()) < min_text_length:
            return True
        
        # Filter repeated blocks (headers/footers)
        if enable_repeated_block_filter:
            if is_repeated_block(text, block_hash_counter, repeated_block_threshold):
                return True
        
        # Filter header/footer based on position
        if is_header_footer_block(text, bbox, page_height):
            return True
        
        # Filter TOC and noise
        if is_toc_or_noise_block(text):
            return True
        
        return False
    
    @staticmethod
    def collect_all_blocks(doc: Any) -> List[List[Any]]:
        """
        Collect raw text blocks from all pages in document.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            List of lists of block tuples, one list per page
        """
        all_blocks = []
        for page_idx in range(doc.page_count):
            try:
                page = doc.load_page(page_idx)
                blocks = page.get_text("blocks")
                all_blocks.append(blocks)
            except Exception as e:
                logger.warning(f"Failed to get blocks for page {page_idx+1}: {e}")
                all_blocks.append([])
        return all_blocks
    
    @staticmethod
    def build_block_hash_counter(all_blocks: List[List[Any]]) -> Counter:
        """
        Build a counter of block hashes across entire document.
        Used for detecting repeated blocks (headers/footers).
        
        Args:
            all_blocks: All blocks from all pages
            
        Returns:
            Counter mapping block hash to occurrence count
        """
        from loaders.normalizers.text_utils import compute_block_hash
        
        block_hash_counter = Counter()
        for blocks_in_page in all_blocks:
            for block_tuple in blocks_in_page:
                # block_tuple: (x0, y0, x1, y1, text, block_no, block_type)
                if len(block_tuple) >= 5:
                    text = block_tuple[4]
                    if text and len(text.strip()) >= 5:
                        block_hash = compute_block_hash(text)
                        if block_hash:
                            block_hash_counter[block_hash] += 1
        return block_hash_counter
