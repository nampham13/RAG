
import ftfy
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional
from .page import PDFPage
from .base import LoaderBaseModel
from cleantext import clean

@dataclass
class PDFDocument(LoaderBaseModel):

    @staticmethod
    def collect_all_blocks(doc: Any) -> list:
        """Return list of blocks for each page using pymupdf page.get_text('blocks')."""
        all_blocks = []
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            get_text = getattr(page, "get_text", None)
            if callable(get_text):
                result = get_text("blocks")
                blocks = result if isinstance(result, list) else []
            else:
                blocks = []
            all_blocks.append(blocks)
        return all_blocks
    file_path: str
    num_pages: int
    meta: Dict[str, Any] = field(default_factory=dict)
    pages: List[PDFPage] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repeated_block_hashes: Set[str] = field(default_factory=set)

    @staticmethod
    def extract_metadata(file_path: str):
        """Extract metadata using PyPDF2. Returns (doc_title, page_labels, meta, num_pages)."""
        import PyPDF2
        import logging
        doc_title = None
        page_labels = None
        meta = {}
        num_pages = 0
        try:
            pdf_r = PyPDF2.PdfReader(file_path)
            meta = dict(pdf_r.metadata or {})
            num_pages = len(pdf_r.pages)
            try:
                page_labels = None
                if hasattr(pdf_r, "outlines") and pdf_r.outlines:
                    page_labels = {}
                    for i, _ in enumerate(pdf_r.pages):
                        page_labels[i] = str(i + 1)
            except Exception:
                page_labels = None
            title = None
            if isinstance(meta, dict):
                title = meta.get("/Title") or meta.get("Title")
            doc_title = title if isinstance(title, str) and title.strip() else None
        except Exception as e:
            logging.warning(f"Meta extraction failed: {e}")
        return doc_title, page_labels, meta, num_pages
    
    def normalize(self, config: Optional[dict] = None) -> 'PDFDocument':
        # 1. Chuẩn hóa file_path
        if self.file_path:
            self.file_path = clean(ftfy.fix_text(str(self.file_path)).strip())
        # 2. Chuẩn hóa meta
        if self.meta:
            self.meta = {k: clean(ftfy.fix_text(str(v)).strip()) for k, v in self.meta.items() if k}

        # 3. Chuẩn hóa từng page/block/table trước
        if self.pages:
            for p in self.pages:
                if hasattr(p, 'normalize') and callable(p.normalize):
                    p.normalize(config=config)

        # 4. Thu thập block_hash_counter sau khi đã normalize text
        from ..normalizers.block_utils import compute_block_hash, should_filter_block
        from collections import Counter
        block_hash_counter = Counter()
        for page in self.pages:
            blocks = getattr(page, 'blocks', [])
            for b in blocks:
                text = None
                if isinstance(b, (list, tuple)) and len(b) >= 5:
                    text = b[4]
                elif not isinstance(b, (list, tuple)) and hasattr(b, 'text'):
                    text = getattr(b, 'text', None)
                if text is None:
                    continue
                if isinstance(text, str) and len(text.strip()) >= 5:
                    block_hash = compute_block_hash(text)
                    if block_hash:
                        block_hash_counter[block_hash] += 1

        # 5. Lọc blocks dựa trên block_hash_counter
        for page in self.pages:
            blocks = getattr(page, 'blocks', [])
            filtered_blocks = []
            for b in blocks:
                text = None
                bbox = None
                if isinstance(b, (list, tuple)) and len(b) >= 5:
                    text = b[4]
                    bbox = (b[0], b[1], b[2], b[3]) if len(b) >= 4 else None
                elif not isinstance(b, (list, tuple)) and hasattr(b, 'text'):
                    text = getattr(b, 'text', None)
                    bbox = getattr(b, 'bbox', None)
                if text is None:
                    filtered_blocks.append(b)
                    continue
                # Check if should filter
                if should_filter_block(text, bbox, config, block_hash_counter):
                    # Mark as filtered for Block objects only (not tuple/list)
                    if not isinstance(b, (list, tuple)) and hasattr(b, 'metadata'):
                        if not getattr(b, 'metadata', None):
                            b.metadata = {}
                        b.metadata['filtered'] = True
                        b.metadata['filter_reason'] = 'repeated_or_noise'
                    # Skip adding to filtered list
                    continue
                else:
                    filtered_blocks.append(b)
            # Update page blocks
            page.blocks = filtered_blocks

        # 6. Chuẩn hóa warnings
        if self.warnings:
            self.warnings = list({clean(ftfy.fix_text(str(w)).strip()) for w in self.warnings if w})
        # 7. Chuẩn hóa repeated_block_hashes
        if self.repeated_block_hashes:
            self.repeated_block_hashes = set([str(h).strip() for h in self.repeated_block_hashes if h])
        return self