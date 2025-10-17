
import typing
import fitz
import pdfminer.high_level
import io
import ftfy
from cleantext import clean
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from .base import LoaderBaseModel

@dataclass
class PDFPage(LoaderBaseModel):
    @staticmethod
    def extract_text_with_layout(page: Any) -> Dict[str, Any]:
        """Layout-aware text extraction: return concatenated text and line spans with bbox."""
        try:
            blocks = page.get_text("blocks")
        except Exception:
            return {"text": "", "lines": []}
        parts: List[str] = []
        lines: List[Tuple[int,int,Tuple[float,float,float,float]]] = []  # (start,end,bbox)
        offset = 0
        for blk in blocks:
            if not isinstance(blk, (list, tuple)) or len(blk) < 5:
                continue
            x0, y0, x1, y1, blk_text = blk[:5]
            if not isinstance(blk_text, str) or not blk_text.strip():
                offset += len(blk_text) if isinstance(blk_text, str) else 0
                continue
            for raw in blk_text.splitlines(keepends=True):
                start = offset
                end = start + len(raw)
                parts.append(raw)
                lines.append((start, end, (float(x0), float(y0), float(x1), float(y1))))
                offset = end
        return {"text": "".join(parts), "lines": lines}

    @staticmethod
    def expand_char_bbox_map(lines: List[Tuple[int,int,Tuple[float,float,float,float]]]) -> Dict[int, Tuple[float,float,float,float]]:
        mapping: Dict[int, Tuple[float,float,float,float]] = {}
        for start, end, bbox in lines:
            for i in range(start, end):
                mapping[i] = bbox
        return mapping

    @staticmethod
    def pdfminer_extract_text_from_single_page(doc: Any, page_index: int) -> str:

        try:
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=page_index, to_page=page_index)
            buf: bytes = sub.tobytes()
            return pdfminer.high_level.extract_text(io.BytesIO(buf)) or ""
        except Exception:
            return ""
    page_number: int
    text: str = ""
    blocks: List[Any] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)  # Raw table data, không phải chunks
    warnings: List[str] = field(default_factory=list)
    source: Dict[str, Any] = field(default_factory=dict)
    normalized_tables: list = field(default_factory=list)

    def __post_init__(self):
        self.blocks = list(self.blocks) if isinstance(self.blocks, list) else []

    @staticmethod
    def from_fitz_page(
        page: Any,
        page_idx: int,
        doc: Any,
        file_path: str,
        doc_id: str,
        doc_title: str,
        page_labels: dict,
        extract_text: bool = True,
        extract_tables: bool = True,
    ) -> 'PDFPage':
        """
        Tạo PDFPage từ fitz page - chỉ load và parse dữ liệu, KHÔNG chunk.
        Tuân thủ Single Responsibility: loader chỉ load dữ liệu thô.
        """
        one_based = page_idx + 1
        page_label = page_labels.get(page_idx) if isinstance(page_labels, dict) else None

        # Extract text and blocks
        text_value = ""
        blocks = []
        warnings: List[str] = []
        
        if extract_text:
            try:
                # Get blocks first
                get_text = getattr(page, "get_text", None)
                if callable(get_text):
                    result = get_text("blocks")
                    blocks = result if isinstance(result, list) else []
                
                # Extract full text
                layout = PDFPage.extract_text_with_layout(page)
                text_value = layout.get("text", "") or ""
                
                if not text_value.strip():
                    # Fallback to pdfminer
                    text_value = PDFPage.pdfminer_extract_text_from_single_page(doc, page_idx)
                    if text_value.strip():
                        warnings.append("Used pdfminer fallback")
                    else:
                        warnings.append("No text found on page")
                        
            except Exception as e:
                warnings.append(f'Text extraction failed: {e}')

        # Extract raw tables (không chunk)
        raw_tables = []
        if extract_tables:
            try:
                # This would need to be implemented to extract raw table matrices
                # without chunking - to be implemented later
                pass
            except Exception as e:
                warnings.append(f'Table extraction failed: {e}')

        # Page metadata
        W, H = float(page.rect.width), float(page.rect.height)
        page_size = {"width": W, "height": H}

        return PDFPage(
            page_number=one_based,
            text=text_value,
            blocks=blocks,
            tables=raw_tables,  # Raw table data, chưa chunk
            warnings=warnings,
            source={
                "file_path": file_path, 
                "page_number": one_based, 
                "page_size": page_size,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "page_label": page_label
            },
        )


    def normalize(self, config: Optional[dict] = None) -> 'PDFPage':
        """
        Chuẩn hóa text, blocks, tables, warnings, source, normalized_tables.
        Trả về self (có thể chain).
        """
        # 1. Chuẩn hóa text
        if self.text:
            self.text = clean(ftfy.fix_text(self.text.strip()))
                # Tách câu bằng spaCy nếu text đủ dài
            if len(self.text) > 40:
                    try:
                        from loaders.normalizers.spacy_utils import sent_tokenize
                        self.sentences = sent_tokenize(self.text)
                    except Exception:
                        self.sentences = []
        # 2. Chuẩn hóa blocks
        if self.blocks:
            from .block import Block
            normalized_blocks = []
            for b in self.blocks:
                # Convert tuple/list to Block object if needed
                if isinstance(b, (tuple, list)) and len(b) >= 5:
                    # block format: (x0, y0, x1, y1, text, block_no, block_type)
                    block_obj = Block(
                        text=b[4] if len(b) > 4 else "",
                        bbox=(b[0], b[1], b[2], b[3]) if len(b) >= 4 else None,
                        metadata={"page_number": self.page_number}
                    )
                    block_obj.normalize(config=config)
                    normalized_blocks.append(block_obj)
                elif not isinstance(b, (list, tuple)) and hasattr(b, 'normalize') and callable(b.normalize):
                    # Đảm bảo block đã có page_number trong metadata
                    if hasattr(b, 'metadata') and isinstance(b.metadata, dict):
                        b.metadata['page_number'] = self.page_number
                    elif hasattr(b, 'metadata'):
                        b.metadata = {'page_number': self.page_number}
                    b.normalize(config=config)
                    normalized_blocks.append(b)
                else:
                    normalized_blocks.append(b)
            self.blocks = normalized_blocks
        # 3. Chuẩn hóa tables
        if self.tables:
            for t in self.tables:
                if hasattr(t, 'normalize') and callable(t.normalize):
                    t.normalize(config=config)
        # 4. Chuẩn hóa warnings
        if self.warnings:
            norm_warn = set()
            for w in self.warnings:
                w1 = clean(ftfy.fix_text(str(w)).strip())
                if w1:
                    norm_warn.add(w1)
            self.warnings = list(norm_warn)
        # 5. Chuẩn hóa source
        if self.source:
            self.source = {k: clean(ftfy.fix_text(str(v)).strip()) for k, v in self.source.items() if k}
        # 6. Chuẩn hóa normalized_tables
        if hasattr(self, 'normalized_tables') and self.normalized_tables:
            for t in self.normalized_tables:
                if hasattr(t, 'normalize') and callable(t.normalize):
                    t.normalize(config=config)
        return self
    