import hashlib
import re
import ftfy
from dataclasses import dataclass, field
from typing import Any, Optional, Dict

from .table import TableSchema
from .base import LoaderBaseModel

def calc_stable_id(doc_id: str, page_number: int, norm_text: str, bbox: Any) -> str:
    # Băm dựa trên doc_id, page_number, text[:N], bbox (làm tròn)
    N = 32
    text_part = norm_text[:N] if isinstance(norm_text, str) else ""
    bbox_part = tuple(round(float(x), 2) for x in bbox) if bbox else ()
    raw = f"{doc_id}|{page_number}|{text_part}|{bbox_part}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

@dataclass
class Block(LoaderBaseModel):
    
    text: str = ""
    bbox: Any = None
    text_source: Optional[str] = None
    stable_id: Optional[str] = None
    content_sha256: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def validate(self):
        # Có thể mở rộng kiểm tra, hiện tại chỉ trả về True để khớp base
        return True

    def set_stable_id_and_hash(self, doc_id: str, page_number: int):
        self.stable_id = calc_stable_id(doc_id, page_number, self.text, self.bbox)
        self.content_sha256 = hashlib.sha256((self.text or "").encode("utf-8")).hexdigest()
        
    def normalize(self, config: Optional[dict] = None) -> 'Block':
        """
        Chuẩn hóa text (dùng clean-text, ftfy), bbox, metadata, tính lại stable_id và content_sha256.
        CHỈ chuẩn hóa dữ liệu, KHÔNG lọc tại đây.
        Filtering sẽ được thực hiện ở tầng document sau khi thu thập block_hash_counter.
        Trả về self (có thể chain).
        """
        if config is None:
            config = {}
        
        # 1. Chuẩn hóa text
        if self.text:
            text = self.text.strip()
            
            # Fix unicode và encoding issues
            text = ftfy.fix_text(text)
            
            # REMOVED: Clean text using cleantext library (was corrupting text)
            # text = clean(text)
            
            # De-hyphenation: nối từ bị ngắt dòng bởi dấu gạch ngang cuối dòng
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            
            # Normalize line breaks and control characters
            text = re.sub(r"[\r\f]+", "\n", text)  # Convert \r, \f to \n
            
            # Remove zero-width characters and invisible Unicode
            text = re.sub(r"[\u200b\u200c\u200d\ufeff]+", "", text)
            
            # Remove TOC dots (e.g., "1. INTRODUCTION ....................")
            text = re.sub(r'\.{3,}', '', text)
            
            # Normalize whitespace (multiple spaces -> single, multiple newlines -> max 2)
            from loaders.normalizers.block_utils import normalize_whitespace
            text = normalize_whitespace(text)
            
            self.text = text
            
            # Tách câu bằng spaCy nếu text đủ dài
            if len(self.text) > 40:
                try:
                    from loaders.normalizers.spacy_utils import sent_tokenize
                    self.sentences = sent_tokenize(self.text)
                except Exception:
                    self.sentences = []
        
        # 2. Chuẩn hóa bbox
        if self.bbox and isinstance(self.bbox, (tuple, list)) and len(self.bbox) == 4:
            self.bbox = tuple(round(float(x), 2) for x in self.bbox)
        
        # 3. Chuẩn hóa metadata
        if self.metadata:
            self.metadata = {k: v for k, v in self.metadata.items() if k and v is not None}
        
        # 4. Tính lại stable_id và content_sha256 nếu có doc_id, page_number trong metadata
        doc_id = self.metadata.get('doc_id') if self.metadata else None
        page_number = self.metadata.get('page_number') if self.metadata else None
        if doc_id is not None and page_number is not None:
            self.set_stable_id_and_hash(doc_id, page_number)
        else:
            self.content_sha256 = hashlib.sha256((self.text or "").encode("utf-8")).hexdigest()
        
        return self

@dataclass
class TableBlock(Block):
    """
    Block đại diện cho bảng. Kế thừa từ Block và có thêm table data.
    """
    table: Optional['TableSchema'] = None
    block_type: str = "table"

    def normalize(self, config: Optional[dict] = None) -> 'TableBlock':
        """
        Chuẩn hóa TableBlock: gọi normalize của Block và normalize table.
        """
        # Gọi normalize của Block trước
        super().normalize(config)
        
        # Normalize table nếu có
        if self.table and hasattr(self.table, 'normalize'):
            self.table.normalize(config)
        
        return self
