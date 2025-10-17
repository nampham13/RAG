"""
Provenance Aggregation Model
===========================
Tổng hợp thông tin provenance từ các blocks đóng góp.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from .block_span import BlockSpan
from loaders.model.base import LoaderBaseModel


@dataclass
class ProvenanceAgg(LoaderBaseModel):
    """
    Tổng hợp thông tin provenance từ tất cả các blocks đóng góp.
    Single Responsibility: Theo dõi nguồn gốc và metadata của chunk.
    """
    source_blocks: List[str] = field(default_factory=list)  # List các block IDs
    spans: List[BlockSpan] = field(default_factory=list)    # Character spans trong blocks
    page_numbers: Set[int] = field(default_factory=set)     # Tập các page numbers
    doc_id: Optional[str] = None                             # Document ID
    file_path: Optional[str] = None                          # Đường dẫn file nguồn
    metadata: Dict[str, Any] = field(default_factory=dict)   # Metadata bổ sung

    def add_span(self, span: BlockSpan):
        """Thêm một span vào provenance"""
        self.spans.append(span)
        if span.block_id not in self.source_blocks:
            self.source_blocks.append(span.block_id)
        if span.page_number is not None:
            self.page_numbers.add(span.page_number)

    def total_source_chars(self) -> int:
        """Tổng số ký tự từ tất cả spans"""
        return sum(span.length() for span in self.spans)

    def page_range(self) -> Optional[tuple]:
        """Trả về (min_page, max_page) hoặc None"""
        if not self.page_numbers:
            return None
        return (min(self.page_numbers), max(self.page_numbers))

    def to_dict(self) -> dict:
        """Chuyển đổi sang dictionary"""
        return {
            'source_blocks': self.source_blocks,
            'spans': [span.to_dict() for span in self.spans],
            'page_numbers': sorted(list(self.page_numbers)),
            'doc_id': self.doc_id,
            'file_path': self.file_path,
            'metadata': self.metadata
        }