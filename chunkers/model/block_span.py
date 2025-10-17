"""
Block Span Model
================
Đại diện cho character offsets trong source blocks.
"""

from dataclasses import dataclass
from typing import Optional
from loaders.model.base import LoaderBaseModel


@dataclass
class BlockSpan(LoaderBaseModel):
    """
    Đại diện cho character offsets trong source blocks.
    Single Responsibility: Quản lý vị trí và phạm vi của text trong block nguồn.
    """
    block_id: str                    # ID của block nguồn
    start_char: int                  # Vị trí ký tự bắt đầu
    end_char: int                    # Vị trí ký tự kết thúc
    page_number: Optional[int] = None  # Số trang (nếu có)

    def length(self) -> int:
        """Độ dài của span"""
        return max(0, self.end_char - self.start_char)

    def overlaps(self, other: 'BlockSpan') -> bool:
        """Kiểm tra xem span này có overlap với span khác không"""
        if self.block_id != other.block_id:
            return False
        return not (self.end_char <= other.start_char or other.end_char <= self.start_char)

    def to_dict(self) -> dict:
        """Chuyển đổi sang dictionary"""
        return {
            'block_id': self.block_id,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'page_number': self.page_number
        }