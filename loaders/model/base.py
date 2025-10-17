from abc import ABC
from dataclasses import asdict, is_dataclass
from typing import Any, Type, TypeVar

T = TypeVar('T', bound='LoaderBaseModel')

class LoaderBaseModel(ABC):
    """
    Base class cho các model của loader (PDFPage, TableSchema, ...).
    - Cung cấp to_dict chuẩn hóa cho dataclass hoặc object thường
    - validate: có thể override ở các class con
    - from_dict: khởi tạo từ dict (nếu cần)
    """
    def to_dict(self) -> dict:
        """Chuyển object thành dict (hỗ trợ dataclass)."""
        if is_dataclass(self):
            return asdict(self)
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """Tạo instance từ dict (hỗ trợ dataclass)."""
        if is_dataclass(cls):
            return cls(**data)
        obj = cls()
        obj.__dict__.update(data)
        return obj

    def validate(self) -> bool:
        """Kiểm tra hợp lệ, override ở class con nếu cần."""
        return True
    def normalize(self, *args, **kwargs) -> 'LoaderBaseModel':
        """
        Chuẩn hóa dữ liệu. Mặc định chưa triển khai, class con phải override nếu muốn dùng.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.normalize() chưa được triển khai.")
