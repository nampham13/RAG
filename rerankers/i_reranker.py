"""
Reranker Interface
==================
Abstract interface cho tất cả reranking providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class IReranker(ABC):
    """
    Interface cho reranking providers.
    Single Responsibility: Define contract cho reranking operations.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
        text_key: str = "text"
    ) -> List[Dict]:
        """
        Re-rank candidates theo relevance với query.

        Args:
            query: Query string
            candidates: List của candidate documents
            top_k: Number of top results to return (None = all)
            text_key: Key trong dict chứa text content

        Returns:
            List[Dict]: Reranked candidates với rerank_score
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connection tới reranking service.

        Returns:
            bool: True nếu connection thành công
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Lấy tên model đang sử dụng.

        Returns:
            str: Model name
        """
        pass