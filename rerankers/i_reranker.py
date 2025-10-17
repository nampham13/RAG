# ...existing code...
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class IReranker(ABC):
    """
    Interface cho các reranker.
    Mỗi reranker phải implement method `rerank`.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank `candidates` dựa trên `query`.

        Args:
            query: văn bản truy vấn
            candidates: list các candidate, mỗi phần tử ít nhất có 'id' và 'text'
            top_k: số kết quả trả về (sắp xếp giảm dần theo score)

        Returns:
            List các candidate đã bổ sung trường 'score' và sắp xếp theo score giảm dần.
        """
        raise NotImplementedError

    def fit(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Tùy chọn: precompute / build index cho candidates.
        Một số implementation (BM25, embedding-based) sẽ override.
        """
        return None