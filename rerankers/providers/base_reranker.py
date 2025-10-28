"""
Base Reranker
=============
Base class cho tất cả reranking providers.
"""

import logging
from typing import List, Dict, Optional
from abc import ABC

from ..i_reranker import IReranker
from ..model.reranker_profile import RerankerProfile

logger = logging.getLogger(__name__)


class BaseReranker(IReranker, ABC):
    """
    Base class cho tất cả reranking providers.
    Single Responsibility: Cung cấp common functionality cho rerankers.
    """

    def __init__(self, profile: RerankerProfile):
        """
        Initialize base reranker.

        Args:
            profile: Reranker profile configuration
        """
        self.profile = profile
        self._validate_profile()

    def _validate_profile(self):
        """Validate reranker profile"""
        if not self.profile.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.profile.provider:
            raise ValueError("provider cannot be empty")

    @property
    def model_name(self) -> str:
        """
        Lấy tên model đang sử dụng.

        Returns:
            str: Model name
        """
        return self.profile.model_id

    def _extract_text(self, candidate: Dict, text_key: str) -> str:
        """
        Extract text from candidate dict.

        Args:
            candidate: Candidate dictionary
            text_key: Primary key to look for text

        Returns:
            str: Extracted text
        """
        # Try primary key first
        text = candidate.get(text_key)
        if text:
            return str(text)

        # Fallback to common keys
        for key in ["snippet", "text", "content", "body"]:
            text = candidate.get(key)
            if text:
                return str(text)

        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.profile.model_id})"