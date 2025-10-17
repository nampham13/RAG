"""
Score Model
===========
Metrics đánh giá chất lượng của chunk.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from loaders.model.base import LoaderBaseModel


@dataclass
class Score(LoaderBaseModel):
    """
    Metrics đánh giá chất lượng của chunk.
    Single Responsibility: Đo lường và đánh giá chất lượng chunk.
    """
    coherence_score: float = 0.0      # Điểm coherence ngữ nghĩa (0-1)
    completeness_score: float = 0.0   # Độ hoàn chỉnh của thông tin (0-1)
    token_ratio: float = 0.0          # Tỷ lệ token_count/max_tokens
    overlap_ratio: float = 0.0        # Tỷ lệ overlap với chunk trước (nếu có)
    structural_integrity: float = 0.0  # Điểm toàn vẹn cấu trúc (0-1)

    def overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Tính điểm tổng thể dựa trên trọng số.
        Mặc định: coherence=0.3, completeness=0.3, token_ratio=0.2, structural=0.2
        """
        if weights is None:
            weights = {
                'coherence': 0.3,
                'completeness': 0.3,
                'token_ratio': 0.2,
                'structural': 0.2
            }

        score = (
            self.coherence_score * weights.get('coherence', 0.0) +
            self.completeness_score * weights.get('completeness', 0.0) +
            self.token_ratio * weights.get('token_ratio', 0.0) +
            self.structural_integrity * weights.get('structural', 0.0)
        )
        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict:
        """Chuyển đổi sang dictionary"""
        return {
            'coherence_score': self.coherence_score,
            'completeness_score': self.completeness_score,
            'token_ratio': self.token_ratio,
            'overlap_ratio': self.overlap_ratio,
            'structural_integrity': self.structural_integrity,
            'overall_score': self.overall_score()
        }