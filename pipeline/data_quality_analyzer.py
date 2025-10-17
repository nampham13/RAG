"""
Data Quality Analyzer for RAG Pipeline
======================================
Handles data quality analysis, reporting, and comparison functionality.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

from loaders.pdf_loader import PDFLoader
from chunkers.hybrid_chunker import HybridChunker
from embedders.i_embedder import IEmbedder

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """
    Handles data quality analysis throughout the RAG pipeline.
    Single Responsibility: Analyze and report data quality metrics.
    """

    def __init__(self, loader: PDFLoader, chunker: HybridChunker, embedder: IEmbedder, output_dir: Path):
        """
        Initialize Data Quality Analyzer.

        Args:
            loader: PDF loader instance
            chunker: Document chunker instance
            embedder: Embedding model instance
            output_dir: Base output directory
        """
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.output_dir = output_dir

    def analyze_data_quality(self, pdf_path: str | Path) -> Dict[str, Any]:
        """
        Analyze data quality throughout the RAG pipeline.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with quality analysis results
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Analyzing data quality for: {pdf_path.name}")

        results = {
            "file_name": pdf_path.name,
            "stages": {},
            "metrics": {},
            "quality_score": 0.0
        }

        # Stage 1: Raw PDF extraction
        logger.info("Stage 1: Raw PDF extraction...")
        pdf_doc = self.loader.load(str(pdf_path))

        raw_text = ""
        raw_tables = []
        for page in pdf_doc.pages:
            for block in page.blocks:
                if hasattr(block, 'text') and block.text:
                    raw_text += block.text + "\n"
                if hasattr(block, 'table') and block.table:
                    raw_tables.append(block.table)

        results["stages"]["raw_extraction"] = {
            "total_pages": len(pdf_doc.pages),
            "total_blocks": sum(len(p.blocks) for p in pdf_doc.pages),
            "raw_text_length": len(raw_text),
            "raw_table_count": len(raw_tables),
            "raw_text_sample": raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        }

        # Stage 2: After normalization
        logger.info("Stage 2: After normalization...")
        pdf_doc_normalized = pdf_doc.normalize()

        normalized_text = ""
        normalized_tables = []
        for page in pdf_doc_normalized.pages:
            for block in page.blocks:
                if hasattr(block, 'text') and block.text:
                    normalized_text += block.text + "\n"
                if hasattr(block, 'table') and block.table:
                    normalized_tables.append(block.table)

        results["stages"]["normalized"] = {
            "normalized_text_length": len(normalized_text),
            "normalized_table_count": len(normalized_tables),
            "text_preservation_ratio": len(normalized_text) / len(raw_text) if raw_text else 0,
            "normalized_text_sample": normalized_text[:500] + "..." if len(normalized_text) > 500 else normalized_text
        }

        # Stage 3: After chunking
        logger.info("Stage 3: After chunking...")
        chunk_set = self.chunker.chunk(pdf_doc_normalized)

        chunk_texts = [chunk.text for chunk in chunk_set.chunks]
        chunk_lengths = [len(chunk.text) for chunk in chunk_set.chunks]
        token_counts = [chunk.token_count for chunk in chunk_set.chunks]

        results["stages"]["chunking"] = {
            "total_chunks": len(chunk_set.chunks),
            "chunk_strategy": chunk_set.chunk_strategy,
            "total_tokens": chunk_set.total_tokens,
            "avg_chunk_length": np.mean(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "avg_tokens_per_chunk": np.mean(token_counts),
            "chunk_text_sample": chunk_texts[0][:300] + "..." if chunk_texts else ""
        }

        # Stage 4: After embedding
        logger.info("Stage 4: After embedding...")
        embeddings_data = []
        vector_qualities = []

        for chunk in chunk_set.chunks:
            try:
                embedding = self.embedder.embed(chunk.text)
                embeddings_data.append({
                    "chunk_id": chunk.chunk_id,
                    "embedding": embedding,
                    "text_length": len(chunk.text),
                    "token_count": chunk.token_count
                })

                # Calculate vector quality metrics
                vector_norm = np.linalg.norm(embedding)
                vector_std = np.std(embedding)
                vector_qualities.append({
                    "norm": vector_norm,
                    "std": vector_std,
                    "sparsity": np.count_nonzero(embedding) / len(embedding)
                })

            except Exception as e:
                logger.warning(f"Failed to embed chunk {chunk.chunk_id}: {e}")

        if vector_qualities:
            results["stages"]["embedding"] = {
                "total_embeddings": len(embeddings_data),
                "embedding_dimension": len(embeddings_data[0]["embedding"]) if embeddings_data else 0,
                "avg_vector_norm": np.mean([vq["norm"] for vq in vector_qualities]),
                "avg_vector_std": np.mean([vq["std"] for vq in vector_qualities]),
                "avg_sparsity": np.mean([vq["sparsity"] for vq in vector_qualities]),
                "vector_quality_sample": {
                    "norm": vector_qualities[0]["norm"],
                    "std": vector_qualities[0]["std"],
                    "sparsity": vector_qualities[0]["sparsity"]
                }
            }

        # Overall quality metrics
        results["metrics"] = {
            "text_preservation": results["stages"]["normalized"]["text_preservation_ratio"],
            "chunking_efficiency": len(chunk_texts) / len(normalized_text) * 1000 if normalized_text else 0,  # chunks per 1000 chars
            "embedding_coverage": len(embeddings_data) / len(chunk_set.chunks) if chunk_set.chunks else 0,
            "avg_information_density": np.mean(token_counts) / np.mean(chunk_lengths) if chunk_lengths else 0
        }

        # Calculate overall quality score (0-100)
        preservation_score = min(100, results["metrics"]["text_preservation"] * 100)
        coverage_score = results["metrics"]["embedding_coverage"] * 100
        density_score = min(100, results["metrics"]["avg_information_density"] * 1000)

        results["quality_score"] = (preservation_score * 0.4 + coverage_score * 0.4 + density_score * 0.2)

        logger.info(f"Data quality analysis complete. Score: {results['quality_score']:.1f}/100")
        return results

    def save_quality_report(self, quality_report: Dict[str, Any], output_path: Optional[str | Path] = None) -> Path:
        """
        Save quality analysis report to JSON file.

        Args:
            quality_report: Quality analysis results
            output_path: Optional output path, defaults to data/quality_reports/

        Returns:
            Path to saved report file
        """
        if output_path is None:
            quality_dir = self.output_dir / "quality_reports"
            quality_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = quality_dir / f"{quality_report['file_name']}_quality_{timestamp}.json"
        else:
            output_path = Path(output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved quality report to: {output_path}")
        return output_path

    def compare_quality_reports(self, report1: Dict[str, Any], report2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two quality reports and show differences.

        Args:
            report1: First quality report
            report2: Second quality report

        Returns:
            Dict with comparison results
        """
        comparison = {
            "files": [report1["file_name"], report2["file_name"]],
            "quality_scores": [report1["quality_score"], report2["quality_score"]],
            "score_difference": report2["quality_score"] - report1["quality_score"],
            "stage_differences": {}
        }

        # Compare each stage
        for stage in ["raw_extraction", "normalized", "chunking", "embedding"]:
            if stage in report1["stages"] and stage in report2["stages"]:
                stage1 = report1["stages"][stage]
                stage2 = report2["stages"][stage]

                differences = {}
                for key in stage1.keys():
                    if key in stage2 and isinstance(stage1[key], (int, float)) and isinstance(stage2[key], (int, float)):
                        diff = stage2[key] - stage1[key]
                        if abs(diff) > 0.01:  # Only show meaningful differences
                            differences[key] = {
                                "value1": stage1[key],
                                "value2": stage2[key],
                                "difference": diff,
                                "percent_change": (diff / stage1[key] * 100) if stage1[key] != 0 else 0
                            }

                if differences:
                    comparison["stage_differences"][stage] = differences

        return comparison