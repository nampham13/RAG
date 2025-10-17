"""
Summary Generator - Document and Batch Summary Creation
=======================================================
Handles creation and persistence of document summaries and batch processing reports.
Single Responsibility: Summary generation and JSON file output.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generates and saves document summaries and batch processing reports.
    Single Responsibility: Summary creation and JSON persistence.
    """

    def __init__(self, metadata_dir: Path, output_dir: Path):
        """
        Initialize SummaryGenerator.

        Args:
            metadata_dir: Directory to store individual document summaries
            output_dir: Root output directory for batch summaries
        """
        self.metadata_dir = metadata_dir
        self.output_dir = output_dir
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def create_document_summary(self, pdf_doc, chunk_set, embeddings_data: List[Dict[str, Any]],
                               faiss_file: Path, metadata_map_file: Path) -> Dict[str, Any]:
        """
        Create lightweight document summary.

        Args:
            pdf_doc: Processed PDFDocument
            chunk_set: ChunkSet from chunking
            embeddings_data: List of embedding dictionaries
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file

        Returns:
            Document summary dictionary
        """
        return {
            "document": {
                "file_name": Path(pdf_doc.file_path).name,
                "file_path": pdf_doc.file_path,
                "pages": len(pdf_doc.pages),
                "processed_date": datetime.now().isoformat()
            },
            "processing": {
                "chunks": len(chunk_set.chunks),
                "tokens": chunk_set.total_tokens,
                "strategy": chunk_set.chunk_strategy,
                "embeddings": len(embeddings_data),
                "dimension": embeddings_data[0]["embedding_dimension"] if embeddings_data else 0,
                "model": embeddings_data[0]["embedding_model"] if embeddings_data else "unknown"
            },
            "files": {
                "faiss_index": str(faiss_file),
                "metadata_map": str(metadata_map_file)
            },
            "statistics": {
                "text_chunks": sum(1 for e in embeddings_data if not e["is_table"]),
                "table_chunks": sum(1 for e in embeddings_data if e["is_table"])
            }
        }

    def save_document_summary(self, summary: Dict[str, Any], file_name: str, timestamp: str) -> Path:
        """
        Save document summary to JSON file.

        Args:
            summary: Document summary dictionary
            file_name: Base filename
            timestamp: Timestamp string

        Returns:
            Path to saved summary file
        """
        summary_file = self.metadata_dir / f"{file_name}_summary_{timestamp}.json"

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved document summary: {summary_file.name}")
        return summary_file

    def create_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary of batch processing results.

        Args:
            results: List of individual processing results

        Returns:
            Batch summary dictionary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "total_chunks": sum(r.get("chunks", 0) for r in results if r.get("success", False)),
            "total_embeddings": sum(r.get("embeddings", 0) for r in results if r.get("success", False)),
            "results": results
        }

    def save_batch_summary(self, summary: Dict[str, Any]) -> Path:
        """
        Save batch summary to JSON file.

        Args:
            summary: Batch summary dictionary

        Returns:
            Path to saved batch summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"batch_summary_{timestamp}.json"

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Batch processing completed - Total: {summary['total_files']}, Successful: {summary['successful']}, Failed: {summary['failed']}")
        logger.info(f"Saved batch summary: {summary_file}")

        return summary_file