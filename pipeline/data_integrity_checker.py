"""
Data Integrity Checker for RAG Pipeline
=======================================
Handles data integrity validation across all pipeline outputs.
"""

import json
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DataIntegrityChecker:
    """
    Handles data integrity checking across all pipeline outputs.
    Single Responsibility: Validate data consistency and integrity.
    """

    def __init__(self, chunks_dir: Path, embeddings_dir: Path, vectors_dir: Path, metadata_dir: Path):
        """
        Initialize Data Integrity Checker.

        Args:
            chunks_dir: Directory containing chunk files
            embeddings_dir: Directory containing embedding files
            vectors_dir: Directory containing FAISS index files
            metadata_dir: Directory containing metadata files
        """
        self.chunks_dir = chunks_dir
        self.embeddings_dir = embeddings_dir
        self.vectors_dir = vectors_dir
        self.metadata_dir = metadata_dir

    def check_data_integrity(self, pdf_path: str | Path,
                           chunks_file: Optional[str | Path] = None,
                           embeddings_file: Optional[str | Path] = None,
                           faiss_index_file: Optional[str | Path] = None,
                           metadata_file: Optional[str | Path] = None) -> Dict[str, Any]:
        """
        Check data integrity across all pipeline outputs.

        Args:
            pdf_path: Path to original PDF file
            chunks_file: Path to chunks file (auto-detect if None)
            embeddings_file: Path to embeddings file (auto-detect if None)
            faiss_index_file: Path to FAISS index file (auto-detect if None)
            metadata_file: Path to metadata file (auto-detect if None)

        Returns:
            Dict with integrity check results
        """
        pdf_path = Path(pdf_path)
        file_name = pdf_path.stem

        integrity_report = {
            "file_name": pdf_path.name,
            "checks": {},
            "overall_integrity": True,
            "issues": []
        }

        # Auto-detect file paths if not provided
        if chunks_file is None:
            chunks_file = self.chunks_dir / f"{file_name}_chunks_*.txt"
            chunks_files = list(self.chunks_dir.glob(f"{file_name}_chunks_*.txt"))
            chunks_file = chunks_files[-1] if chunks_files else None

        if embeddings_file is None:
            embeddings_file = self.embeddings_dir / f"{file_name}_embeddings_*.json"
            embeddings_files = list(self.embeddings_dir.glob(f"{file_name}_embeddings_*.json"))
            embeddings_file = embeddings_files[-1] if embeddings_files else None

        if faiss_index_file is None:
            faiss_index_file = self.vectors_dir / f"{file_name}_vectors_*.faiss"
            faiss_files = list(self.vectors_dir.glob(f"{file_name}_vectors_*.faiss"))
            faiss_index_file = faiss_files[-1] if faiss_files else None

        if metadata_file is None:
            metadata_file = self.vectors_dir / f"{file_name}_metadata_map_*.pkl"
            metadata_files = list(self.vectors_dir.glob(f"{file_name}_metadata_map_*.pkl"))
            metadata_file = metadata_files[-1] if metadata_files else None

        # Check 1: File existence
        integrity_report["checks"]["file_existence"] = {
            "chunks_file": chunks_file.exists() if chunks_file and isinstance(chunks_file, Path) else False,
            "embeddings_file": embeddings_file.exists() if embeddings_file and isinstance(embeddings_file, Path) else False,
            "faiss_index_file": faiss_index_file.exists() if faiss_index_file and isinstance(faiss_index_file, Path) else False,
            "metadata_file": metadata_file.exists() if metadata_file and isinstance(metadata_file, Path) else False
        }

        if not all(integrity_report["checks"]["file_existence"].values()):
            integrity_report["overall_integrity"] = False
            integrity_report["issues"].append("Missing output files")
            return integrity_report

        # Check 2: Load and parse chunks
        try:
            chunks_data = []
            if chunks_file and isinstance(chunks_file, Path):
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Parse chunks from text file - split by separator with newlines
                    separator = '=' * 80 + '\n\n'
                    chunk_blocks = content.split(separator)[1:]  # Skip header

                    for block in chunk_blocks:
                        if not block.strip():  # Skip empty blocks
                            continue

                        if 'CHUNK' in block:
                            lines = block.strip().split('\n')
                            chunk_info = {}
                            text_start = 0
                            for i, line in enumerate(lines):
                                if line.startswith('CHUNK'):
                                    parts = line.split('|')
                                    if len(parts) >= 1:
                                        chunk_info['chunk_id'] = parts[0].split(':')[1].strip()
                                    if len(parts) >= 2 and 'Page:' in parts[1]:
                                        try:
                                            chunk_info['page'] = int(parts[1].split(':')[1].strip())
                                        except (ValueError, IndexError):
                                            chunk_info['page'] = None
                                    if len(parts) >= 3 and 'Tokens:' in parts[2]:
                                        try:
                                            chunk_info['tokens'] = int(parts[2].split(':')[1].strip())
                                        except (ValueError, IndexError):
                                            chunk_info['tokens'] = None
                                    if len(parts) >= 4 and 'Type:' in parts[3]:
                                        chunk_info['type'] = parts[3].split(':')[1].strip()
                                elif line.startswith('-' * 40):
                                    text_start = i + 1
                                    break
                            if text_start > 0 and text_start < len(lines):
                                chunk_info['text'] = '\n'.join(lines[text_start:]).strip()
                                chunks_data.append(chunk_info)

            integrity_report["checks"]["chunks_parsing"] = {
                "success": True,
                "chunk_count": len(chunks_data),
                "chunks_with_text": sum(1 for c in chunks_data if c.get('text'))
            }

        except Exception as e:
            integrity_report["checks"]["chunks_parsing"] = {
                "success": False,
                "error": str(e)
            }
            integrity_report["overall_integrity"] = False
            integrity_report["issues"].append(f"Chunks parsing failed: {e}")
            return integrity_report

        # Check 3: Load and validate embeddings
        try:
            embeddings_data = []
            if embeddings_file and isinstance(embeddings_file, Path):
                with open(embeddings_file, 'r', encoding='utf-8') as f:
                    embeddings_data = json.load(f)

            integrity_report["checks"]["embeddings_loading"] = {
                "success": True,
                "embedding_count": len(embeddings_data),
                "embedding_dimension": len(embeddings_data[0]["embedding"]) if embeddings_data else 0,
                "all_have_embeddings": all("embedding" in e for e in embeddings_data)
            }

        except Exception as e:
            integrity_report["checks"]["embeddings_loading"] = {
                "success": False,
                "error": str(e)
            }
            integrity_report["overall_integrity"] = False
            integrity_report["issues"].append(f"Embeddings loading failed: {e}")
            return integrity_report

        # Check 4: Cross-reference chunks vs embeddings
        chunk_ids_from_chunks = {c['chunk_id'] for c in chunks_data}
        chunk_ids_from_embeddings = {e['chunk_id'] for e in embeddings_data}

        integrity_report["checks"]["chunk_embedding_consistency"] = {
            "chunks_in_chunks_file": len(chunk_ids_from_chunks),
            "chunks_in_embeddings_file": len(chunk_ids_from_embeddings),
            "chunks_in_both": len(chunk_ids_from_chunks & chunk_ids_from_embeddings),
            "chunks_only_in_chunks": len(chunk_ids_from_chunks - chunk_ids_from_embeddings),
            "chunks_only_in_embeddings": len(chunk_ids_from_embeddings - chunk_ids_from_chunks),
            "perfect_match": chunk_ids_from_chunks == chunk_ids_from_embeddings
        }

        if not integrity_report["checks"]["chunk_embedding_consistency"]["perfect_match"]:
            integrity_report["overall_integrity"] = False
            integrity_report["issues"].append("Chunk-embedding mismatch detected")

        # Check 5: FAISS index validation
        try:
            import faiss
            index = faiss.read_index(str(faiss_index_file))

            integrity_report["checks"]["faiss_index"] = {
                "success": True,
                "vector_count": index.ntotal,
                "dimension": index.d,
                "matches_embeddings": index.ntotal == len(embeddings_data)
            }

            if index.ntotal != len(embeddings_data):
                integrity_report["overall_integrity"] = False
                integrity_report["issues"].append(f"FAISS index vector count ({index.ntotal}) != embeddings count ({len(embeddings_data)})")

        except Exception as e:
            integrity_report["checks"]["faiss_index"] = {
                "success": False,
                "error": str(e)
            }
            integrity_report["overall_integrity"] = False
            integrity_report["issues"].append(f"FAISS index validation failed: {e}")

        # Check 6: Metadata validation
        try:
            import pickle
            if metadata_file and isinstance(metadata_file, Path):
                with open(metadata_file, 'rb') as f:
                    metadata_map = pickle.load(f)

                integrity_report["checks"]["metadata_validation"] = {
                    "success": True,
                    "metadata_entries": len(metadata_map),
                    "matches_embeddings": len(metadata_map) == len(embeddings_data)
                }

                if len(metadata_map) != len(embeddings_data):
                    integrity_report["overall_integrity"] = False
                    integrity_report["issues"].append(f"Metadata count ({len(metadata_map)}) doesn't match embeddings count ({len(embeddings_data)})")
            else:
                integrity_report["checks"]["metadata_validation"] = {
                    "success": False,
                    "error": "metadata_file is None or not a Path"
                }
                integrity_report["overall_integrity"] = False
                integrity_report["issues"].append("Metadata file not available")

        except Exception as e:
            integrity_report["checks"]["metadata_validation"] = {
                "success": False,
                "error": str(e)
            }
            integrity_report["overall_integrity"] = False
            integrity_report["issues"].append(f"Metadata validation failed: {e}")

        # Check 7: Content integrity (hash verification)
        content_hashes = []
        for chunk in chunks_data:
            if 'text' in chunk:
                content_hash = hashlib.md5(chunk['text'].encode('utf-8')).hexdigest()
                content_hashes.append(content_hash)

        embedding_hashes = []
        for embedding in embeddings_data:
            if 'text' in embedding:
                embedding_hash = hashlib.md5(embedding['text'].encode('utf-8')).hexdigest()
                embedding_hashes.append(embedding_hash)

        integrity_report["checks"]["content_integrity"] = {
            "chunk_content_hashes": len(content_hashes),
            "embedding_content_hashes": len(embedding_hashes),
            "content_matches": content_hashes == embedding_hashes,
            "unique_chunks": len(set(content_hashes)),
            "duplicate_chunks": len(content_hashes) - len(set(content_hashes))
        }

        if not integrity_report["checks"]["content_integrity"]["content_matches"]:
            integrity_report["overall_integrity"] = False
            integrity_report["issues"].append("Content mismatch between chunks and embeddings")

        # Summary
        integrity_report["summary"] = {
            "total_checks": len(integrity_report["checks"]),
            "passed_checks": sum(1 for check in integrity_report["checks"].values()
                               if isinstance(check, dict) and check.get("success", True)),
            "integrity_score": len(integrity_report["issues"]) == 0
        }

        logger.info(f"Data integrity check complete. Integrity: {integrity_report['overall_integrity']}")
        return integrity_report