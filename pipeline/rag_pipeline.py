"""
RAG Pipeline - Complete Implementation
========================================
Pipeline hoàn chỉnh: PDF -> Chunks -> Embeddings -> Vector Storage -> Retrieval

Output: Tất cả dữ liệu được lưu vào data folder
"""

import json
import logging
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from loaders.pdf_loader import PDFLoader
from chunkers.hybrid_chunker import HybridChunker
from embedders.embedder_factory import EmbedderFactory
from embedders.providers.ollama import OllamaModelSwitcher, OllamaModelType
from pipeline.vector_store import VectorStore
from pipeline.summary_generator import SummaryGenerator
from pipeline.retriever import Retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG Pipeline implementation.
    Single Responsibility: Orchestrate full PDF → Vector Storage workflow.
    """
    
    def __init__(self, 
                 output_dir: str = "data",
                 pdf_dir: Optional[str | Path] = None,
                 model_type: OllamaModelType = OllamaModelType.GEMMA):
        """
        Initialize RAG Pipeline.
        
        Args:
            output_dir: Directory để lưu output files
            pdf_dir: Directory chứa PDF files (default: output_dir/pdf)
            model_type: Ollama model type (GEMMA hoặc BGE_M3)
        """
        self.output_dir = Path(output_dir)
        self.pdf_dir = Path(pdf_dir) if pdf_dir else self.output_dir / "pdf"
        self.model_type = model_type
        
        # Create output subdirectories
        self.chunks_dir = self.output_dir / "chunks"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.vectors_dir = self.output_dir / "vectors"
        self.metadata_dir = self.output_dir / "metadata"
        self.cache_dir = self.output_dir / "cache"
        
        # Create all directories
        for directory in [self.chunks_dir, self.embeddings_dir, 
                         self.vectors_dir, self.metadata_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache for processed chunks
        self.processed_chunks_cache = self.cache_dir / "processed_chunks.json"
        
        # Initialize components
        logger.info("Initializing RAG Pipeline...")
        self.loader = PDFLoader.create_default()
        self.chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
        
        # Initialize embedder with model switcher
        self.model_switcher = OllamaModelSwitcher()
        if model_type == OllamaModelType.GEMMA:
            self.embedder = self.model_switcher.switch_to_gemma()
        else:
            self.embedder = self.model_switcher.switch_to_bge_m3()
        
        # Initialize supporting components
        self.vector_store = VectorStore(self.vectors_dir)
        self.summary_generator = SummaryGenerator(self.metadata_dir, self.output_dir)
        self.retriever = Retriever(self.embedder)
    
    def switch_model(self, model_type: OllamaModelType) -> None:
        """
        Switch the embedding model.
        
        Args:
            model_type: New model type to switch to
        """
        if model_type == OllamaModelType.GEMMA:
            self.embedder = self.model_switcher.switch_to_gemma()
        else:
            self.embedder = self.model_switcher.switch_to_bge_m3()
        
        self.model_type = model_type
        logger.info(f"Switched to model: {self.embedder.profile.model_id}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the current pipeline configuration.
        
        Returns:
            Dict with pipeline information
        """
        return {
            "output_dir": str(self.output_dir),
            "pdf_dir": str(self.pdf_dir),
            "model_type": self.model_type.value,
            "embedder_model": self.embedder.profile.model_id,
            "embedder_dimension": self.embedder.dimension,
            "loader": "PDFLoader",
            "chunker": str(self.chunker),
            "vector_store": "FAISS",
            "cache_enabled": True
        }
    
    def process_pdf(self, pdf_path: str | Path, chunk_callback=None) -> Dict[str, Any]:
        """
        Process single PDF through complete pipeline.
        
        Args:
            pdf_path: Path to PDF file (str or Path)
            chunk_callback: Optional callback function(current, total) for progress
            
        Returns:
            Dict with processing results and file paths
        """
        pdf_path = Path(pdf_path)
        file_name = pdf_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Step 1: Load PDF
        logger.info("Loading PDF...")
        pdf_doc = self.loader.load(str(pdf_path))
        pdf_doc = pdf_doc.normalize()
        logger.info(f"Loaded {len(pdf_doc.pages)} pages, {sum(len(p.blocks) for p in pdf_doc.pages)} blocks")
        
        # Step 2: Chunk document
        logger.info("Chunking document...")
        chunk_set = self.chunker.chunk(pdf_doc)
        logger.info(f"Created {len(chunk_set.chunks)} chunks, strategy: {chunk_set.chunk_strategy}, tokens: {chunk_set.total_tokens}")
        
        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")
        embeddings_data = []
        skipped_chunks = 0
        total_chunks = len(chunk_set.chunks)
        
        # Call callback with initial state
        if chunk_callback:
            chunk_callback(0, total_chunks)
        
        for idx, chunk in enumerate(chunk_set.chunks, 1):
            # Create content hash for duplicate checking
            content_hash = hashlib.md5(chunk.text.encode('utf-8')).hexdigest()
            
            # Test connection on first chunk
            if idx == 1 and not self.embedder.test_connection():
                raise ConnectionError("Cannot connect to Ollama server!")
            
            try:
                embedding = self.embedder.embed(chunk.text)
            except Exception as e:
                logger.warning(f"Error embedding chunk {idx}: {e}")
                embedding = [0.0] * self.embedder.dimension
            
            # Prepare embedding data với full metadata
            chunk_embedding = {
                "chunk_id": chunk.chunk_id,
                "chunk_index": idx - 1,
                "text": chunk.text,
                "text_length": len(chunk.text),
                "token_count": chunk.token_count,
                
                # Embedding vector
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "embedding_model": self.embedder.profile.model_id,
                
                # Source metadata
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "page_number": list(chunk.provenance.page_numbers)[0] if chunk.provenance and chunk.provenance.page_numbers else None,
                "page_numbers": sorted(list(chunk.provenance.page_numbers)) if chunk.provenance else [],
                
                # Block tracing
                "block_type": chunk.metadata.get("block_type") or chunk.metadata.get("type"),
                "block_ids": chunk.provenance.source_blocks if chunk.provenance else [],
                
                # Table detection
                "is_table": chunk.metadata.get("block_type") == "table",
                
                # Provenance
                "provenance": {
                    "source_file": str(pdf_path),
                    "extraction_method": "PDFLoader",
                    "chunking_strategy": chunk_set.chunk_strategy or "unknown",
                    "embedding_model": self.embedder.profile.model_id,
                    "timestamp": timestamp
                }
            }
            
            # Add table data if applicable
            if chunk_embedding["is_table"]:
                table_payload = chunk.metadata.get("table_payload")
                if table_payload:
                    chunk_embedding["table_data"] = {
                        "table_id": getattr(table_payload, "id", None),
                        "header": getattr(table_payload, "header", []),
                        "num_rows": len(getattr(table_payload, "rows", [])),
                        "page_number": getattr(table_payload, "page_number", None)
                    }
            
            embeddings_data.append(chunk_embedding)

            # Update progress via callback
            if chunk_callback:
                chunk_callback(idx, total_chunks)
            
            if (idx - skipped_chunks) % 10 == 0:
                logger.info(f"Processed {idx}/{len(chunk_set.chunks)} chunks ({skipped_chunks} skipped)...")
        
        logger.info(f"Generated {len(embeddings_data)} embeddings ({skipped_chunks} chunks skipped)")
        
        # Step 3.5: Save chunks and embeddings to files
        logger.info("Saving chunks and embeddings...")
        
        # Save chunks as simple text file (for debugging)
        chunks_file = self.chunks_dir / f"{file_name}_chunks_{timestamp}.txt"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            f.write(f'Document: {pdf_path.name}\n')
            f.write(f'Total chunks: {len(chunk_set.chunks)}\n')
            f.write(f'New chunks processed: {len(embeddings_data)}\n')
            f.write(f'Skipped chunks: {skipped_chunks}\n')
            f.write(f'Timestamp: {timestamp}\n')
            f.write('=' * 80 + '\n\n')
            
            for i, chunk in enumerate(chunk_set.chunks, 1):
                # Only write chunks that were actually processed (have embeddings)
                if any(e['chunk_id'] == chunk.chunk_id for e in embeddings_data):
                    f.write(f'CHUNK {i}: {chunk.chunk_id}\n')
                    page = list(chunk.provenance.page_numbers)[0] if chunk.provenance and chunk.provenance.page_numbers else 'N/A'
                    f.write(f'Page: {page} | Tokens: {chunk.token_count} | Type: {chunk.chunk_type.value}\n')
                    f.write('-' * 40 + '\n')
                    f.write(chunk.text.strip())
                    f.write('\n\n' + '=' * 80 + '\n\n')
        
        # Save embeddings
        embeddings_file = self.embeddings_dir / f"{file_name}_embeddings_{timestamp}.json"
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved chunks to: {chunks_file}")
        logger.info(f"Saved embeddings to: {embeddings_file}")
        
        # Step 4: Create FAISS index and save (only if we have new embeddings)
        if embeddings_data:
            logger.info("Creating FAISS vector index...")
            faiss_file, metadata_map_file = self.vector_store.create_index(
                embeddings_data, file_name, timestamp
            )
        else:
            logger.info("No new embeddings - skipping FAISS index creation")
            # Use placeholder paths for skipped processing
            faiss_file = self.vectors_dir / f"{file_name}_vectors_{timestamp}.faiss"
            metadata_map_file = self.vectors_dir / f"{file_name}_metadata_map_{timestamp}.pkl"
            # Create empty files to indicate processing was attempted
            faiss_file.touch()
            metadata_map_file.touch()
        
        # Step 5: Save document summary (lightweight)
        logger.info("Creating document summary...")
        summary = self.summary_generator.create_document_summary(
            pdf_doc, chunk_set, embeddings_data, faiss_file, metadata_map_file
        )
        
        summary_file = self.summary_generator.save_document_summary(
            summary, file_name, timestamp
        )
        
        # Summary
        logger.info(f"Pipeline completed - Pages: {len(pdf_doc.pages)}, Chunks: {len(chunk_set.chunks)}, Embeddings: {len(embeddings_data)}, Skipped: {skipped_chunks}")
        
        return {
            "success": True,
            "file_name": pdf_path.name,
            "pages": len(pdf_doc.pages),
            "chunks": len(chunk_set.chunks),
            "embeddings": len(embeddings_data),
            "skipped_chunks": skipped_chunks,
            "dimension": self.embedder.dimension,
            "files": {
                "chunks": str(chunks_file),
                "embeddings": str(embeddings_file),
                "faiss_index": str(faiss_file),
                "metadata_map": str(metadata_map_file),
                "summary": str(summary_file)
            }
        }
    
    def process_directory(self, pdf_dir: Optional[str | Path] = None) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDFs (default: self.pdf_dir)
            
        Returns:
            List of processing results
        """
        if pdf_dir is None:
            pdf_dir = self.pdf_dir
        else:
            pdf_dir = Path(pdf_dir)
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF file(s)")
        
        results = []
        for idx, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"Processing {idx}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                result = self.process_pdf(str(pdf_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                results.append({
                    "success": False,
                    "file_name": pdf_file.name,
                    "error": str(e)
                })
        
        # Create summary
        batch_summary = self.summary_generator.create_batch_summary(results)
        self.summary_generator.save_batch_summary(batch_summary)
        
        return results
    
    def load_index(self, faiss_file: Path, metadata_map_file: Path) -> tuple:
        """
        Load existing FAISS index and metadata map.
        
        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file
            
        Returns:
            Tuple of (faiss_index, metadata_map)
        """
        return self.vector_store.load_index(faiss_file, metadata_map_file)
    
    def search_similar(self, faiss_file: Path, metadata_map_file: Path,
                    query_text: str, top_k: int = 5, use_softmax: bool = False,
                    temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using FAISS index.
        
        Args:
            faiss_file: Path to FAISS index file
            metadata_map_file: Path to metadata map file
            query_text: Query text to search
            top_k: Number of results to return
            
        Returns:
            List of similar chunks with metadata and distances
        """
        return self.retriever.search_similar(faiss_file, metadata_map_file, query_text, top_k, use_softmax, temperature)

def main():
    """Main entry point for RAG Pipeline."""
    logger.info("Starting RAG Pipeline")
    
    # Initialize pipeline với Gemma embedder
    pipeline = RAGPipeline(
        output_dir="data",
        model_type=OllamaModelType.GEMMA
    )
    
    logger.info("RAG Pipeline initialized and ready to use")
    
    # Process all PDFs in data/pdf directory
    try:
        results = pipeline.process_directory()
        
        logger.info("All processing completed successfully")
        logger.info(f"Output files saved to: {pipeline.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()