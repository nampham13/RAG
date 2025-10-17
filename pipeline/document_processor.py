"""
Document Processor - Handles batch PDF processing with progress tracking
"""
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pipeline.rag_pipeline import RAGPipeline
from embedders.providers.ollama import OllamaModelType

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing with progress tracking"""
    
    def __init__(self, data_dir: Path, model_type: OllamaModelType = OllamaModelType.GEMMA):
        self.data_dir = Path(data_dir)
        self.pdf_dir = self.data_dir / "pdf"
        self.vectors_dir = self.data_dir / "vectors"
        self.pipeline = RAGPipeline(output_dir=str(self.data_dir), model_type=model_type)
        
    def get_unprocessed_pdfs(self) -> List[Path]:
        """Get list of PDFs that haven't been processed yet"""
        if not self.pdf_dir.exists():
            return []
        
        all_pdfs = list(self.pdf_dir.glob("*.pdf"))
        
        # Check which PDFs already have vector files
        processed_stems = set()
        if self.vectors_dir.exists():
            for vector_file in self.vectors_dir.glob("*_vectors_*.faiss"):
                # Extract base filename from vector file name
                stem = vector_file.name.split("_vectors_")[0]
                processed_stems.add(stem)
        
        # Filter out already processed PDFs
        unprocessed = [pdf for pdf in all_pdfs if pdf.stem not in processed_stems]
        return unprocessed
    
    def process_documents(self, progress_callback=None) -> Dict[str, Any]:
        """
        Process all unprocessed PDFs with progress tracking
        
        Args:
            progress_callback: Function(current_file, file_idx, total_files, current_chunk, total_chunks, elapsed) called on progress
            
        Returns:
            Dictionary with processing results
        """
        unprocessed_pdfs = self.get_unprocessed_pdfs()
        
        if not unprocessed_pdfs:
            return {
                "success": True,
                "total_files": 0,
                "processed": 0,
                "message": "No unprocessed documents found"
            }
        
        total_files = len(unprocessed_pdfs)
        results = []
        start_time = time.time()
        
        for idx, pdf_file in enumerate(unprocessed_pdfs, 1):
            elapsed = time.time() - start_time
            
            # Track chunk progress
            current_chunks = [0]  # Use list to allow modification in nested function
            total_chunks = [0]
            
            def chunk_callback(current, total):
                current_chunks[0] = current
                total_chunks[0] = total
                if progress_callback:
                    progress_callback(
                        pdf_file.name,  # current_file
                        idx,            # file_idx
                        total_files,    # total_files
                        current,        # current_chunk
                        total,          # total_chunks
                        elapsed         # elapsed
                    )
            
            try:
                result = self.pipeline.process_pdf(str(pdf_file), chunk_callback=chunk_callback)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                results.append({
                    "success": False,
                    "file_name": pdf_file.name,
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "total_files": total_files,
            "processed": len([r for r in results if r.get("success", False)]),
            "failed": len([r for r in results if not r.get("success", False)]),
            "total_time": total_time,
            "results": results
        }