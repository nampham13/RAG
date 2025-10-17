#!/usr/bin/env python3
"""
RAG Pipeline Runner
===================
Chạy pipeline hoàn chỉnh để xử lý tất cả PDF trong data/pdf/
"""

import logging
from pathlib import Path
from pipeline.rag_pipeline import RAGPipeline
from embedders.providers.ollama import OllamaModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    logger.info("Starting RAG Pipeline Runner")

    # Initialize pipeline
    pipeline = RAGPipeline(
        output_dir="data",
        pdf_dir="data/pdf",
        model_type=OllamaModelType.GEMMA
    )

    # Process all PDFs in the directory
    pdf_dir = Path("data/pdf")
    if pdf_dir.exists() and pdf_dir.is_dir():
        logger.info(f"Processing all PDFs in {pdf_dir}")
        pipeline.process_directory(pdf_dir)
        logger.info("Pipeline processing completed")
    else:
        logger.error(f"PDF directory {pdf_dir} does not exist")

if __name__ == "__main__":
    main()
