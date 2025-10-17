#!/usr/bin/env python3
"""
End-to-End Test with Real PDF Data
===================================
Test RAG pipeline với file PDF thật
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print('🚀 Testing RAG Pipeline with Real PDF Data')
    print('=' * 60)

    try:
        from pipeline.rag_pipeline import RAGPipeline
        from embedders.providers.ollama import OllamaModelType

        # Initialize pipeline
        print('📋 Initializing RAG Pipeline...')
        pipeline = RAGPipeline(model_type=OllamaModelType.GEMMA)

        # Process the PDF
        pdf_path = r'C:\Users\ENGUYEHWC\Prototype\Version_4\data\pdf\Process_Service Configuration Management.pdf'
        print(f'📄 Processing PDF: {Path(pdf_path).name}')

        result = pipeline.process_pdf(pdf_path)

        print('✅ PDF Processing Complete!')
        print(f'📊 Status: {"success" if result.get("success") else "failed"}')
        print(f'📄 Filename: {result.get("file_name", "unknown")}')
        print(f'📑 Pages: {result.get("pages", "unknown")}')
        print(f'🔢 Chunks: {result.get("chunks", "unknown")}')
        print(f'🎯 Embeddings: {result.get("embeddings", "unknown")}')
        print(f'📐 Dimension: {result.get("dimension", "unknown")}')

        # Test search functionality
        print('\n🔍 Testing Search Functionality...')

        # Find FAISS files
        vectors_dir = Path('data/vectors')
        faiss_files = list(vectors_dir.glob('*_vectors_*.faiss'))

        if faiss_files:
            faiss_file = faiss_files[0]
            metadata_file = faiss_file.with_name(faiss_file.name.replace('_vectors_', '_metadata_map_')).with_suffix('.pkl')

            print(f'🎯 Found FAISS index: {faiss_file.name}')

            # Test search
            search_results = pipeline.search_similar(
                faiss_file=faiss_file,
                metadata_map_file=metadata_file,
                query_text='service configuration management',
                top_k=3
            )

            print(f'🔎 Search Results ({len(search_results)} found):')
            for i, result in enumerate(search_results, 1):
                print(f'{i}. Score: {result["similarity_score"]:.4f}')
                print(f'   Page: {result["page_number"]}')
                print(f'   Text: {result["text"][:100]}...')
                print()
        else:
            print('⚠️  No FAISS files found for search testing')

        print('🎉 End-to-End Test COMPLETED SUCCESSFULLY!')

    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()