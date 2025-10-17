#!/usr/bin/env python3
"""
RAG Pipeline Runner Script
==========================
Chạy pipeline RAG để xử lý PDF và tạo embeddings

Usage:
    python scripts/run_pipeline.py
    # hoặc từ thư mục gốc:
    python -m scripts.run_pipeline
"""

import sys
import os
from pathlib import Path

# Thêm thư mục gốc vào Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.rag_pipeline import RAGPipeline
from embedders.providers.ollama import OllamaModelType


def run_pipeline():
    """Chạy RAG pipeline để xử lý tất cả PDF và tạo embeddings"""
    print("🚀 CHẠY RAG PIPELINE - XỬ LÝ PDF VÀ TẠO EMBEDDINGS")
    print("=" * 70)

    try:
        # Khởi tạo pipeline với Gemma embedder
        print("🔧 Khởi tạo pipeline...")
        pipeline = RAGPipeline(
            output_dir="data",
            model_type=OllamaModelType.GEMMA
        )

        print("✅ Pipeline đã khởi tạo thành công")
        print("📁 Đang xử lý tất cả PDF trong thư mục data/pdf...")

        # Xử lý tất cả PDF trong thư mục
        results = pipeline.process_directory()

        print(f"\n✅ HOÀN THÀNH! Đã xử lý {len(results)} PDF")
        print("\n📊 KẾT QUẢ CHI TIẾT:")

        total_pages = 0
        total_chunks = 0
        total_embeddings = 0

        for i, result in enumerate(results, 1):
            print(f"\n--- PDF {i}: {result.get('file_name', 'Unknown')} ---")
            print(f"📄 Số trang: {result.get('pages_processed', 0)}")
            print(f"✂️ Số chunks: {result.get('chunks_created', 0)}")
            print(f"🧠 Embeddings: {result.get('embeddings_created', 0)}")
            print(f"💾 Vector index: {'✅' if result.get('vector_index_saved') else '❌'}")
            print(f"📋 Metadata: {'✅' if result.get('metadata_saved') else '❌'}")

            # Tính tổng
            total_pages += result.get('pages_processed', 0)
            total_chunks += result.get('chunks_created', 0)
            total_embeddings += result.get('embeddings_created', 0)

            if result.get('errors'):
                print(f"⚠️ Lỗi: {result['errors']}")

        print("\n" + "="*70)
        print("🎉 PIPELINE HOÀN THÀNH!")
        print(f"📊 TỔNG KẾT:")
        print(f"   • Tổng số PDF: {len(results)}")
        print(f"   • Tổng số trang: {total_pages}")
        print(f"   • Tổng số chunks: {total_chunks}")
        print(f"   • Tổng số embeddings: {total_embeddings}")
        print(f"   • Dữ liệu lưu tại: {project_root}/data/")

    except Exception as e:
        print(f"❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Main entry point"""
    success = run_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()