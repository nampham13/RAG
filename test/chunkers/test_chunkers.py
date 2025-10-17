import pytest
from unittest.mock import MagicMock

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chunkers.fixed_size_chunker import FixedSizeChunker
from chunkers.hybrid_chunker import HybridChunker, ChunkerMode
from chunkers.semantic_chunker import SemanticChunker
from chunkers.rule_based_chunker import RuleBasedChunker
from chunkers.model import Chunk, ChunkSet
from loaders.model.block import Block


class TestFixedSizeChunker:
    """Test cơ bản cho FixedSizeChunker"""

    def test_initialization(self):
        """Test constructor với default parameters"""
        chunker = FixedSizeChunker()
        assert chunker.max_tokens == 512
        assert chunker.overlap_tokens == 50

    def test_initialization_custom_params(self):
        """Test constructor với custom parameters"""
        chunker = FixedSizeChunker(max_tokens=256, overlap_tokens=25)
        assert chunker.max_tokens == 256
        assert chunker.overlap_tokens == 25

    def test_chunk_text_basic(self):
        """Test chunk_text với text đơn giản"""
        chunker = FixedSizeChunker(max_tokens=50, overlap_tokens=10)

        test_text = "This is a simple test text for chunking. " * 20
        blocks = [Block(text=test_text)]

        # Create PDFDocument
        from loaders.model.document import PDFDocument
        from loaders.model.page import PDFPage
        page = PDFPage(page_number=1, blocks=blocks)
        document = PDFDocument(file_path="test.pdf", num_pages=1, pages=[page])

        result = chunker.chunk(document)

        assert isinstance(result, ChunkSet)
        assert len(result.chunks) > 0

        # Kiểm tra mỗi chunk không vượt quá max_tokens
        for chunk in result.chunks:
            assert len(chunk.text.split()) <= 100  # Rough estimate


class TestHybridChunker:
    """Test cơ bản cho HybridChunker"""

    def test_initialization_default(self):
        """Test constructor với default parameters"""
        chunker = HybridChunker()
        assert chunker.max_tokens == 512
        assert chunker.overlap_tokens == 50
        assert chunker.mode == ChunkerMode.AUTO

    def test_initialization_custom_mode(self):
        """Test constructor với custom mode"""
        chunker = HybridChunker(mode=ChunkerMode.SEMANTIC_FIRST)
        assert chunker.mode == ChunkerMode.SEMANTIC_FIRST

    def test_chunk_with_empty_blocks(self):
        """Test chunk với empty blocks"""
        chunker = HybridChunker()

        # Create empty PDFDocument
        from loaders.model.document import PDFDocument
        document = PDFDocument(file_path="test.pdf", num_pages=0, pages=[])

        result = chunker.chunk(document)
        assert isinstance(result, ChunkSet)
        assert len(result.chunks) == 0

    def test_chunk_with_simple_text(self):
        """Test chunk với text đơn giản"""
        chunker = HybridChunker(max_tokens=100, overlap_tokens=10)

        test_text = "This is a test document. " * 10
        blocks = [Block(text=test_text)]

        # Create PDFDocument
        from loaders.model.document import PDFDocument
        from loaders.model.page import PDFPage
        page = PDFPage(page_number=1, blocks=blocks)
        document = PDFDocument(file_path="test.pdf", num_pages=1, pages=[page])

        result = chunker.chunk(document)

        assert isinstance(result, ChunkSet)
        assert len(result.chunks) > 0


class TestSemanticChunker:
    """Test cơ bản cho SemanticChunker"""

    def test_initialization(self):
        """Test constructor"""
        chunker = SemanticChunker()
        assert chunker.max_tokens == 512
        assert chunker.overlap_tokens == 50

    def test_chunk_basic(self):
        """Test chunk method cơ bản"""
        chunker = SemanticChunker(max_tokens=100)

        test_text = "This is the first sentence. This is the second sentence. This is the third sentence."
        blocks = [Block(text=test_text)]

        # Create PDFDocument
        from loaders.model.document import PDFDocument
        from loaders.model.page import PDFPage
        page = PDFPage(page_number=1, blocks=blocks)
        document = PDFDocument(file_path="test.pdf", num_pages=1, pages=[page])

        result = chunker.chunk(document)

        assert isinstance(result, ChunkSet)
        assert len(result.chunks) > 0


class TestRuleBasedChunker:
    """Test cơ bản cho RuleBasedChunker"""

    def test_initialization(self):
        """Test constructor"""
        chunker = RuleBasedChunker()
        assert chunker.max_tokens == 512
        assert chunker.overlap_tokens == 50

    def test_chunk_basic(self):
        """Test chunk method cơ bản"""
        chunker = RuleBasedChunker(max_tokens=100)

        test_text = """
        # Header 1

        This is paragraph 1.

        ## Header 2

        This is paragraph 2.

        ### Header 3

        This is paragraph 3.
        """

        blocks = [Block(text=test_text)]

        # Create PDFDocument
        from loaders.model.document import PDFDocument
        from loaders.model.page import PDFPage
        page = PDFPage(page_number=1, blocks=blocks)
        document = PDFDocument(file_path="test.pdf", num_pages=1, pages=[page])

        result = chunker.chunk(document)

        assert isinstance(result, ChunkSet)
        assert len(result.chunks) > 0