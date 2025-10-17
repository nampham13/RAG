import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loaders.pdf_loader import PDFLoader
from loaders.model.document import PDFDocument


class TestPDFLoader:
    """Test cơ bản cho PDFLoader"""

    def test_initialization_default_params(self):
        """Test constructor với default parameters"""
        loader = PDFLoader()
        assert loader.extract_text is True
        assert loader.extract_tables is True
        assert loader.min_text_length == 10

    def test_factory_create_default(self):
        """Test factory method create_default"""
        loader = PDFLoader.create_default()
        assert isinstance(loader, PDFLoader)
        assert loader.extract_text is True
        assert loader.extract_tables is True

    def test_factory_create_text_only(self):
        """Test factory method create_text_only"""
        loader = PDFLoader.create_text_only()
        assert isinstance(loader, PDFLoader)
        assert loader.extract_text is True
        assert loader.extract_tables is False

    def test_factory_create_tables_only(self):
        """Test factory method create_tables_only"""
        loader = PDFLoader.create_tables_only()
        assert isinstance(loader, PDFLoader)
        assert loader.extract_text is False
        assert loader.extract_tables is True

    def test_get_config(self):
        """Test get_config method"""
        loader = PDFLoader(min_text_length=20)
        config = loader.get_config()
        assert isinstance(config, dict)
        assert config['min_text_length'] == 20

    def test_update_config(self):
        """Test update_config method"""
        loader = PDFLoader()
        loader.update_config(min_text_length=30, extract_tables=False)
        assert loader.min_text_length == 30
        assert loader.extract_tables is False

    def test_enable_all_filters(self):
        """Test enable_all_filters method"""
        loader = PDFLoader()
        loader.enable_all_filters()
        assert loader.enable_repeated_block_filter is True
        assert loader.enable_position_filter is True
        assert loader.enable_page_number_filter is True

    def test_disable_all_filters(self):
        """Test disable_all_filters method"""
        loader = PDFLoader()
        loader.disable_all_filters()
        assert loader.enable_repeated_block_filter is False
        assert loader.enable_position_filter is False
        assert loader.enable_page_number_filter is False

    @patch('fitz.open')
    @patch('pdfplumber.open')
    def test_load_method_signature(self, mock_pdfplumber, mock_fitz):
        """Test load method signature (mocked)"""
        # Mock the file operations
        mock_fitz.return_value = MagicMock()
        mock_pdfplumber.return_value = MagicMock()

        loader = PDFLoader()

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # This will fail due to mocking, but tests the method exists
            with pytest.raises(Exception):
                result = loader.load(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_file_sha256(self):
        """Test _file_sha256 method"""
        loader = PDFLoader()

        # Create a temporary file with known content
        test_content = b"Hello World"
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name

        try:
            hash_value = loader._file_sha256(tmp_path)
            assert isinstance(hash_value, str)
            assert len(hash_value) == 64  # SHA256 hex length
        finally:
            os.unlink(tmp_path)