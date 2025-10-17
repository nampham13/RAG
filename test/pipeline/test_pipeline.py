import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import faiss

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.rag_pipeline import RAGPipeline
from pipeline.vector_store import VectorStore
from pipeline.retriever import Retriever
from pipeline.summary_generator import SummaryGenerator
from embedders.providers.ollama import OllamaModelType


class TestVectorStore:
    """Test cơ bản cho VectorStore"""

    def test_initialization(self):
        """Test constructor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(Path(tmp_dir))
            assert store.vectors_dir == Path(tmp_dir)
            assert store.vectors_dir.exists()

    def test_create_index_basic(self):
        """Test create_index với data cơ bản"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(Path(tmp_dir))

            # Mock embeddings data
            embeddings_data = [
                {
                    'chunk_id': 'chunk_1',
                    'text': 'test text 1',
                    'embedding': np.random.rand(768).tolist(),
                    'embedding_dimension': 768,
                    'text_length': 10,
                    'file_name': 'test.pdf',
                    'file_path': '/test/test.pdf',
                    'page_number': 1,
                    'page_numbers': [1],
                    'chunk_index': 0,
                    'block_type': 'text',
                    'block_ids': [],
                    'is_table': False,
                    'token_count': 5
                },
                {
                    'chunk_id': 'chunk_2',
                    'text': 'test text 2',
                    'embedding': np.random.rand(768).tolist(),
                    'embedding_dimension': 768,
                    'text_length': 10,
                    'file_name': 'test.pdf',
                    'file_path': '/test/test.pdf',
                    'page_number': 2,
                    'page_numbers': [2],
                    'chunk_index': 1,
                    'block_type': 'text',
                    'block_ids': [],
                    'is_table': False,
                    'token_count': 5
                }
            ]

            # Create index
            faiss_file, metadata_file = store.create_index(embeddings_data, "test_doc", "20241201_120000")

            # Verify files exist
            assert faiss_file.exists()
            assert metadata_file.exists()

    def test_load_index(self):
        """Test load_index method"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(Path(tmp_dir))

            # First create an index
            embeddings_data = [
                {
                    'chunk_id': 'chunk_1',
                    'text': 'test text',
                    'embedding': np.random.rand(768).tolist(),
                    'embedding_dimension': 768,
                    'text_length': 9,
                    'file_name': 'test.pdf',
                    'file_path': '/test/test.pdf',
                    'page_number': 1,
                    'page_numbers': [1],
                    'chunk_index': 0,
                    'block_type': 'text',
                    'block_ids': [],
                    'is_table': False,
                    'token_count': 5
                }
            ]

            faiss_file, metadata_file = store.create_index(embeddings_data, "test_doc", "20241201_120000")

            # Load the index
            loaded_index, loaded_metadata = store.load_index(faiss_file, metadata_file)

            # Verify loaded data
            assert isinstance(loaded_index, faiss.Index)
            assert isinstance(loaded_metadata, dict)
            assert len(loaded_metadata) > 0


class TestRetriever:
    """Test cơ bản cho Retriever"""

    def test_initialization(self):
        """Test constructor"""
        mock_embedder = Mock()
        retriever = Retriever(mock_embedder)
        assert retriever.embedder == mock_embedder

    @patch.object(Retriever, '_load_index_and_metadata')
    def test_search_similar(self, mock_load_method):
        """Test search_similar method"""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.embed.return_value = np.random.rand(768)

        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]]),  # similarities
            np.array([[0, 1]])      # indices
        )
        mock_metadata = {0: {'text': 'result 1', 'page_number': 1}, 1: {'text': 'result 2', 'page_number': 2}}
        mock_load_method.return_value = (mock_index, mock_metadata)

        retriever = Retriever(mock_embedder)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create dummy files
            faiss_file = Path(tmp_dir) / "test.faiss"
            metadata_file = Path(tmp_dir) / "test.pkl"
            faiss_file.touch()
            metadata_file.touch()

            results = retriever.search_similar(
                faiss_file=faiss_file,
                metadata_map_file=metadata_file,
                query_text="test query",
                top_k=2
            )

            # Verify results
            assert len(results) == 2
            assert results[0]['text'] == 'result 1'
            assert results[0]['similarity_score'] == 0.9
            assert results[1]['text'] == 'result 2'
            assert results[1]['similarity_score'] == 0.8

            # Verify calls
            mock_embedder.embed.assert_called_once_with("test query")
            mock_load_method.assert_called_once_with(faiss_file, metadata_file)
            mock_index.search.assert_called_once()


class TestSummaryGenerator:
    """Test cơ bản cho SummaryGenerator"""

    def test_initialization(self):
        """Test constructor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_dir = Path(tmp_dir) / "metadata"
            output_dir = Path(tmp_dir) / "output"

            generator = SummaryGenerator(metadata_dir, output_dir)
            assert generator.metadata_dir == metadata_dir
            assert generator.output_dir == output_dir

    def test_save_document_summary(self):
        """Test save_document_summary method"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_dir = Path(tmp_dir) / "metadata"
            output_dir = Path(tmp_dir) / "output"

            generator = SummaryGenerator(metadata_dir, output_dir)

            # Test data
            summary = {
                'filename': 'test.pdf',
                'total_pages': 10,
                'total_chunks': 25,
                'total_tokens': 5000
            }

            summary_file = generator.save_document_summary(summary, "test_doc", "20241201_120000")

            assert summary_file.exists()
            assert summary_file.suffix == '.json'


class TestRAGPipeline:
    """Test cơ bản cho RAGPipeline"""

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    def test_initialization(self, mock_summary_gen, mock_vector_store,
                           mock_embedder_factory, mock_chunker, mock_loader):
        """Test constructor"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = RAGPipeline(output_dir=tmp_dir, model_type=OllamaModelType.GEMMA)

            assert pipeline.output_dir == Path(tmp_dir)
            assert pipeline.model_type == OllamaModelType.GEMMA

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    def test_initialization_with_pdf_dir(self, mock_summary_gen, mock_vector_store,
                                       mock_embedder_factory, mock_chunker, mock_loader):
        """Test constructor với pdf_dir parameter"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = Path(tmp_dir) / "custom_pdf_dir"
            pipeline = RAGPipeline(output_dir=tmp_dir, pdf_dir=pdf_dir, model_type=OllamaModelType.GEMMA)

            assert pipeline.output_dir == Path(tmp_dir)
            assert pipeline.pdf_dir == pdf_dir
            assert pipeline.model_type == OllamaModelType.GEMMA

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    def test_get_info(self, mock_summary_gen, mock_vector_store,
                     mock_embedder_factory, mock_chunker, mock_loader):
        """Test get_info method"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = RAGPipeline(output_dir=tmp_dir, model_type=OllamaModelType.GEMMA)

            info = pipeline.get_info()

            assert isinstance(info, dict)
            assert 'output_dir' in info
            assert 'pdf_dir' in info
            assert 'model_type' in info
            assert 'embedder_model' in info
            assert 'embedder_dimension' in info
            assert 'loader' in info
            assert 'chunker' in info
            assert 'vector_store' in info
            assert 'cache_enabled' in info

            assert info['output_dir'] == tmp_dir
            assert info['loader'] == 'PDFLoader'
            assert info['vector_store'] == 'FAISS'
            assert info['cache_enabled'] == True

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    @patch('pipeline.rag_pipeline.OllamaModelSwitcher')
    def test_switch_model(self, mock_model_switcher, mock_summary_gen, mock_vector_store,
                         mock_embedder_factory, mock_chunker, mock_loader):
        """Test switch_model method"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup mock switcher
            mock_gemma_embedder = Mock()
            mock_gemma_embedder.profile.model_id = 'embeddinggemma:latest'
            mock_model_switcher.return_value.switch_to_gemma.return_value = mock_gemma_embedder

            pipeline = RAGPipeline(output_dir=tmp_dir, model_type=OllamaModelType.GEMMA)

            # Switch to BGE_M3
            mock_bge_embedder = Mock()
            mock_bge_embedder.profile.model_id = 'bge-m3:latest'
            mock_model_switcher.return_value.switch_to_bge_m3.return_value = mock_bge_embedder

            pipeline.switch_model(OllamaModelType.BGE_M3)

            assert pipeline.model_type == OllamaModelType.BGE_M3
            assert pipeline.embedder == mock_bge_embedder
            mock_model_switcher.return_value.switch_to_bge_m3.assert_called_once()

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    @patch('pipeline.rag_pipeline.DataQualityAnalyzer')
    def test_analyze_data_quality_delegation(self, mock_quality_analyzer, mock_summary_gen,
                                           mock_vector_store, mock_embedder_factory,
                                           mock_chunker, mock_loader):
        """Test analyze_data_quality delegation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = RAGPipeline(output_dir=tmp_dir)

            mock_result = {'quality_score': 95.5, 'issues': []}
            mock_quality_analyzer.return_value.analyze_data_quality.return_value = mock_result

            result = pipeline.analyze_data_quality('test.pdf')

            assert result == mock_result
            mock_quality_analyzer.return_value.analyze_data_quality.assert_called_once_with('test.pdf')

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    @patch('pipeline.rag_pipeline.DataIntegrityChecker')
    def test_check_data_integrity_delegation(self, mock_integrity_checker, mock_summary_gen,
                                           mock_vector_store, mock_embedder_factory,
                                           mock_chunker, mock_loader):
        """Test check_data_integrity delegation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = RAGPipeline(output_dir=tmp_dir)

            mock_result = {'overall_integrity': True, 'issues': []}
            mock_integrity_checker.return_value.check_data_integrity.return_value = mock_result

            result = pipeline.check_data_integrity('test.pdf')

            assert result == mock_result
            mock_integrity_checker.return_value.check_data_integrity.assert_called_once_with(
                'test.pdf', None, None, None, None
            )

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    @patch('pipeline.rag_pipeline.ChunkCacheManager')
    def test_chunk_cache_methods(self, mock_cache_manager, mock_summary_gen,
                               mock_vector_store, mock_embedder_factory,
                               mock_chunker, mock_loader):
        """Test chunk cache delegation methods"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = RAGPipeline(output_dir=tmp_dir)

            # Test is_chunk_processed
            mock_cache_manager.return_value.is_chunk_processed.return_value = True
            result = pipeline.is_chunk_processed('chunk_1', 'hash123')
            assert result == True
            mock_cache_manager.return_value.is_chunk_processed.assert_called_once_with('chunk_1', 'hash123')

            # Test mark_chunk_processed
            metadata = {'page': 1, 'tokens': 50}
            pipeline.mark_chunk_processed('chunk_1', 'hash123', metadata)
            mock_cache_manager.return_value.mark_chunk_processed.assert_called_once_with('chunk_1', 'hash123', metadata)

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    def test_load_index_delegation(self, mock_summary_gen, mock_vector_store,
                                 mock_embedder_factory, mock_chunker, mock_loader):
        """Test load_index delegation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = RAGPipeline(output_dir=tmp_dir)

            mock_index = Mock()
            mock_metadata = {'test': 'data'}
            mock_vector_store.return_value.load_index.return_value = (mock_index, mock_metadata)

            faiss_file = Path(tmp_dir) / "test.faiss"
            metadata_file = Path(tmp_dir) / "test.pkl"

            result = pipeline.load_index(faiss_file, metadata_file)

            assert result == (mock_index, mock_metadata)
            mock_vector_store.return_value.load_index.assert_called_once_with(faiss_file, metadata_file)

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    def test_process_directory_with_no_pdfs(self, mock_summary_gen, mock_vector_store,
                                          mock_embedder_factory, mock_chunker, mock_loader):
        """Test process_directory với empty directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = RAGPipeline(output_dir=tmp_dir)
            results = pipeline.process_directory(tmp_dir)

            assert results == []

    @patch('pipeline.rag_pipeline.PDFLoader')
    @patch('pipeline.rag_pipeline.HybridChunker')
    @patch('pipeline.rag_pipeline.EmbedderFactory')
    @patch('pipeline.rag_pipeline.VectorStore')
    @patch('pipeline.rag_pipeline.SummaryGenerator')
    def test_process_directory_uses_pdf_dir(self, mock_summary_gen, mock_vector_store,
                                          mock_embedder_factory, mock_chunker, mock_loader):
        """Test process_directory uses self.pdf_dir as default"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = Path(tmp_dir) / "custom_pdfs"
            pdf_dir.mkdir()

            pipeline = RAGPipeline(output_dir=tmp_dir, pdf_dir=pdf_dir)

            # Mock os.path.exists and listdir to simulate empty directory
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.glob', return_value=[]):
                    results = pipeline.process_directory()

                    assert results == []


class TestDataQualityAnalyzer:
    """Test cho DataQualityAnalyzer"""

    @patch('pipeline.data_quality_analyzer.PDFLoader')
    @patch('pipeline.data_quality_analyzer.HybridChunker')
    @patch('pipeline.data_quality_analyzer.IEmbedder')
    def test_initialization(self, mock_embedder, mock_chunker, mock_loader):
        """Test constructor"""
        from pipeline.data_quality_analyzer import DataQualityAnalyzer

        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = DataQualityAnalyzer(
                loader=mock_loader,
                chunker=mock_chunker,
                embedder=mock_embedder,
                output_dir=Path(tmp_dir)
            )

            assert analyzer.loader == mock_loader
            assert analyzer.chunker == mock_chunker
            assert analyzer.embedder == mock_embedder
            assert analyzer.output_dir == Path(tmp_dir)

    @patch('pipeline.data_quality_analyzer.PDFLoader')
    @patch('pipeline.data_quality_analyzer.HybridChunker')
    @patch('pipeline.data_quality_analyzer.IEmbedder')
    def test_analyze_data_quality_basic(self, mock_embedder, mock_chunker, mock_loader):
        """Test analyze_data_quality với mock data"""
        from pipeline.data_quality_analyzer import DataQualityAnalyzer

        # Setup mocks
        mock_block = Mock()
        mock_block.text = "test block text"
        mock_block.block_type = "text"

        mock_page = Mock()
        mock_page.blocks = [mock_block]

        mock_pdf_doc = Mock()
        mock_pdf_doc.pages = [mock_page]

        mock_loader.load.return_value = mock_pdf_doc
        mock_loader.load.return_value.normalize.return_value = mock_pdf_doc

        mock_chunk = Mock()
        mock_chunk.text = "test chunk"
        mock_chunk.token_count = 10
        mock_chunk.provenance = None
        mock_chunk.metadata = {}

        mock_chunk_set = Mock()
        mock_chunk_set.chunks = [mock_chunk]
        mock_chunk_set.chunk_strategy = "hybrid"
        mock_chunk_set.total_tokens = 100

        mock_chunker.chunk.return_value = mock_chunk_set

        mock_embedding = np.random.rand(768)
        mock_embedder.embed.return_value = mock_embedding
        mock_embedder.dimension = 768
        mock_embedder.profile.model_id = "test-model"

        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = DataQualityAnalyzer(
                loader=mock_loader,
                chunker=mock_chunker,
                embedder=mock_embedder,
                output_dir=Path(tmp_dir)
            )

            result = analyzer.analyze_data_quality("test.pdf")

            assert isinstance(result, dict)
            assert 'stages' in result
            assert 'quality_score' in result  # Correct key name
            assert 'metrics' in result  # Check for metrics instead of recommendations

    @patch('pipeline.data_quality_analyzer.PDFLoader')
    @patch('pipeline.data_quality_analyzer.HybridChunker')
    @patch('pipeline.data_quality_analyzer.IEmbedder')
    def test_save_quality_report(self, mock_embedder, mock_chunker, mock_loader):
        """Test save_quality_report method"""
        from pipeline.data_quality_analyzer import DataQualityAnalyzer

        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = DataQualityAnalyzer(
                loader=mock_loader,
                chunker=mock_chunker,
                embedder=mock_embedder,
                output_dir=Path(tmp_dir)
            )

            quality_report = {
                'file_name': 'test.pdf',
                'overall_score': 95.5,
                'stages': {'raw': {}, 'normalized': {}, 'chunked': {}, 'embedded': {}},
                'recommendations': []
            }

            report_file = analyzer.save_quality_report(quality_report)

            assert report_file.exists()
            assert report_file.suffix == '.json'
            assert 'test.pdf_quality_' in report_file.name  # Correct filename pattern

    @patch('pipeline.data_quality_analyzer.PDFLoader')
    @patch('pipeline.data_quality_analyzer.HybridChunker')
    @patch('pipeline.data_quality_analyzer.IEmbedder')
    def test_compare_quality_reports(self, mock_embedder, mock_chunker, mock_loader):
        """Test compare_quality_reports method"""
        from pipeline.data_quality_analyzer import DataQualityAnalyzer

        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = DataQualityAnalyzer(
                loader=mock_loader,
                chunker=mock_chunker,
                embedder=mock_embedder,
                output_dir=Path(tmp_dir)
            )

            report1 = {
                'file_name': 'doc1.pdf', 
                'quality_score': 90.0, 
                'total_chunks': 100,
                'stages': {'raw_extraction': {}, 'normalized': {}, 'chunking': {}, 'embedding': {}}
            }
            report2 = {
                'file_name': 'doc2.pdf', 
                'quality_score': 95.0, 
                'total_chunks': 105,
                'stages': {'raw_extraction': {}, 'normalized': {}, 'chunking': {}, 'embedding': {}}
            }

            comparison = analyzer.compare_quality_reports(report1, report2)

            assert isinstance(comparison, dict)
            assert 'score_difference' in comparison  # Correct key name
            assert 'files' in comparison
            assert comparison['score_difference'] == 5.0


class TestDataIntegrityChecker:
    """Test cho DataIntegrityChecker"""

    def test_initialization(self):
        """Test constructor"""
        from pipeline.data_integrity_checker import DataIntegrityChecker

        with tempfile.TemporaryDirectory() as tmp_dir:
            chunks_dir = Path(tmp_dir) / "chunks"
            embeddings_dir = Path(tmp_dir) / "embeddings"
            vectors_dir = Path(tmp_dir) / "vectors"
            metadata_dir = Path(tmp_dir) / "metadata"

            checker = DataIntegrityChecker(chunks_dir, embeddings_dir, vectors_dir, metadata_dir)

            assert checker.chunks_dir == chunks_dir
            assert checker.embeddings_dir == embeddings_dir
            assert checker.vectors_dir == vectors_dir
            assert checker.metadata_dir == metadata_dir

    def test_check_data_integrity_missing_files(self):
        """Test check_data_integrity với missing files"""
        from pipeline.data_integrity_checker import DataIntegrityChecker

        with tempfile.TemporaryDirectory() as tmp_dir:
            chunks_dir = Path(tmp_dir) / "chunks"
            embeddings_dir = Path(tmp_dir) / "embeddings"
            vectors_dir = Path(tmp_dir) / "vectors"
            metadata_dir = Path(tmp_dir) / "metadata"

            checker = DataIntegrityChecker(chunks_dir, embeddings_dir, vectors_dir, metadata_dir)

            result = checker.check_data_integrity("test.pdf")

            assert isinstance(result, dict)
            assert 'overall_integrity' in result
            assert 'issues' in result
            assert result['overall_integrity'] == False
            assert len(result['issues']) > 0

    @patch('builtins.open')
    @patch('json.load')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_check_data_integrity_with_mock_files(self, mock_glob, mock_exists, mock_json_load, mock_open):
        """Test check_data_integrity với mock files"""
        from pipeline.data_integrity_checker import DataIntegrityChecker

        # Setup mocks
        mock_exists.return_value = True
        mock_glob.return_value = [Mock()]  # Mock file list

        mock_chunks_data = [{'chunk_id': 'chunk_1', 'text': 'test text'}]
        mock_embeddings_data = [{'chunk_id': 'chunk_1', 'text': 'test text', 'embedding': [0.1] * 768}]

        # Mock file reading with context manager
        mock_file_handle = Mock()
        mock_file_handle.read.return_value = "CHUNK 1: chunk_1\nPage: 1 | Tokens: 10 | Type: text\n---\ntest text\n\n========"
        mock_open.return_value.__enter__.return_value = mock_file_handle
        mock_open.return_value.__exit__.return_value = None

        mock_json_load.return_value = mock_embeddings_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            chunks_dir = Path(tmp_dir) / "chunks"
            embeddings_dir = Path(tmp_dir) / "embeddings"
            vectors_dir = Path(tmp_dir) / "vectors"
            metadata_dir = Path(tmp_dir) / "metadata"

            checker = DataIntegrityChecker(chunks_dir, embeddings_dir, vectors_dir, metadata_dir)

            with patch('faiss.read_index') as mock_faiss:
                mock_index = Mock()
                mock_index.ntotal = 1
                mock_index.d = 768
                mock_faiss.return_value = mock_index

                with patch('pickle.load') as mock_pickle:
                    mock_pickle.return_value = {'0': {'chunk_id': 'chunk_1'}}

                    result = checker.check_data_integrity("test.pdf")

                    assert isinstance(result, dict)
                    assert 'overall_integrity' in result
                    assert 'checks' in result


class TestChunkCacheManager:
    """Test cho ChunkCacheManager"""

    def test_initialization(self):
        """Test constructor"""
        from pipeline.chunk_cache_manager import ChunkCacheManager

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "cache.json"
            manager = ChunkCacheManager(cache_file)

            assert manager.cache_file == cache_file

    def test_is_chunk_processed(self):
        """Test is_chunk_processed method"""
        from pipeline.chunk_cache_manager import ChunkCacheManager

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "cache.json"
            manager = ChunkCacheManager(cache_file)

            # Initially should return False
            assert not manager.is_chunk_processed('chunk_1', 'hash123')

            # Mark as processed
            manager.mark_chunk_processed('chunk_1', 'hash123', {'page': 1})

            # Now should return True
            assert manager.is_chunk_processed('chunk_1', 'hash123')

    def test_mark_chunk_processed(self):
        """Test mark_chunk_processed method"""
        from pipeline.chunk_cache_manager import ChunkCacheManager

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "cache.json"
            manager = ChunkCacheManager(cache_file)

            metadata = {'page': 1, 'tokens': 50, 'is_table': False}
            manager.mark_chunk_processed('chunk_1', 'hash123', metadata)

            # Verify cache file was created
            assert cache_file.exists()

            # Verify data was saved
            assert manager.is_chunk_processed('chunk_1', 'hash123')

    def test_get_cache_stats(self):
        """Test get_cache_stats method"""
        from pipeline.chunk_cache_manager import ChunkCacheManager

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "cache.json"
            manager = ChunkCacheManager(cache_file)

            # Add some data
            manager.mark_chunk_processed('chunk_1', 'hash1', {'page': 1})
            manager.mark_chunk_processed('chunk_2', 'hash2', {'page': 2})

            stats = manager.get_cache_stats()

            assert isinstance(stats, dict)
            assert 'total_cached_chunks' in stats  # Correct key name
            assert 'cache_file_size' in stats
            assert stats['total_cached_chunks'] == 2

    def test_clear_cache(self):
        """Test clear_cache method"""
        from pipeline.chunk_cache_manager import ChunkCacheManager

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = Path(tmp_dir) / "cache.json"
            manager = ChunkCacheManager(cache_file)

            # Add data
            manager.mark_chunk_processed('chunk_1', 'hash1', {'page': 1})

            # Verify data exists
            assert manager.is_chunk_processed('chunk_1', 'hash1')

            # Clear cache
            manager.clear_cache()

            # Verify cache is empty and file may or may not exist
            assert not manager.is_chunk_processed('chunk_1', 'hash1')