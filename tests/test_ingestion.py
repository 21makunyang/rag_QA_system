"""
Tests for ingestion module
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from src.ingestion.connectors import PDFConnector, TextFileConnector
from src.ingestion.chunking import (
    ChunkingStrategy,
    FixedSizeChunking,
    SentenceBasedChunking,
    ChunkingFactory
)
from src.config import Config


def test_pdf_connector_supports():
    """Test PDF connector file support"""
    connector = PDFConnector()
    assert connector.supports("test.pdf") is True
    assert connector.supports("test.txt") is False
    assert connector.supports("test.PDF") is True


def test_text_file_connector_supports():
    """Test text file connector file support"""
    connector = TextFileConnector()
    assert connector.supports("test.txt") is True
    assert connector.supports("test.md") is True
    assert connector.supports("test.json") is True
    assert connector.supports("test.pdf") is False


def test_text_file_connector_load():
    """Test text file connector loading"""
    connector = TextFileConnector()

    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\nIt has multiple lines.")
        temp_file = f.name

    try:
        documents = connector.load(temp_file)
        assert len(documents) == 1
        assert "test document" in documents[0].text
        assert documents[0].metadata['file_name'] == Path(temp_file).name
    finally:
        os.unlink(temp_file)


def test_text_file_connector_nonexistent_file():
    """Test text file connector with nonexistent file"""
    connector = TextFileConnector()
    with pytest.raises(FileNotFoundError):
        connector.load("nonexistent.txt")


def test_fixed_size_chunking():
    """Test fixed size chunking strategy"""
    config = Config.CHUNKING
    chunker = FixedSizeChunking(config)

    # Create test documents
    from llama_index.core import Document
    docs = [
        Document(text="This is a short document."),
        Document(text="This is a longer document that should be split into multiple chunks because it exceeds the chunk size limit.")
    ]

    chunks = chunker.chunk_documents(docs)
    assert len(chunks) >= len(docs)  # At least as many chunks as documents


def test_sentence_based_chunking():
    """Test sentence based chunking strategy"""
    config = Config.CHUNKING
    chunker = SentenceBasedChunking(config)

    # Create test documents
    from llama_index.core import Document
    docs = [
        Document(text="First sentence. Second sentence. Third sentence.")
    ]

    chunks = chunker.chunk_documents(docs)
    assert len(chunks) >= 1


def test_chunking_factory():
    """Test chunking factory"""
    config = Config.CHUNKING

    # Test fixed strategy
    config.strategy = "fixed"
    chunker = ChunkingFactory.create_strategy(config)
    assert isinstance(chunker, FixedSizeChunking)

    # Test sentence strategy
    config.strategy = "sentence"
    chunker = ChunkingFactory.create_strategy(config)
    assert isinstance(chunker, SentenceBasedChunking)

    # Test unknown strategy (should default to fixed)
    config.strategy = "unknown"
    chunker = ChunkingFactory.create_strategy(config)
    assert isinstance(chunker, FixedSizeChunking)


def test_chunking_with_empty_documents():
    """Test chunking with empty documents"""
    config = Config.CHUNKING
    chunker = FixedSizeChunking(config)

    from llama_index.core import Document
    docs = [Document(text="")]

    chunks = chunker.chunk_documents(docs)
    assert len(chunks) == 0  # Empty documents should not create chunks


def test_chunking_preserves_metadata():
    """Test that chunking preserves document metadata"""
    config = Config.CHUNKING
    chunker = FixedSizeChunking(config)

    from llama_index.core import Document
    original_metadata = {'source': 'test', 'page': 1}
    docs = [Document(text="Test document content.", metadata=original_metadata)]

    chunks = chunker.chunk_documents(docs)
    assert len(chunks) > 0
    assert 'source' in chunks[0].metadata
    assert chunks[0].metadata['source'] == 'test'