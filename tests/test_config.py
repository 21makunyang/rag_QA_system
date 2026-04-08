"""
Tests for configuration module
"""

import pytest
from src.config import Config, ModelConfig, ChunkingConfig, VectorStoreConfig


def test_config_initialization():
    """Test configuration initialization"""
    config = Config()
    assert config.MODEL_CONFIGS is not None
    assert len(config.MODEL_CONFIGS) >= 2


def test_model_config_access():
    """Test accessing model configurations"""
    mistral_config = Config.get_model_config("mistral-7b")
    assert mistral_config.backend == "ollama"
    assert mistral_config.model_name == "mistral:7b"

    t5_config = Config.get_model_config("t5-base")
    assert t5_config.backend == "huggingface"
    assert t5_config.model_name == "t5-base"


def test_unknown_model_raises_error():
    """Test that unknown model raises error"""
    with pytest.raises(ValueError):
        Config.get_model_config("unknown-model")


def test_chunking_config():
    """Test chunking configuration"""
    assert Config.CHUNKING.chunk_size == 256
    assert Config.CHUNKING.chunk_overlap == 25
    assert Config.CHUNKING.strategy == "fixed"


def test_vector_store_config():
    """Test vector store configuration"""
    assert Config.VECTOR_STORE.store_type == "chroma"
    assert Config.VECTOR_STORE.persist_dir == "./data/vector_store"
    assert Config.VECTOR_STORE.collection_name == "cs6493_collection"


def test_embedding_model():
    """Test embedding model configuration"""
    assert Config.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"