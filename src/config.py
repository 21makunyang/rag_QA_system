"""
Configuration management for CS6493 LLM Applications
"""

import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for LLM backend"""
    backend: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 512
    api_base: str = None
    additional_kwargs: Dict[str, Any] = None

@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    chunk_size: int = 256
    chunk_overlap: int = 25  # 10% of chunk_size
    strategy: str = "fixed"  # fixed, sentence, semantic

@dataclass
class VectorStoreConfig:
    """Configuration for vector storage"""
    store_type: str = "chroma"  # chroma, faiss
    persist_dir: str = "./data/vector_store"
    collection_name: str = "cs6493_collection"

class Config:
    """Main configuration class"""

    # Model configurations
    MODEL_CONFIGS = {
        "mistral-7b": ModelConfig(
            backend="ollama",
            model_name="mistral:7b",
            temperature=0.7,
            max_tokens=512,
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        ),
        "t5-base": ModelConfig(
            backend="huggingface",
            model_name="t5-base",
            temperature=0.7,
            max_tokens=512
        ),
        "llama2-7b": ModelConfig(
            backend="ollama",
            model_name="llama2:7b",
            temperature=0.7,
            max_tokens=512,
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        )
    }

    # Chunking configuration
    CHUNKING = ChunkingConfig(
        chunk_size=256,
        chunk_overlap=25,
        strategy="fixed"
    )

    # Vector store configuration
    VECTOR_STORE = VectorStoreConfig()

    # Embedding configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Data paths
    DATA_DIR = "./data"
    DOCUMENTS_DIR = "./data/documents"
    VECTOR_STORE_DIR = "./data/vector_store"
    RESULTS_DIR = "./data/results"

    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        return cls.MODEL_CONFIGS[model_name]

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(cls.VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)