"""
Ollama backend integration for LLM inference
"""

import logging
import time
from typing import Dict, List, Any, Optional

from llama_index.llms.ollama import Ollama
from src.config import ModelConfig

logger = logging.getLogger(__name__)

class OllamaBackend:
    """Ollama LLM backend implementation"""

    def __init__(self, config: ModelConfig):
        """Initialize Ollama backend"""
        self.config = config
        self.llm = Ollama(
            model=config.model_name,
            base_url=config.api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_timeout=30.0
        )
        logger.info(f"Initialized Ollama backend with model: {config.model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from Ollama

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        try:
            start_time = time.time()
            response = self.llm.complete(prompt, **kwargs)
            end_time = time.time()

            latency = end_time - start_time
            logger.info(f"Ollama generation completed in {latency:.2f}s")

            return response.text

        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise

    def generate_stream(self, prompt: str, **kwargs):
        """
        Generate streaming response from Ollama

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments

        Yields:
            Generated text chunks
        """
        try:
            start_time = time.time()

            for chunk in self.llm.stream_complete(prompt, **kwargs):
                yield chunk.text

            end_time = time.time()
            latency = end_time - start_time
            logger.info(f"Ollama streaming generation completed in {latency:.2f}s")

        except Exception as e:
            logger.error(f"Error in streaming generation from Ollama: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Ollama

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            # Note: Ollama embeddings API may vary
            # This is a placeholder - adjust based on Ollama version
            embedding = self.llm.embeddings.create(
                model=self.config.model_name,
                input=text
            )
            return embedding['data'][0]['embedding']

        except Exception as e:
            logger.error(f"Error getting embedding from Ollama: {e}")
            raise

    def health_check(self) -> bool:
        """Check if Ollama backend is healthy"""
        try:
            # Simple health check by generating a short response
            response = self.generate("Hello", max_tokens=5)
            return len(response) > 0
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "backend": "ollama",
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_base": self.config.api_base
        }