"""
HuggingFace backend integration for LLM inference
"""

import logging
import time
from typing import Dict, List, Any, Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    AutoModelForCausalLM
)
from src.config import ModelConfig

logger = logging.getLogger(__name__)

class HuggingFaceBackend:
    """HuggingFace LLM backend implementation"""

    def __init__(self, config: ModelConfig):
        """Initialize HuggingFace backend"""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        self._load_model()

    def _load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading HuggingFace model: {self.config.model_name}")

        try:
            # Determine model type and load accordingly
            if "t5" in self.config.model_name.lower():
                # T5 models are seq2seq
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            else:
                # Assume causal language models for others
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("Model moved to GPU")
            else:
                logger.info("GPU not available, using CPU")

            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation" if not "t5" in self.config.model_name.lower() else "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.config.max_tokens,
                temperature=self.config.temperature,
                device=0 if torch.cuda.is_available() else -1
            )

            logger.info(f"Successfully loaded model: {self.config.model_name}")

        except Exception as e:
            logger.error(f"Error loading model {self.config.model_name}: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from HuggingFace model

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        try:
            start_time = time.time()

            # Prepare generation parameters
            gen_kwargs = {
                "max_length": self.config.max_tokens,
                "temperature": self.config.temperature,
                "do_sample": True,
                **kwargs
            }

            if "t5" in self.config.model_name.lower():
                # T5 models need different handling
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    input_ids = input_ids.to("cuda")

                outputs = self.model.generate(
                    input_ids,
                    max_length=self.config.max_tokens,
                    temperature=self.config.temperature,
                    num_beams=4,
                    early_stopping=True,
                    **kwargs
                )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # Use pipeline for other models
                output = self.pipeline(prompt, **gen_kwargs)
                response = output[0]['generated_text']

                # Remove the prompt from the response if it's included
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()

            end_time = time.time()
            latency = end_time - start_time
            logger.info(f"HuggingFace generation completed in {latency:.2f}s")

            return response

        except Exception as e:
            logger.error(f"Error generating response from HuggingFace: {e}")
            raise

    def generate_stream(self, prompt: str, **kwargs):
        """
        Generate streaming response from HuggingFace

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments

        Yields:
            Generated text chunks
        """
        try:
            start_time = time.time()

            # Note: Streaming is more complex with Transformers
            # This is a simplified implementation
            gen_kwargs = {
                "max_length": self.config.max_tokens,
                "temperature": self.config.temperature,
                "do_sample": True,
                **kwargs
            }

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")

            # Generate all at once and yield chunks (simplified streaming)
            outputs = self.model.generate(
                input_ids,
                **gen_kwargs
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            # Yield the response in chunks for consistency
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                yield response[i:i+chunk_size]

            end_time = time.time()
            latency = end_time - start_time
            logger.info(f"HuggingFace streaming generation completed in {latency:.2f}s")

        except Exception as e:
            logger.error(f"Error in streaming generation from HuggingFace: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using HuggingFace model

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            # Use the model's embeddings layer
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")

            # Get embeddings from the model
            with torch.no_grad():
                outputs = self.model.get_input_embeddings()(input_ids)
                # Average pooling to get single vector
                embedding = torch.mean(outputs, dim=1).squeeze().cpu().numpy()

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error getting embedding from HuggingFace: {e}")
            raise

    def health_check(self) -> bool:
        """Check if HuggingFace backend is healthy"""
        try:
            # Simple health check by generating a short response
            if self.model is None or self.tokenizer is None:
                return False

            response = self.generate("Hello", max_length=10)
            return len(response) > 0
        except Exception as e:
            logger.error(f"HuggingFace health check failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "backend": "huggingface",
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }