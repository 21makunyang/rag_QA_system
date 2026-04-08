"""
Evaluation metrics for LLM applications
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculator for various LLM evaluation metrics"""

    def __init__(self):
        """Initialize metrics calculator"""
        self.embedding_model = None
        self.embedding_tokenizer = None
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load model for embedding-based metrics"""
        try:
            # Use a lightweight model for embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)

            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to("cuda")

            logger.info(f"Loaded embedding model: {model_name}")

        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            logger.warning("Embedding-based metrics will not be available")

    def calculate_response_metrics(
        self,
        query: str,
        response: str,
        expected_answer: Optional[str] = None,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a response

        Args:
            query: Original query
            response: Generated response
            expected_answer: Optional expected answer for accuracy metrics
            retrieved_docs: Retrieved documents for context metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic text metrics
        metrics.update(self._calculate_text_metrics(query, response))

        # Response quality metrics
        metrics.update(self._calculate_quality_metrics(response))

        # Context utilization metrics (if retrieved docs provided)
        if retrieved_docs:
            metrics.update(self._calculate_context_metrics(query, retrieved_docs, response))

        # Accuracy metrics (if expected answer provided)
        if expected_answer:
            metrics.update(self._calculate_accuracy_metrics(response, expected_answer))

        return metrics

    def _calculate_text_metrics(self, query: str, response: str) -> Dict[str, Any]:
        """Calculate basic text-based metrics"""
        query_length = len(query.split())
        response_length = len(response.split())
        response_chars = len(response)

        return {
            "query_length": query_length,
            "response_length": response_length,
            "response_chars": response_chars,
            "length_ratio": response_length / query_length if query_length > 0 else 0,
            "response_efficiency": response_chars / response_length if response_length > 0 else 0
        }

    def _calculate_quality_metrics(self, response: str) -> Dict[str, Any]:
        """Calculate response quality metrics"""
        # Calculate basic readability metrics
        sentences = response.split('.')
        avg_sentence_length = len(response.split()) / len(sentences) if sentences else 0

        # Count unique words (lexical diversity)
        words = response.lower().split()
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words) if words else 0

        # Check for repetition
        bigrams = list(zip(words[:-1], words[1:]))
        unique_bigrams = len(set(bigrams))
        repetition_ratio = 1 - (unique_bigrams / len(bigrams)) if bigrams else 0

        return {
            "avg_sentence_length": avg_sentence_length,
            "lexical_diversity": lexical_diversity,
            "repetition_ratio": repetition_ratio,
            "word_count": len(words)
        }

    def _calculate_context_metrics(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        response: str
    ) -> Dict[str, Any]:
        """Calculate context utilization metrics"""
        # Number of documents retrieved
        num_docs = len(retrieved_docs)

        # Average retrieval score
        scores = [doc.get('score', 0) for doc in retrieved_docs]
        avg_score = np.mean(scores) if scores else 0

        # Document coverage (how many docs were actually used)
        # Simple heuristic: check if document keywords appear in response
        doc_keywords_used = 0
        for doc in retrieved_docs:
            doc_text = doc.get('text', '').lower()
            # Extract key terms (simple approach: use first few words)
            key_terms = doc_text.split()[:5]
            for term in key_terms:
                if term in response.lower():
                    doc_keywords_used += 1
                    break

        coverage_ratio = doc_keywords_used / num_docs if num_docs > 0 else 0

        # Query-document relevance
        if self.embedding_model and self.embedding_tokenizer:
            query_embedding = self._get_embedding(query)
            doc_embeddings = [self._get_embedding(doc.get('text', '')) for doc in retrieved_docs]

            similarities = [
                cosine_similarity([query_embedding], [doc_emb])[0][0]
                for doc_emb in doc_embeddings
            ]
            avg_relevance = np.mean(similarities) if similarities else 0
        else:
            avg_relevance = 0

        return {
            "num_retrieved_docs": num_docs,
            "avg_retrieval_score": avg_score,
            "context_coverage_ratio": coverage_ratio,
            "avg_query_doc_relevance": avg_relevance
        }

    def _calculate_accuracy_metrics(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """Calculate accuracy metrics against expected answer"""
        if not self.embedding_model or not self.embedding_tokenizer:
            return {"accuracy_error": "Embedding model not available"}

        try:
            # Calculate semantic similarity
            response_embedding = self._get_embedding(response)
            expected_embedding = self._get_embedding(expected_answer)

            similarity = cosine_similarity([response_embedding], [expected_embedding])[0][0]

            # Keyword overlap (simple metric)
            response_words = set(response.lower().split())
            expected_words = set(expected_answer.lower().split())
            overlap = len(response_words.intersection(expected_words))
            overlap_ratio = overlap / len(expected_words) if expected_words else 0

            return {
                "semantic_similarity": similarity,
                "keyword_overlap_ratio": overlap_ratio,
                "exact_match": response.strip() == expected_answer.strip()
            }

        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            return {"accuracy_error": str(e)}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if not self.embedding_model or not self.embedding_tokenizer:
            raise ValueError("Embedding model not available")

        try:
            # Tokenize
            inputs = self.embedding_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Get embedding
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use mean pooling
                embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()

            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(384)  # Default dimension for MiniLM

    def calculate_latency_metrics(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Calculate latency metrics"""
        total_time = end_time - start_time

        return {
            "total_latency": total_time,
            "latency_category": self._categorize_latency(total_time)
        }

    def _categorize_latency(self, latency: float) -> str:
        """Categorize latency into performance buckets"""
        if latency < 1.0:
            return "excellent"
        elif latency < 3.0:
            return "good"
        elif latency < 5.0:
            return "average"
        elif latency < 10.0:
            return "poor"
        else:
            return "very_poor"

    def calculate_cost_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str
    ) -> Dict[str, Any]:
        """Calculate approximate cost metrics (for API-based models)"""
        # Simplified cost calculation (adjust rates as needed)
        cost_rates = {
            "mistral-7b": {"input": 0.000001, "output": 0.000002},
            "t5-base": {"input": 0.0000005, "output": 0.000001},
            "default": {"input": 0.000001, "output": 0.000002}
        }

        rates = cost_rates.get(model_name, cost_rates["default"])

        input_cost = prompt_tokens * rates["input"]
        output_cost = completion_tokens * rates["output"]
        total_cost = input_cost + output_cost

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_query": total_cost
        }

    def calculate_comprehensive_metrics(
        self,
        query: str,
        response: str,
        expected_answer: Optional[str] = None,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        latency: Optional[float] = None,
        token_counts: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Calculate all metrics in one call"""
        metrics = {}

        # Response metrics
        metrics.update(self.calculate_response_metrics(
            query, response, expected_answer, retrieved_docs
        ))

        # Latency metrics
        if latency:
            metrics.update({"latency": latency})
            metrics.update(self._categorize_latency(latency))

        # Cost metrics
        if token_counts:
            model_info = {"model_name": "unknown"}
            if hasattr(self, 'llm_backend'):
                model_info = self.llm_backend.get_model_info()

            cost_metrics = self.calculate_cost_metrics(
                token_counts.get("prompt", 0),
                token_counts.get("completion", 0),
                model_info.get("model_name", "default")
            )
            metrics.update(cost_metrics)

        # Overall score (simple weighted average)
        metrics["overall_quality_score"] = self._calculate_overall_score(metrics)

        return metrics

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)"""
        score_components = []

        # Response quality (30%)
        if "lexical_diversity" in metrics:
            score_components.append(metrics["lexical_diversity"] * 30)

        # Context utilization (30%)
        if "context_coverage_ratio" in metrics:
            score_components.append(metrics["context_coverage_ratio"] * 30)

        # Accuracy (40%) - if available
        if "semantic_similarity" in metrics:
            score_components.append(metrics["semantic_similarity"] * 40)

        if not score_components:
            return 0.0

        return sum(score_components) / len(score_components) * 100

    def generate_report(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        results = []

        for i, test_case in enumerate(test_cases):
            try:
                metrics = self.calculate_comprehensive_metrics(
                    query=test_case["query"],
                    response=test_case["response"],
                    expected_answer=test_case.get("expected_answer"),
                    retrieved_docs=test_case.get("retrieved_docs"),
                    latency=test_case.get("latency"),
                    token_counts=test_case.get("token_counts")
                )
                results.append({
                    "test_id": i,
                    "query": test_case["query"],
                    "metrics": metrics
                })
            except Exception as e:
                logger.error(f"Error evaluating test case {i}: {e}")
                results.append({
                    "test_id": i,
                    "query": test_case["query"],
                    "error": str(e)
                })

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)

        return {
            "individual_results": results,
            "aggregate_metrics": aggregate_metrics,
            "total_tests": len(test_cases),
            "successful_tests": len([r for r in results if "error" not in r])
        }

    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all test cases"""
        successful_results = [r for r in results if "error" not in r]

        if not successful_results:
            return {"error": "No successful test results"}

        # Extract all metric values
        all_metrics = {}
        for result in successful_results:
            for metric_name, metric_value in result["metrics"].items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

        # Calculate statistics
        aggregate = {}
        for metric_name, values in all_metrics.items():
            aggregate[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }

        return aggregate