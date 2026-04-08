"""
Test cases for evaluating LLM applications
"""

import logging
from typing import List, Dict, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TestCaseManager:
    """Manager for evaluation test cases"""

    def __init__(self):
        """Initialize test case manager"""
        self.test_cases = []
        self.categories = {
            "factual_qa": "Factual Question Answering",
            "summarization": "Document Summarization",
            "inference": "Inference and Reasoning",
            "creative": "Creative Tasks",
            "edge_cases": "Edge Cases and Error Handling"
        }

    def add_test_case(
        self,
        query: str,
        expected_answer: str,
        category: str = "factual_qa",
        difficulty: str = "medium",
        metadata: Dict[str, Any] = None
    ):
        """
        Add a test case

        Args:
            query: Test query
            expected_answer: Expected answer
            category: Test category
            difficulty: Difficulty level (easy, medium, hard)
            metadata: Additional metadata
        """
        test_case = {
            "id": len(self.test_cases) + 1,
            "query": query,
            "expected_answer": expected_answer,
            "category": category,
            "difficulty": difficulty,
            "metadata": metadata or {}
        }

        self.test_cases.append(test_case)
        logger.info(f"Added test case {test_case['id']}: {query[:50]}...")

    def add_test_cases_from_list(self, test_cases: List[Dict[str, Any]]):
        """Add multiple test cases from a list"""
        for test_case in test_cases:
            self.add_test_case(**test_case)

    def add_factual_qa_cases(self):
        """Add predefined factual QA test cases"""
        cases = [
            {
                "query": "What is the capital of France?",
                "expected_answer": "Paris",
                "category": "factual_qa",
                "difficulty": "easy",
                "metadata": {"topic": "geography"}
            },
            {
                "query": "Who wrote 'To Kill a Mockingbird'?",
                "expected_answer": "Harper Lee",
                "category": "factual_qa",
                "difficulty": "easy",
                "metadata": {"topic": "literature"}
            },
            {
                "query": "What is the chemical symbol for gold?",
                "expected_answer": "Au",
                "category": "factual_qa",
                "difficulty": "easy",
                "metadata": {"topic": "chemistry"}
            },
            {
                "query": "What year did World War II end?",
                "expected_answer": "1945",
                "category": "factual_qa",
                "difficulty": "medium",
                "metadata": {"topic": "history"}
            },
            {
                "query": "Explain the theory of relativity in simple terms.",
                "expected_answer": "The theory of relativity describes how space and time are connected and how gravity works.",
                "category": "factual_qa",
                "difficulty": "hard",
                "metadata": {"topic": "physics"}
            }
        ]

        self.add_test_cases_from_list(cases)

    def add_summarization_cases(self):
        """Add predefined summarization test cases"""
        cases = [
            {
                "query": "Summarize the main points of the document.",
                "expected_answer": "A concise summary of key points.",
                "category": "summarization",
                "difficulty": "medium",
                "metadata": {
                    "document_type": "general",
                    "expected_length": "short"
                }
            },
            {
                "query": "What are the main themes discussed?",
                "expected_answer": "Key themes and concepts identified.",
                "category": "summarization",
                "difficulty": "hard",
                "metadata": {
                    "document_type": "academic",
                    "expected_length": "medium"
                }
            }
        ]

        self.add_test_cases_from_list(cases)

    def add_inference_cases(self):
        """Add predefined inference test cases"""
        cases = [
            {
                "query": "Based on the information, what can we conclude?",
                "expected_answer": "Logical conclusion based on the context.",
                "category": "inference",
                "difficulty": "medium",
                "metadata": {"reasoning_type": "deductive"}
            },
            {
                "query": "What are the implications of this finding?",
                "expected_answer": "Analysis of implications and consequences.",
                "category": "inference",
                "difficulty": "hard",
                "metadata": {"reasoning_type": "analytical"}
            }
        ]

        self.add_test_cases_from_list(cases)

    def add_edge_cases(self):
        """Add predefined edge case test cases"""
        cases = [
            {
                "query": "",
                "expected_answer": "Please provide a valid question.",
                "category": "edge_cases",
                "difficulty": "easy",
                "metadata": {"case_type": "empty_query"}
            },
            {
                "query": "What is the meaning of life?",
                "expected_answer": "The meaning of life is subjective and varies between individuals.",
                "category": "edge_cases",
                "difficulty": "hard",
                "metadata": {"case_type": "philosophical"}
            },
            {
                "query": "A" * 1000,
                "expected_answer": "Your query is too long. Please shorten it.",
                "category": "edge_cases",
                "difficulty": "medium",
                "metadata": {"case_type": "very_long_query"}
            },
            {
                "query": "你好，世界！",
                "expected_answer": "Hello, world!",
                "category": "edge_cases",
                "difficulty": "easy",
                "metadata": {"case_type": "non_english"}
            }
        ]

        self.add_test_cases_from_list(cases)

    def get_test_cases(
        self,
        category: str = None,
        difficulty: str = None,
        max_cases: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get test cases with optional filtering

        Args:
            category: Filter by category
            difficulty: Filter by difficulty
            max_cases: Maximum number of cases to return

        Returns:
            Filtered test cases
        """
        filtered_cases = self.test_cases

        if category:
            filtered_cases = [tc for tc in filtered_cases if tc["category"] == category]

        if difficulty:
            filtered_cases = [tc for tc in filtered_cases if tc["difficulty"] == difficulty]

        if max_cases and max_cases < len(filtered_cases):
            filtered_cases = filtered_cases[:max_cases]

        return filtered_cases

    def get_test_case_by_id(self, test_id: int) -> Dict[str, Any]:
        """Get test case by ID"""
        for test_case in self.test_cases:
            if test_case["id"] == test_id:
                return test_case
        raise ValueError(f"Test case {test_id} not found")

    def save_test_cases(self, file_path: str):
        """Save test cases to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.test_cases, f, indent=2)

            logger.info(f"Saved {len(self.test_cases)} test cases to {file_path}")

        except Exception as e:
            logger.error(f"Error saving test cases: {e}")
            raise

    def load_test_cases(self, file_path: str):
        """Load test cases from JSON file"""
        try:
            with open(file_path, 'r') as f:
                self.test_cases = json.load(f)

            logger.info(f"Loaded {len(self.test_cases)} test cases from {file_path}")

        except FileNotFoundError:
            logger.warning(f"Test case file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            raise

    def generate_test_suite(self) -> Dict[str, Any]:
        """Generate a comprehensive test suite"""
        self.add_factual_qa_cases()
        self.add_summarization_cases()
        self.add_inference_cases()
        self.add_edge_cases()

        return {
            "total_cases": len(self.test_cases),
            "categories": self.categories,
            "cases_by_category": {
                category: len(self.get_test_cases(category=category))
                for category in self.categories.keys()
            },
            "cases_by_difficulty": {
                "easy": len(self.get_test_cases(difficulty="easy")),
                "medium": len(self.get_test_cases(difficulty="medium")),
                "hard": len(self.get_test_cases(difficulty="hard"))
            }
        }

    def export_for_evaluation(self, output_dir: str = "./data/results"):
        """Export test cases for evaluation"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save test cases
        self.save_test_cases(f"{output_dir}/test_cases.json")

        # Generate summary
        summary = self.generate_test_suite()
        with open(f"{output_dir}/test_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported test suite to {output_dir}")
        return summary