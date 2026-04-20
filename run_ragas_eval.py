"""
Entry point for Ragas-based RAG system evaluation.

Quick start
-----------
1. Copy .env.example to .env and fill in OPENROUTER_API_KEY.
2. Install dependencies:
       pip install -r requirements.txt
3. Run with built-in test data (recommended first run):
       python run_ragas_eval.py
4. Re-run without re-indexing documents (faster):
       python run_ragas_eval.py --skip-indexing
5. Use a custom test dataset (JSON file):
       python run_ragas_eval.py --test-data path/to/questions.json
6. Use a local Ollama model for RAG generation instead of OpenRouter:
       python run_ragas_eval.py --rag-model mistral-7b

Custom test dataset format (JSON)
-----------------------------------
The file must be a JSON array of objects with at least these two fields:

    [
      {
        "user_input": "Your question here?",
        "reference": "The ideal / ground-truth answer here."
      },
      ...
    ]

Environment variables (set in .env or system environment)
----------------------------------------------------------
    OPENROUTER_API_KEY   (required) API key from https://openrouter.ai
    OLLAMA_API_BASE      (optional) Ollama server URL, default http://localhost:11434
    LOG_LEVEL            (optional) Logging verbosity, default INFO
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make sure the project root is importable even when called directly
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ragas_eval.py",
        description="Evaluate the RAG system using the Ragas framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help=(
            "Skip document indexing and reuse the vector store from a "
            "previous run.  Useful for faster re-runs with the same data."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        metavar="K",
        help="Number of documents to retrieve per query (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./ragas_results.csv",
        metavar="PATH",
        help="CSV file path for evaluation results (default: %(default)s).",
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="./data/eval/ragas_eval_store",
        metavar="DIR",
        help=(
            "Persistent directory for the evaluation ChromaDB collection "
            "(default: %(default)s).  Isolated from the main vector store."
        ),
    )
    parser.add_argument(
        "--rag-model",
        type=str,
        default="openrouter",
        choices=["openrouter", "mistral-7b", "llama2-7b", "t5-base"],
        metavar="MODEL",
        help=(
            "LLM backend used for RAG answer generation during evaluation. "
            "Choices: openrouter (default), mistral-7b, llama2-7b, t5-base. "
            "Ollama must be running locally for mistral-7b / llama2-7b."
        ),
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help=(
            "Path to a JSON file with custom test questions and reference answers. "
            "If omitted, the built-in NLP/RAG domain dataset is used."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_test_data(path: str):
    """Load and validate a custom test dataset from a JSON file."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Test data file not found: {p}")
        sys.exit(1)

    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list) or not data:
        print("[ERROR] Test data file must be a non-empty JSON array.")
        sys.exit(1)

    for i, item in enumerate(data):
        if "user_input" not in item or "reference" not in item:
            print(
                f"[ERROR] Item {i} is missing 'user_input' or 'reference' key. "
                "See --help for the expected format."
            )
            sys.exit(1)

    return data


def _configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    _configure_logging()
    args = _parse_args()

    # Validate API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print(
            "\n[ERROR] OPENROUTER_API_KEY is not set.\n"
            "Please add the following line to your .env file (copy .env.example):\n\n"
            "    OPENROUTER_API_KEY=sk-or-v1-<your-key>\n"
        )
        sys.exit(1)

    # Load custom test data if provided
    test_data = None
    if args.test_data:
        print(f"Loading custom test data from: {args.test_data}")
        test_data = _load_test_data(args.test_data)
        print(f"  → {len(test_data)} test sample(s) loaded.\n")

    # Import here so the project root sys.path insertion takes effect first
    from src.evaluation.ragas_eval import RagasEvaluator  # noqa: PLC0415

    print("=" * 68)
    print("  CS6493 RAG System – Ragas Evaluation")
    print("=" * 68)
    print(f"  RAG generation model : {args.rag_model}")
    print(f"  Ragas judge LLM      : NVIDIA Nemotron-3 Super 120B (OpenRouter)")
    print(f"  Ragas embedding      : sentence-transformers/all-MiniLM-L6-v2")
    print(f"  Retrieve top-k       : {args.top_k}")
    print(f"  Vector store dir     : {args.vector_store_dir}")
    print(f"  Output CSV           : {args.output}")
    print(f"  Skip indexing        : {args.skip_indexing}")
    print("=" * 68 + "\n")

    evaluator = RagasEvaluator(
        openrouter_api_key=api_key,
        rag_model=args.rag_model,
        output_path=args.output,
        vector_store_dir=args.vector_store_dir,
        top_k=args.top_k,
    )

    evaluator.run(test_data=test_data, skip_indexing=args.skip_indexing)


if __name__ == "__main__":
    main()
