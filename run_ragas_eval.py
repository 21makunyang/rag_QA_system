"""
Entry point for the Ragas-based RAG system evaluation.

Quick start
-----------
1. Place your PDF files in:   data/eval/ragas_docs/
2. Run:
       python run_ragas_eval.py

That's it.  The script will:
  • Load all PDFs from the folder
  • Auto-generate test questions with Ragas (generator_llm + critic_llm)
  • Index the documents and answer every question with the RAG pipeline
  • Score the answers with 4 Ragas metrics
  • Print results and save them to  ragas_results.csv

Options
-------
    --docs-dir      PATH    Folder with PDF/text files (default: ./data/eval/ragas_docs/)
    --testset-size  N       Number of questions to generate (default: 8)
    --top-k         K       Documents retrieved per question (default: 3)
    --output        PATH    CSV output path (default: ./ragas_results.csv)
    --rag-model     MODEL   openrouter | mistral-7b | llama2-7b | t5-base  (default: openrouter)
    --skip-indexing         Reuse vector store from a previous run (faster re-runs)
    --vector-store-dir DIR  Isolated ChromaDB directory (default: ./data/eval/ragas_eval_store)

Supported document formats
---------------------------
    .pdf   .txt   .md

Model configuration (hard-coded)
---------------------------------
    generator_llm / critic_llm / Ragas judge:
        NVIDIA Nemotron-3 Super 120B  (nvidia/nemotron-3-super-120b-a12b:free)
        via OpenRouter  (https://openrouter.ai/api/v1)
    Embedding:
        sentence-transformers/all-MiniLM-L6-v2  (local HuggingFace)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ragas_eval.py",
        description="Evaluate the RAG system with Ragas (PDF folder → auto testset → metrics).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="./data/eval/ragas_docs",
        metavar="PATH",
        help=(
            "Folder containing the PDF / text files used for evaluation. "
            "All .pdf, .txt, and .md files are loaded recursively "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=8,
        metavar="N",
        help="Number of test questions to auto-generate (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        metavar="K",
        help="Documents retrieved per query (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./ragas_results.csv",
        metavar="PATH",
        help="CSV file for evaluation results (default: %(default)s).",
    )
    parser.add_argument(
        "--rag-model",
        type=str,
        default="openrouter",
        choices=["openrouter", "mistral-7b", "llama2-7b", "t5-base"],
        metavar="MODEL",
        help=(
            "LLM backend for RAG answer generation. "
            "Choices: openrouter (default), mistral-7b, llama2-7b, t5-base."
        ),
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Reuse the vector store from a previous run (skips document indexing).",
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="./data/eval/ragas_eval_store",
        metavar="DIR",
        help=(
            "Isolated ChromaDB directory for the evaluation collection "
            "(default: %(default)s)."
        ),
    )
    return parser.parse_args()


def _configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _configure_logging()
    args = _parse_args()

    from src.evaluation.ragas_eval import RagasEvaluator, _OPENROUTER_MODEL  # noqa: PLC0415
    from src.config import Config  # noqa: PLC0415

    docs_path = Path(args.docs_dir)

    print()
    print("=" * 72)
    print("  CS6493 RAG System – Ragas Evaluation")
    print("=" * 72)
    print(f"  Document folder  : {docs_path.resolve()}")
    print(f"  Testset size     : {args.testset_size} questions (auto-generated)")
    print(f"  generator_llm    : {_OPENROUTER_MODEL}  (T=0.4)")
    print(f"  critic_llm       : {_OPENROUTER_MODEL}  (T=0.0)")
    print(f"  RAG answer model : {args.rag_model}")
    print(f"  Embedding model  : {Config.EMBEDDING_MODEL}")
    print(f"  Retrieve top-k   : {args.top_k}")
    print(f"  Vector store     : {args.vector_store_dir}")
    print(f"  Output CSV       : {args.output}")
    print(f"  Skip indexing    : {args.skip_indexing}")
    print("=" * 72)

    # Check docs folder exists and warn if empty before starting
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[WARNING] Folder created but is empty: {docs_path.resolve()}\n"
            "  Please add PDF or text files to this folder and re-run.\n"
        )
        sys.exit(1)

    evaluator = RagasEvaluator(
        docs_dir=args.docs_dir,
        rag_model=args.rag_model,
        output_path=args.output,
        vector_store_dir=args.vector_store_dir,
        top_k=args.top_k,
        testset_size=args.testset_size,
    )
    evaluator.run(skip_indexing=args.skip_indexing)


if __name__ == "__main__":
    main()
