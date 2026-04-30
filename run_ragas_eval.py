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
  • Chunk documents with the same strategy as the main RAG system
  • Index the chunks and answer every question with the RAG pipeline
  • Score the answers with 4 Ragas metrics
  • Print results and save them to  data/eval/ragas_eval_results/
    with a timestamped filename

Options
-------
    --docs-dir           PATH    Folder with PDF/text files (default: ./data/eval/ragas_docs/)
    --testset-size       N       Number of questions to generate (default: 8)
    --top-k              K       Documents retrieved per question (default: 3)
    --output             PATH    Output directory or CSV path
                                 (default: ./data/eval/ragas_eval_results/)
    --rag-model          MODEL   mistral-7b | llama2-7b | t5-base  (default: mistral-7b)
    --chunking-strategy  STR     fixed | sentence  (default: inherits Config.CHUNKING.strategy)
    --skip-indexing              Reuse vector store from a previous run (faster re-runs)
    --vector-store-dir   DIR     Isolated ChromaDB directory (default: ./data/eval/ragas_eval_store)

Supported document formats
---------------------------
    .pdf   .txt   .md

Model configuration (hard-coded, local Ollama)
---------------------------------
    generator_llm:
        mistral:7b  (via Ollama)
    critic/transforms_llm:
        llama2:7b   (via Ollama)
    Ragas judge:
        mistral:7b  (via Ollama)
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
        default=5,
        metavar="K",
        help="Documents retrieved per query (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/eval/ragas_eval_results",
        metavar="PATH",
        help=(
            "Output directory or CSV path for evaluation results. "
            "A timestamp will be appended to the filename automatically "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--rag-model",
        type=str,
        default="mistral-7b",
        choices=["mistral-7b", "llama2-7b", "t5-base"],
        metavar="MODEL",
        help=(
            "LLM backend for RAG answer generation. "
            "Choices: mistral-7b (default), llama2-7b, t5-base."
        ),
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default=None,
        choices=["fixed", "sentence", "hierarchical"],
        metavar="STR",
        help=(
            "Chunking strategy used when indexing documents into the evaluation "
            "vector store.  Must match what the main RAG system uses so that the "
            "evaluation reflects real retrieval behaviour. "
            "Choices: fixed, sentence.  Default: inherit from Config.CHUNKING.strategy."
        ),
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Reuse the vector store from a previous run (skips document indexing).",
    )
    parser.add_argument(
        "--clear-vector-store",
        action="store_true",
        help="Clear the vector store before indexing (removes all existing data).",
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
    parser.add_argument(
        "--export-testset",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Export the generated test set to a CSV file at the specified path. "
            "If this option is provided, the test set will be generated and saved, "
            "then the evaluation will proceed normally."
        ),
    )
    parser.add_argument(
        "--import-testset",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Import a previously generated test set from a CSV file. "
            "If this option is provided, test set generation will be skipped "
            "and the imported questions will be used for evaluation."
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

    from src.evaluation.ragas_eval import (  # noqa: PLC0415
        RagasEvaluator,
        _OLLAMA_BASE_URL,
        _OLLAMA_CRITIC_MODEL,
        _OLLAMA_GENERATOR_MODEL,
        _OLLAMA_JUDGE_MODEL,
        _RAGAS_DOCS_PER_QUESTION,
        _RAGAS_MAX_TESTSET_DOCS,
        _RAGAS_MAX_WORKERS,
        _RAGAS_PARSE_MIN_DOCS,
        import_testset_from_csv,
        export_testset_to_csv,
    )
    from src.config import Config  # noqa: PLC0415

    docs_path = Path(args.docs_dir)

    print()
    print("=" * 72)
    print("  CS6493 RAG System – Ragas Evaluation  (local Ollama)")
    print("=" * 72)
    effective_chunking = args.chunking_strategy or Config.CHUNKING.strategy
    print(f"  Document folder  : {docs_path.resolve()}")
    print(f"  Testset size     : {args.testset_size} questions (auto-generated)")
    print(f"  generator LLM    : {_OLLAMA_GENERATOR_MODEL}  (T=0.4, via Ollama)")
    print(f"  critic LLM       : {_OLLAMA_CRITIC_MODEL}  (T=0.0, via Ollama)")
    print(f"  Ragas judge LLM  : {_OLLAMA_JUDGE_MODEL}  (T=0.0, via Ollama)")
    print(f"  RAG answer model : {args.rag_model}")
    print(f"  Embedding model  : {Config.EMBEDDING_MODEL}")
    print(f"  Chunking strategy: {effective_chunking} "
          f"(chunk_size={Config.CHUNKING.chunk_size}, overlap={Config.CHUNKING.chunk_overlap})")
    print(f"  Ollama server    : {_OLLAMA_BASE_URL}")
    print(f"  Retrieve top-k   : {args.top_k}")
    print(f"  Vector store     : {args.vector_store_dir}")
    print(f"  Output CSV       : {args.output}")
    print(f"  Skip indexing    : {args.skip_indexing}")
    print("  Stability config : "
          f"RAGAS_MAX_WORKERS={_RAGAS_MAX_WORKERS}, "
          f"RAGAS_DOCS_PER_QUESTION={_RAGAS_DOCS_PER_QUESTION}, "
          f"RAGAS_MAX_TESTSET_DOCS={_RAGAS_MAX_TESTSET_DOCS}, "
          f"RAGAS_PARSE_MIN_DOCS={_RAGAS_PARSE_MIN_DOCS}")
    print("=" * 72)

    # Check docs folder exists and warn if empty before starting
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[WARNING] Folder created but is empty: {docs_path.resolve()}\n"
            "  Please add PDF or text files to this folder and re-run.\n"
        )
        sys.exit(1)

    # Handle test set import if specified
    imported_test_data = None
    if args.import_testset:
        imported_test_data = import_testset_from_csv(args.import_testset)
        print(f"  Testset import  : {args.import_testset} ({len(imported_test_data)} questions)")
        print(f"  Testset size     : {len(imported_test_data)} questions (imported)")
    else:
        print(f"  Testset size     : {args.testset_size} questions (auto-generated)")

    if args.export_testset:
        print(f"  Testset export  : {args.export_testset}")

    evaluator = RagasEvaluator(
        docs_dir=args.docs_dir,
        rag_model=args.rag_model,
        output_path=args.output,
        vector_store_dir=args.vector_store_dir,
        top_k=args.top_k,
        testset_size=args.testset_size,
        chunking_strategy=args.chunking_strategy,
    )

    # Run evaluation with optional imported test data
    results_df = evaluator.run(skip_indexing=args.skip_indexing, test_data=imported_test_data)

    # Export test set if requested (only if we generated it, not imported)
    if args.export_testset and not args.import_testset:
        export_testset_to_csv(
            test_data=[{
                "user_input": row.get("user_input", ""),
                "retrieved_contexts": row.get("retrieved_contexts", ""),
                "response": row.get("response", ""),
                "reference": row.get("reference", "")
            } for _, row in results_df.iterrows()],
            output_path=args.export_testset,
            document_count=len(evaluator._source_pdf_files) if evaluator._source_pdf_files else 0
        )


if __name__ == "__main__":
    main()
