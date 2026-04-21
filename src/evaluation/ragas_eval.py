"""
Ragas-based evaluation module for the RAG QA system.
Requires: ragas>=0.2.0

Pipeline
--------
1. Load PDF / text files from a user-specified folder.
   PDFs are split into per-page chunks so Ragas can generate
   diverse questions across the whole document.
2. Ragas TestsetGenerator automatically builds a labelled test dataset.
   - llm (generator, T=0.4) : generates questions from the corpus.
   - transforms_llm (critic, T=0.0) : filters / refines the questions.
3. Documents are indexed into an isolated ChromaDB collection.
4. Each generated question is answered by the RAG pipeline
   (Retriever → ResponseGenerator).
5. Ragas evaluates all samples with four metrics:
   Faithfulness · Answer Relevancy · Context Precision · Context Recall
6. Results are printed to the console and exported to CSV.

LLM config (hard-coded)
-----------------------
  generator LLM / critic (transforms) LLM / RAG LLM / judge:
      NVIDIA Nemotron-3 Super 120B  (nvidia/nemotron-3-super-120b-a12b:free)
      via OpenRouter  (https://openrouter.ai/api/v1)
  Embedding:
      sentence-transformers/all-MiniLM-L6-v2  (local HuggingFace)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter credentials (API key embedded directly as requested)
# ─────────────────────────────────────────────────────────────────────────────

# 下面两个模型任选其一
# NVIDIA: Nemotron 3 Super (free)
_OPENROUTER_API_KEY = (
    "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"
)
_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# # Google: Gemma 4 31B (free)
# _OPENROUTER_API_KEY = (
#     "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"
# )
# _OPENROUTER_MODEL = "google/gemma-4-31b-it:free"

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Model backend configuration for evaluation.


# Default folder where the user drops evaluation PDF / text files
_DEFAULT_DOCS_DIR = "./data/eval/ragas_docs"

# Supported file extensions
_SUPPORTED_EXTS = {".pdf", ".txt", ".md"}


# ─────────────────────────────────────────────────────────────────────────────
# Document loading: PDF folder → LlamaIndex docs + LangChain docs
# ─────────────────────────────────────────────────────────────────────────────


def load_documents_from_folder(docs_dir: str) -> Tuple[list, list]:
    """
    Scan *docs_dir* for PDF / text files and return two parallel lists:

    - **llamaindex_docs** : ``llama_index.core.Document`` objects for RAG indexing.
    - **langchain_docs**  : ``langchain_core.documents.Document`` objects for
      the Ragas TestsetGenerator.

    PDFs are split page-by-page so Ragas receives many short, focused chunks
    rather than one giant document—this produces more varied test questions.

    Raises
    ------
    FileNotFoundError  – folder does not exist.
    ValueError         – no supported files found.
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(
            f"Document folder not found: {docs_path.resolve()}\n"
            "Please create the folder and add PDF or text files to it."
        )

    files = sorted(
        f
        for f in docs_path.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS
    )

    if not files:
        raise ValueError(
            f"No supported files found in {docs_path.resolve()}\n"
            f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTS))}\n"
            "Please add at least one PDF or text file and re-run."
        )

    logger.info("Found %d file(s) in %s", len(files), docs_path.resolve())
    for f in files:
        logger.info("  %s", f.name)

    llamaindex_docs: list = []
    for file_path in files:
        loaded = _load_single_file(file_path)
        llamaindex_docs.extend(loaded)
        logger.info("  Loaded %d chunk(s) from %s", len(loaded), file_path.name)

    if not llamaindex_docs:
        raise ValueError("All files were loaded but produced no text content.")

    logger.info("Total: %d document chunk(s) loaded.", len(llamaindex_docs))

    langchain_docs = _to_langchain_docs(llamaindex_docs)
    logger.info("Converted to %d LangChain document(s).", len(langchain_docs))
    return llamaindex_docs, langchain_docs


def _load_single_file(file_path: Path) -> list:
    """Dispatch to the appropriate loader based on file extension."""
    return _load_pdf(file_path) if file_path.suffix.lower() == ".pdf" else _load_text_file(file_path)


def _load_pdf(file_path: Path) -> list:
    """
    Load a PDF and return one LlamaIndex Document per page where possible.

    Loader priority
    ---------------
    1. pdfminer extract_pages  – page-by-page, best for Ragas diversity
    2. pdfminer extract_text   – whole-document fallback (same library,
                                  avoids the llama_index.readers dependency)
    3. LlamaIndex PDFReader    – last resort if pdfminer is unavailable
    """
    from llama_index.core import Document as LlamaDocument  # noqa: PLC0415

    base_meta = {"file_name": file_path.name, "file_path": str(file_path), "source": file_path.name}

    # ── Loader 1: pdfminer page-by-page ─────────────────────────────────────
    try:
        from pdfminer.high_level import extract_pages  # noqa: PLC0415
        from pdfminer.layout import LTTextContainer  # noqa: PLC0415  (LTAnon removed in newer pdfminer)

        pages = []
        for page_num, page_layout in enumerate(extract_pages(str(file_path)), start=1):
            page_text = "".join(
                el.get_text()
                for el in page_layout
                if isinstance(el, LTTextContainer)
            ).strip()
            if page_text:
                pages.append(
                    LlamaDocument(text=page_text, metadata={**base_meta, "page": page_num})
                )

        if pages:
            logger.info("pdfminer extracted %d page(s) from %s", len(pages), file_path.name)
            return pages
        logger.warning("pdfminer found no text pages in %s – trying whole-doc extraction", file_path.name)
    except Exception as err:
        logger.warning("pdfminer page-by-page failed for %s: %s – trying whole-doc extraction", file_path.name, err)

    # ── Loader 2: pdfminer whole-document ────────────────────────────────────
    try:
        from pdfminer.high_level import extract_text  # noqa: PLC0415

        text = extract_text(str(file_path)).strip()
        if text:
            logger.info("pdfminer whole-doc extracted %d chars from %s", len(text), file_path.name)
            return [LlamaDocument(text=text, metadata=base_meta)]
        logger.warning("pdfminer whole-doc extracted no text from %s", file_path.name)
    except Exception as err:
        logger.warning("pdfminer whole-doc failed for %s: %s", file_path.name, err)

    # ── Loader 3: LlamaIndex PDFReader (last resort) ─────────────────────────
    try:
        from src.ingestion.connectors import PDFConnector  # noqa: PLC0415

        docs = PDFConnector().load(str(file_path))
        if docs:
            return docs
    except Exception as err:
        logger.error("LlamaIndex PDFConnector failed for %s: %s", file_path.name, err)

    logger.error("All PDF loaders failed for %s – file will be skipped.", file_path.name)
    return []


def _load_text_file(file_path: Path) -> list:
    """Load a .txt / .md file using the project's TextFileConnector."""
    try:
        from src.ingestion.connectors import TextFileConnector  # noqa: PLC0415

        return TextFileConnector().load(str(file_path))
    except Exception as err:
        logger.error("TextFileConnector failed for %s: %s", file_path.name, err)
        return []


def _to_langchain_docs(llamaindex_docs: list) -> list:
    """Convert LlamaIndex Documents to LangChain Documents for Ragas."""
    try:
        from langchain_core.documents import Document as LCDocument  # noqa: PLC0415
    except ImportError:
        from langchain.schema import Document as LCDocument  # type: ignore[no-redef]  # noqa: PLC0415

    result = []
    for doc in llamaindex_docs:
        text = (doc.text or "").strip()
        if not text:
            continue
        metadata = dict(doc.metadata) if hasattr(doc, "metadata") else {}
        metadata.setdefault("source", metadata.get("file_name", "unknown"))
        result.append(LCDocument(page_content=text, metadata=metadata))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter LLM backend (compatible with OllamaBackend / HuggingFaceBackend)
# ─────────────────────────────────────────────────────────────────────────────


class OpenRouterLLMBackend:
    """
    Wraps the OpenRouter API and implements the same interface as
    OllamaBackend so it can be passed to the existing ResponseGenerator.
    """

    def __init__(
        self,
        model_name: str = _OPENROUTER_MODEL,
        api_key: str = _OPENROUTER_API_KEY,
        base_url: str = _OPENROUTER_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Run: pip install openai") from exc
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def generate_stream(self, prompt: str):
        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def get_model_info(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "backend": "openrouter", "base_url": self.base_url}


# ─────────────────────────────────────────────────────────────────────────────
# Shared LangChain / Ragas model builders
# ─────────────────────────────────────────────────────────────────────────────


def _langchain_llm(temperature: float):
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    return ChatOpenAI(
        model=_OPENROUTER_MODEL,
        api_key=_OPENROUTER_API_KEY,
        base_url=_OPENROUTER_BASE_URL,
        temperature=temperature,
    )


def _langchain_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
    from src.config import Config  # noqa: PLC0415

    return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)


def _ragas_llm(temperature: float):
    """Return a Ragas-wrapped ChatOpenAI instance."""
    from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415

    return LangchainLLMWrapper(_langchain_llm(temperature))


def _ragas_embeddings():
    """Return a Ragas-wrapped HuggingFace embedding model."""
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415

    return LangchainEmbeddingsWrapper(_langchain_embeddings())


# ─────────────────────────────────────────────────────────────────────────────
# Testset generation  (Ragas 0.2.x API)
# ─────────────────────────────────────────────────────────────────────────────


def generate_testset(langchain_docs: list, testset_size: int = 8) -> List[Dict[str, str]]:
    """
    Automatically generate a labelled test dataset from *langchain_docs*
    using the Ragas 0.2.x TestsetGenerator.

    Two separate LLM roles are used:
    - **generator LLM** (temperature=0.4) : synthesises diverse questions
      from the source documents.
    - **transforms LLM / critic** (temperature=0.0) : applied during the
      knowledge-graph build phase to filter and refine the extracted
      propositions before questions are generated from them.

    Parameters
    ----------
    langchain_docs : LangChain Documents from ``load_documents_from_folder()``.
    testset_size   : Number of test samples to generate.

    Returns
    -------
    List of ``{"user_input": ..., "reference": ...}`` dicts.
    """
    from ragas.testset import TestsetGenerator  # noqa: PLC0415

    generator_llm = _ragas_llm(temperature=0.4)
    critic_llm = _ragas_llm(temperature=0.0)
    emb = _ragas_embeddings()

    generator = TestsetGenerator(llm=generator_llm, embedding_model=emb)

    # transforms_llm acts as the critic: it processes the knowledge graph
    # (proposition extraction, filtering) at a lower temperature for quality.
    try:
        testset = generator.generate_with_langchain_docs(
            langchain_docs,
            testset_size=testset_size,
            transforms_llm=critic_llm,
            transforms_embedding_model=emb,
        )
    except TypeError:
        # Older 0.2.x builds that don't accept transforms_llm yet
        logger.info("transforms_llm not supported in this build; using single LLM.")
        testset = generator.generate_with_langchain_docs(
            langchain_docs, testset_size=testset_size
        )

    df = testset.to_pandas()
    logger.info("TestsetGenerator produced %d rows; columns: %s", len(df), list(df.columns))

    samples: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        # 0.2.x uses user_input / reference; guard against missing columns
        question = row.get("user_input") or row.get("question", "")
        ground_truth = row.get("reference") or row.get("ground_truth", "")
        if pd.notna(question) and pd.notna(ground_truth) and question and ground_truth:
            samples.append({"user_input": str(question), "reference": str(ground_truth)})

    logger.info("%d valid test samples extracted.", len(samples))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Ragas evaluation  (Ragas 0.2.x API)
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_with_ragas(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Score collected RAG outputs with four Ragas 0.2.x metrics.

    Uses EvaluationDataset / SingleTurnSample and class-based metrics
    with an explicit judge LLM and embedding model.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: PLC0415
    from ragas.metrics import (  # noqa: PLC0415
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    judge_llm = _ragas_llm(temperature=0.0)
    judge_emb = _ragas_embeddings()

    ragas_samples = [
        SingleTurnSample(
            user_input=s["user_input"],
            retrieved_contexts=s["retrieved_contexts"],
            response=s["response"],
            reference=s["reference"],
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_emb),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    logger.info("Running Ragas 0.2.x evaluation on %d samples ...", len(samples))
    result = evaluate(dataset=dataset, metrics=metrics)
    return result.to_pandas()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator class
# ─────────────────────────────────────────────────────────────────────────────


class RagasEvaluator:
    """
    End-to-end Ragas 0.2.x evaluation pipeline for the RAG QA system.

    Usage
    -----
    1. Drop PDF (or .txt / .md) files into ``docs_dir``
       (default: ``./data/eval/ragas_docs/``).
    2. Run ``python run_ragas_eval.py``.

    Steps
    -----
    1  Load documents from folder (page-by-page PDF splitting).
    2  TestsetGenerator (generator LLM + transforms/critic LLM) creates
       a labelled question set from the documents.
    3  Documents are indexed into an isolated ChromaDB collection.
    4  Each question is answered by the RAG pipeline.
    5  Four Ragas metrics score the answers.
    6  Results are printed and saved to CSV.
    """

    def __init__(
        self,
        docs_dir: str = _DEFAULT_DOCS_DIR,
        rag_model: str = "openrouter",
        output_path: str = "./ragas_results.csv",
        vector_store_dir: str = "./data/eval/ragas_eval_store",
        collection_name: str = "ragas_eval_collection",
        top_k: int = 3,
        testset_size: int = 8,
    ) -> None:
        """
        Parameters
        ----------
        docs_dir         : Folder with PDF / text evaluation files.
        rag_model        : ``"openrouter"`` | ``"mistral-7b"`` |
                           ``"llama2-7b"`` | ``"t5-base"``
        output_path      : Destination CSV file.
        vector_store_dir : Isolated ChromaDB directory (separate from the
                           main application store).
        collection_name  : ChromaDB collection name for this evaluation run.
        top_k            : Documents retrieved per question.
        testset_size     : Number of test questions to auto-generate.
        """
        self.docs_dir = docs_dir
        self.rag_model_name = rag_model
        self.output_path = output_path
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.testset_size = testset_size
        self._retriever = None
        self._response_gen = None

    # ── RAG system ───────────────────────────────────────────────────────────

    def _build_rag_llm(self):
        if self.rag_model_name == "openrouter":
            return OpenRouterLLMBackend()
        from src.config import Config  # noqa: PLC0415
        model_config = Config.get_model_config(self.rag_model_name)
        if model_config.backend == "ollama":
            from src.models.ollama_backend import OllamaBackend  # noqa: PLC0415
            return OllamaBackend(model_config)
        from src.models.huggingface_backend import HuggingFaceBackend  # noqa: PLC0415
        return HuggingFaceBackend(model_config)

    def _init_rag_system(self) -> None:
        from src.config import VectorStoreConfig  # noqa: PLC0415
        from src.query.response_gen import ResponseGenerator  # noqa: PLC0415
        from src.query.retriever import Retriever  # noqa: PLC0415

        vs_config = VectorStoreConfig(
            store_type="chroma",
            persist_dir=self.vector_store_dir,
            collection_name=self.collection_name,
        )
        logger.info("Initialising evaluation RAG system ...")
        self._retriever = Retriever(vs_config)
        self._response_gen = ResponseGenerator(self._build_rag_llm(), self._retriever)
        logger.info("RAG system ready.")

    def _index_documents(self, llamaindex_docs: list) -> None:
        logger.info("Indexing %d document chunk(s) ...", len(llamaindex_docs))
        self._retriever.index_documents(llamaindex_docs)
        logger.info("Indexing complete.")

    # ── Sample collection ────────────────────────────────────────────────────

    def _collect_samples(self, test_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        total = len(test_data)
        for idx, item in enumerate(test_data, start=1):
            question, reference = item["user_input"], item["reference"]
            logger.info("  [%d/%d] %s", idx, total, question[:70])
            try:
                result = self._response_gen.generate_response(
                    query=question, top_k=self.top_k, use_rag=True
                )
                contexts = [doc["text"] for doc in result.get("retrieved_docs", [])]
                answer = result.get("answer", "")
                samples.append(
                    {
                        "user_input": question,
                        "retrieved_contexts": contexts,
                        "response": answer,
                        "reference": reference,
                    }
                )
                logger.info("    → %d context(s), answer: %d chars", len(contexts), len(answer))
            except Exception as exc:
                logger.error("    ✗ Skipped: %s", exc)
        return samples

    # ── Output ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_results(df: pd.DataFrame) -> None:
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        display_cols = [c for c in metric_cols if c in df.columns]
        question_col = next((c for c in ("user_input", "question") if c in df.columns), None)
        show_cols = ([question_col] if question_col else []) + display_cols

        sep = "=" * 72
        print(f"\n{sep}")
        print("  RAGAS EVALUATION RESULTS")
        print(sep)
        print(df[show_cols].to_string(index=True, max_colwidth=55))
        print(f"\n{'─' * 72}")
        print(f"  {'Metric':<32} {'Mean':>8}  {'Min':>8}  {'Max':>8}")
        print(f"{'─' * 72}")
        for col in display_cols:
            data = df[col].dropna()
            if len(data):
                print(f"  {col:<32} {data.mean():>8.4f}  {data.min():>8.4f}  {data.max():>8.4f}")
        print(f"{sep}\n")

    def _export_csv(self, df: pd.DataFrame) -> None:
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"Results saved to: {out.resolve()}\n")

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self, skip_indexing: bool = False) -> pd.DataFrame:
        """
        Execute the full evaluation pipeline.

        Parameters
        ----------
        skip_indexing : Reuse the vector store from a previous run.
        """
        # Step 1 – load documents
        print("\nStep 1/5  Loading documents from folder ...")
        print(f"  Folder : {Path(self.docs_dir).resolve()}")
        llamaindex_docs, langchain_docs = load_documents_from_folder(self.docs_dir)
        print(f"  ✓ {len(llamaindex_docs)} page/chunk(s) loaded.\n")

        # Step 2 – generate test dataset
        print("Step 2/5  Generating test dataset with Ragas TestsetGenerator ...")
        print(f"  generator LLM    : {_OPENROUTER_MODEL}  (T=0.4)")
        print(f"  transforms LLM   : {_OPENROUTER_MODEL}  (T=0.0, critic role)")
        print(f"  Target size      : {self.testset_size} questions\n")

        test_data = generate_testset(langchain_docs=langchain_docs, testset_size=self.testset_size)
        if not test_data:
            raise RuntimeError(
                "TestsetGenerator returned no samples. "
                "Check the API key and that the documents have sufficient text content."
            )
        print(f"  ✓ {len(test_data)} question(s) generated.\n")

        # Step 3 – initialise RAG + index
        print("Step 3/5  Initialising RAG pipeline ...")
        self._init_rag_system()
        if not skip_indexing:
            self._index_documents(llamaindex_docs)
        else:
            print("  (Skipping indexing – reusing existing vector store)\n")

        # Step 4 – answer all questions
        print(f"Step 4/5  Running RAG pipeline on {len(test_data)} question(s) ...")
        samples = self._collect_samples(test_data)
        if not samples:
            raise RuntimeError("No RAG samples collected.")

        # Step 5 – evaluate
        print(f"\nStep 5/5  Evaluating {len(samples)} sample(s) with Ragas 0.2.x ...\n")
        results_df = _evaluate_with_ragas(samples)
        self._print_results(results_df)
        self._export_csv(results_df)
        return results_df
