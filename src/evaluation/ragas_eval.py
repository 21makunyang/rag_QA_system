"""
Ragas-based evaluation module for the RAG QA system.
Requires: ragas>=0.2.0

Pipeline
--------
1. Load PDF / text files from a user-specified folder.
   PDFs are split into per-page chunks so Ragas can generate
   diverse questions across the whole document.
2. Ragas TestsetGenerator automatically builds a labelled test dataset.
   - generator LLM  (mistral:7b, T=0.4) : generates diverse questions.
   - transforms LLM (llama2:7b, T=0.0)  : critic role, filters / refines questions.
3. Documents are chunked using the **same strategy as the main RAG system**
   (FixedSizeChunking or SentenceBasedChunking via ChunkingFactory) and
   then indexed into an isolated ChromaDB collection.
4. Each generated question is answered by the RAG pipeline
   (Retriever → ResponseGenerator).
5. Ragas evaluates all samples with four metrics:
   Faithfulness · Answer Relevancy · Context Precision · Context Recall
6. Results are printed to the console and exported to CSV.

LLM config
----------
  generator LLM  : mistral:7b   via local Ollama  (T=0.4)
  critic LLM     : llama2:7b    via local Ollama  (T=0.0)
  RAG answer LLM : mistral:7b   via local Ollama
  Ragas judge LLM: mistral:7b   via local Ollama  (T=0.0)
  Embedding      : sentence-transformers/all-MiniLM-L6-v2  (local HuggingFace)

Chunking
--------
  The RAG indexing step applies the same ChunkingFactory strategy that the
  main application uses (default: FixedSizeChunking, chunk_size=256,
  overlap=25).  Pass ``chunking_strategy="sentence"`` to switch to
  SentenceBasedChunking, or ``None`` to inherit from Config.CHUNKING.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Local Ollama model configuration
# ─────────────────────────────────────────────────────────────────────────────

# Ollama server URL (override via OLLAMA_API_BASE env var if needed)
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

# Generator LLM: produces diverse candidate questions (higher temperature)
_OLLAMA_GENERATOR_MODEL = "mistral:7b"

# Critic / transforms LLM: filters & refines questions (zero temperature)
_OLLAMA_CRITIC_MODEL = "llama2:7b"

# Judge LLM used by Ragas metrics (zero temperature for consistent scoring)
_OLLAMA_JUDGE_MODEL = "mistral:7b"

# Ragas execution controls (tuned for local Ollama stability)
_RAGAS_TIMEOUT = int(os.environ.get("RAGAS_TIMEOUT", "240"))
_RAGAS_MAX_RETRIES = int(os.environ.get("RAGAS_MAX_RETRIES", "6"))
_RAGAS_MAX_WAIT = int(os.environ.get("RAGAS_MAX_WAIT", "120"))
_RAGAS_MAX_WORKERS = int(os.environ.get("RAGAS_MAX_WORKERS", "1"))
_RAGAS_DOCS_PER_QUESTION = int(os.environ.get("RAGAS_DOCS_PER_QUESTION", "5"))
_RAGAS_MAX_TESTSET_DOCS = int(os.environ.get("RAGAS_MAX_TESTSET_DOCS", "24"))
_RAGAS_PARSE_MIN_DOCS = int(os.environ.get("RAGAS_PARSE_MIN_DOCS", "1"))
_RAGAS_TESTSET_OVERSAMPLE_FACTOR = int(
    os.environ.get("RAGAS_TESTSET_OVERSAMPLE_FACTOR", "2")
)
_RAGAS_MAX_GENERATED_QUESTIONS = int(
    os.environ.get("RAGAS_MAX_GENERATED_QUESTIONS", "48")
)

# Default folder where the user drops evaluation PDF / text files
_DEFAULT_DOCS_DIR = "./data/eval/ragas_docs"
_DEFAULT_RESULTS_DIR = "./data/eval/ragas_eval_results"

# Supported file extensions
_SUPPORTED_EXTS = {".pdf", ".txt", ".md"}
_METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def export_testset_to_csv(test_data: List[Dict[str, Any]], output_path: str, document_count: int = 0) -> None:
    """
    Export test set data to CSV format.

    Parameters
    ----------
    test_data : List of test samples with user_input, retrieved_contexts, response, and reference
    output_path : Path to save the CSV file
    document_count : Number of source documents (for metadata)
    """
    import csv
    from datetime import datetime

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write metadata header
        writer.writerow(["# Generated at:", datetime.now().isoformat()])
        writer.writerow(["# Document count:", document_count])
        writer.writerow(["# Question count:", len(test_data)])
        writer.writerow([])  # Empty row for readability

        # Write data header
        writer.writerow(["user_input", "retrieved_contexts", "response", "reference"])

        # Write test data
        for item in test_data:
            writer.writerow([
                item.get("user_input", ""),
                item.get("retrieved_contexts", ""),
                item.get("response", ""),
                item.get("reference", "")
            ])

    logger.info(f"Test set exported to {output_path} ({len(test_data)} questions)")


def import_testset_from_csv(input_path: str) -> List[Dict[str, str]]:
    """
    Import test set data from CSV format.

    Parameters
    ----------
    input_path : Path to the CSV file to import

    Returns
    -------
    List of test samples with user_input and reference
    """
    import csv

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Test set file not found: {input_path}")

    test_data = []
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)

        # Skip metadata headers (lines starting with #)
        for row in reader:
            if not row or (row[0].startswith('#') if row else False):
                continue
            # First non-comment row should be header
            if row[0] == "user_input" and row[1] == "reference":
                break

        # Read test data
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                test_data.append({
                    "user_input": row[0],
                    "reference": row[1]
                })

    if not test_data:
        raise ValueError(f"No valid test data found in {input_path}")

    logger.info(f"Test set imported from {input_path} ({len(test_data)} questions)")
    return test_data


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


def _extract_source_file_names(
    llamaindex_docs: list, suffixes: tuple[str, ...] | None = None
) -> List[str]:
    """Collect unique source file names from loaded docs."""
    names = {
        str(getattr(doc, "metadata", {}).get("file_name", ""))
        for doc in llamaindex_docs
        if str(getattr(doc, "metadata", {}).get("file_name", ""))
    }
    if suffixes is not None:
        suffixes_lower = tuple(s.lower() for s in suffixes)
        names = {name for name in names if name.lower().endswith(suffixes_lower)}
    return sorted(names)


def _estimate_token_count(text: str) -> int:
    """
    Rough token estimate for mixed EN/CN text.

    - For normal space-separated text, use word count.
    - For dense no-space text (common in extracted slides/PDF blocks),
      fallback to character-based estimation.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0

    words = re.findall(r"\S+", stripped)
    if len(words) <= 1:
        # Character-based fallback (very rough but robust)
        return max(1, len(stripped) // 4)
    return len(words)


def _prepare_langchain_docs_for_testset(
    langchain_docs: list,
    min_tokens: int = 120,
    target_tokens: int = 260,
) -> list:
    """
    Merge very short LangChain docs before feeding them into Ragas.

    Why this is needed:
    Ragas default transforms may reject inputs when document chunks are too short
    (around <=100 tokens). PDF page-level extraction can produce many such chunks
    (e.g., title pages, figure-only pages, slide bullets), even if total pages are large.
    """
    if not langchain_docs:
        return []

    try:
        from langchain_core.documents import Document as LCDocument  # noqa: PLC0415
    except ImportError:
        from langchain.schema import Document as LCDocument  # type: ignore[no-redef]  # noqa: PLC0415

    original_token_counts = [
        _estimate_token_count(getattr(doc, "page_content", "") or "")
        for doc in langchain_docs
    ]
    if original_token_counts:
        logger.info(
            "Ragas input docs before merge: n=%d, avg_tokens=%.1f, min=%d, max=%d",
            len(original_token_counts),
            sum(original_token_counts) / len(original_token_counts),
            min(original_token_counts),
            max(original_token_counts),
        )

    grouped: Dict[str, list] = {}
    group_order: List[str] = []
    for doc in langchain_docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source = str(metadata.get("source") or metadata.get("file_name") or "unknown")
        if source not in grouped:
            grouped[source] = []
            group_order.append(source)
        grouped[source].append(doc)

    merged_docs: list = []
    for source in group_order:
        docs = grouped[source]
        docs = sorted(docs, key=lambda d: int((getattr(d, "metadata", {}) or {}).get("page", 10**9)))

        buffer_texts: List[str] = []
        buffer_tokens = 0
        buffer_start_page = None
        buffer_end_page = None

        def flush_buffer() -> None:
            nonlocal buffer_texts, buffer_tokens, buffer_start_page, buffer_end_page
            if not buffer_texts:
                return
            joined = "\n\n".join(buffer_texts).strip()
            if not joined:
                buffer_texts, buffer_tokens = [], 0
                buffer_start_page, buffer_end_page = None, None
                return
            meta = {"source": source, "merged_for_ragas": True, "merged_parts": len(buffer_texts)}
            if buffer_start_page is not None:
                meta["page_start"] = buffer_start_page
                meta["page_end"] = buffer_end_page
            merged_docs.append(LCDocument(page_content=joined, metadata=meta))
            buffer_texts, buffer_tokens = [], 0
            buffer_start_page, buffer_end_page = None, None

        for doc in docs:
            text = (getattr(doc, "page_content", "") or "").strip()
            if not text:
                continue

            metadata = dict(getattr(doc, "metadata", {}) or {})
            page = metadata.get("page")
            tokens = _estimate_token_count(text)

            # Keep sufficiently long docs as-is for diversity
            if tokens >= min_tokens:
                flush_buffer()
                merged_docs.append(
                    LCDocument(page_content=text, metadata=metadata)
                )
                continue

            # Merge short docs together
            if buffer_start_page is None and page is not None:
                buffer_start_page = page
            if page is not None:
                buffer_end_page = page
            buffer_texts.append(text)
            buffer_tokens += tokens
            if buffer_tokens >= target_tokens:
                flush_buffer()

        flush_buffer()

    merged_token_counts = [
        _estimate_token_count(getattr(doc, "page_content", "") or "")
        for doc in merged_docs
    ]
    if merged_token_counts:
        logger.info(
            "Ragas input docs after merge:  n=%d, avg_tokens=%.1f, min=%d, max=%d",
            len(merged_token_counts),
            sum(merged_token_counts) / len(merged_token_counts),
            min(merged_token_counts),
            max(merged_token_counts),
        )
    return merged_docs if merged_docs else langchain_docs


def _limit_docs_for_testset(langchain_docs: list, testset_size: int) -> list:
    """
    Cap the number of docs sent to Ragas transforms to improve local stability.

    Ragas TestsetGenerator applies multiple LLM transforms to every input doc.
    On local Ollama, sending too many docs at once can cause repeated 502 errors.
    This function keeps representative docs across sources and spreads picks
    within each source to reduce near-duplicate prompts.
    """
    if not langchain_docs:
        return []

    target_max = max(testset_size * _RAGAS_DOCS_PER_QUESTION, testset_size)
    # target_max = min(target_max, _RAGAS_MAX_TESTSET_DOCS)
    if len(langchain_docs) <= target_max:
        return langchain_docs

    grouped: Dict[str, list] = {}
    source_order: List[str] = []
    for doc in langchain_docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source = str(metadata.get("source") or metadata.get("file_name") or "unknown")
        if source not in grouped:
            grouped[source] = []
            source_order.append(source)
        grouped[source].append(doc)

    def _even_positions(total: int, take: int) -> List[int]:
        """Return unique, sorted indices spaced across [0, total)."""
        if take <= 0 or total <= 0:
            return []
        if take >= total:
            return list(range(total))
        if take == 1:
            return [total // 2]

        step = (total - 1) / (take - 1)
        picked: List[int] = []
        seen = set()
        for i in range(take):
            idx = int(round(i * step))
            idx = min(max(idx, 0), total - 1)
            if idx not in seen:
                picked.append(idx)
                seen.add(idx)

        # Rounding can occasionally collapse to fewer points.
        cursor = 0
        while len(picked) < take and cursor < total:
            if cursor not in seen:
                picked.append(cursor)
                seen.add(cursor)
            cursor += 1
        return sorted(picked)

    # First pass: guarantee each source can contribute one spread sample.
    selected: list = []
    selected_ids = set()
    picks_per_source = {source: 0 for source in source_order}
    for source in source_order:
        if len(selected) >= target_max:
            break
        first_pos = _even_positions(len(grouped[source]), 1)
        if not first_pos:
            continue
        doc = grouped[source][first_pos[0]]
        doc_id = id(doc)
        if doc_id in selected_ids:
            continue
        selected.append(doc)
        selected_ids.add(doc_id)
        picks_per_source[source] = 1

    # Second pass: fill remaining quota by increasing per-source samples
    # while keeping sample positions spread across each source.
    while len(selected) < target_max:
        progressed = False
        for source in source_order:
            if len(selected) >= target_max:
                break
            docs = grouped[source]
            current_take = picks_per_source[source]
            if current_take >= len(docs):
                continue
            next_take = current_take + 1
            positions = _even_positions(len(docs), next_take)
            for pos in positions:
                doc = docs[pos]
                doc_id = id(doc)
                if doc_id in selected_ids:
                    continue
                selected.append(doc)
                selected_ids.add(doc_id)
                picks_per_source[source] = next_take
                progressed = True
                break
        if not progressed:
            break

    logger.info(
        "Downsampled docs for testset generation: %d -> %d (target_max=%d)",
        len(langchain_docs),
        len(selected),
        target_max,
    )
    return selected if selected else langchain_docs


# ─────────────────────────────────────────────────────────────────────────────
# Shared LangChain / Ragas model builders  (local Ollama)
# ─────────────────────────────────────────────────────────────────────────────


def _langchain_llm(temperature: float, model_name: str = _OLLAMA_GENERATOR_MODEL):
    """Return a ChatOllama instance pointing at a local Ollama model."""
    try:
        from langchain_ollama import ChatOllama  # noqa: PLC0415
    except ImportError:
        from langchain_community.chat_models import ChatOllama  # type: ignore[no-redef]  # noqa: PLC0415

    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=_OLLAMA_BASE_URL,
    )


def _langchain_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
    from src.config import Config  # noqa: PLC0415

    return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)


def _ragas_llm(temperature: float, model_name: str = _OLLAMA_GENERATOR_MODEL):
    """Return a Ragas-wrapped ChatOllama instance."""
    from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415

    return LangchainLLMWrapper(_langchain_llm(temperature, model_name))


def _ragas_embeddings():
    """Return a Ragas-wrapped HuggingFace embedding model."""
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415

    return LangchainEmbeddingsWrapper(_langchain_embeddings())


def _ragas_run_config(max_workers: int | None = None):
    """Build a conservative RunConfig for local Ollama."""
    from ragas.run_config import RunConfig  # noqa: PLC0415

    workers = max_workers if max_workers is not None else _RAGAS_MAX_WORKERS
    workers = max(1, int(workers))
    return RunConfig(
        timeout=_RAGAS_TIMEOUT,
        max_retries=_RAGAS_MAX_RETRIES,
        max_wait=_RAGAS_MAX_WAIT,
        max_workers=workers,
    )


def _is_ollama_502_error(exc: Exception) -> bool:
    """Detect transient Ollama gateway failures from exception text."""
    message = str(exc).lower()
    return "status code: 502" in message or "502 bad gateway" in message


def _is_ragas_parse_error(exc: Exception) -> bool:
    """Detect output parsing failures raised by Ragas/LangChain parsers."""
    text = str(exc).lower()
    name = exc.__class__.__name__.lower()
    markers = (
        "ragasoutputparserexception",
        "outputparserexception",
        "output parsing failure",
        "failed to parse",
        "the output parser failed to parse",
    )
    return any(m in text for m in markers) or any(m in name for m in markers)


def _is_headline_property_missing_error(exc: Exception) -> bool:
    """Detect a known Ragas transform failure in HeadlineSplitter."""
    text = str(exc).lower()
    return "headlines' property not found" in text or "\"headlines\" property not found" in text


def _build_safe_transforms(llm, embedding_model):
    """
    Build a conservative transform pipeline that avoids HeadlineSplitter.

    Reason:
    Some ragas versions can fail on mixed-length corpora when
    HeadlinesExtractor only annotates long docs but HeadlineSplitter still runs
    on nodes without the ``headlines`` property.
    """
    from ragas.testset.graph import NodeType  # noqa: PLC0415
    from ragas.testset.transforms.default import (  # noqa: PLC0415
        CosineSimilarityBuilder,
        EmbeddingExtractor,
        NERExtractor,
        OverlapScoreBuilder,
        Parallel,
        SummaryExtractor,
        ThemesExtractor,
    )

    def _filter_doc(node, min_tokens: int = 100):
        from ragas.testset.transforms.default import num_tokens_from_string  # noqa: PLC0415

        return (
            node.type == NodeType.DOCUMENT
            and num_tokens_from_string(node.properties.get("page_content", "")) > min_tokens
        )

    summary_extractor = SummaryExtractor(
        llm=llm, filter_nodes=lambda node: _filter_doc(node, 100)
    )
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=lambda node: _filter_doc(node, 100),
    )
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.5,
        filter_nodes=lambda node: _filter_doc(node, 100),
    )
    ner_extractor = NERExtractor(llm=llm)
    ner_overlap_sim = OverlapScoreBuilder(threshold=0.01)
    theme_extractor = ThemesExtractor(
        llm=llm, filter_nodes=lambda node: node.type == NodeType.DOCUMENT
    )
    return [
        summary_extractor,
        # NOTE: CustomNodeFilter is intentionally excluded in the safe profile.
        # It is fragile with local small models because its scoring prompt
        # often returns schema-incompatible JSON and causes parser failures.
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]


def _cap_docs_for_retry(langchain_docs: list, max_docs: int) -> list:
    """Cap docs count while keeping source diversity with round-robin."""
    if not langchain_docs:
        return []
    max_docs = max(1, int(max_docs))
    if len(langchain_docs) <= max_docs:
        return langchain_docs

    grouped: Dict[str, list] = {}
    source_order: List[str] = []
    for doc in langchain_docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source = str(metadata.get("source") or metadata.get("file_name") or "unknown")
        if source not in grouped:
            grouped[source] = []
            source_order.append(source)
        grouped[source].append(doc)

    selected: list = []
    idx_by_source = {src: 0 for src in source_order}
    while len(selected) < max_docs:
        progressed = False
        for src in source_order:
            idx = idx_by_source[src]
            if idx < len(grouped[src]) and len(selected) < max_docs:
                selected.append(grouped[src][idx])
                idx_by_source[src] += 1
                progressed = True
        if not progressed:
            break
    return selected if selected else langchain_docs[:max_docs]


def _generate_testset_once(
    generator,
    docs: list,
    testset_size: int,
    critic_llm,
    emb,
    run_config,
    transforms=None,
):
    """
    One call to TestsetGenerator with compatibility fallback for older 0.2.x.
    """
    try:
        return generator.generate_with_langchain_docs(
            docs,
            testset_size=testset_size,
            transforms=transforms,
            transforms_llm=critic_llm,
            transforms_embedding_model=emb,
            run_config=run_config,
        )
    except TypeError:
        # Older 0.2.x builds may not support transforms_llm params
        logger.info("transforms_llm not supported in this build; using single LLM.")
        return generator.generate_with_langchain_docs(
            docs,
            testset_size=testset_size,
            transforms=transforms,
            run_config=run_config,
        )


def _normalize_question_for_dedup(question: str) -> str:
    """Normalize generated question text for near-duplicate filtering."""
    normalized = re.sub(r"[\W_]+", " ", question.lower(), flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _deduplicate_test_samples(
    samples: List[Dict[str, str]], target_size: int
) -> List[Dict[str, str]]:
    """Drop duplicate questions and keep at most *target_size* unique samples."""
    unique_samples: List[Dict[str, str]] = []
    seen_questions = set()
    duplicate_count = 0

    for sample in samples:
        question = str(sample.get("user_input", "")).strip()
        reference = str(sample.get("reference", "")).strip()
        if not question or not reference:
            continue

        dedup_key = _normalize_question_for_dedup(question)
        if not dedup_key or dedup_key in seen_questions:
            duplicate_count += 1
            continue

        seen_questions.add(dedup_key)
        unique_samples.append({"user_input": question, "reference": reference})

    if duplicate_count:
        logger.info(
            "Removed %d duplicate generated question(s): %d -> %d",
            duplicate_count,
            len(samples),
            len(unique_samples),
        )

    if target_size > 0:
        return unique_samples[:target_size]
    return unique_samples


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

    # Avoid "documents too short" failures in Ragas transforms
    prepared_docs = _prepare_langchain_docs_for_testset(
        langchain_docs=langchain_docs, min_tokens=120, target_tokens=260
    )
    prepared_docs = _limit_docs_for_testset(prepared_docs, testset_size=testset_size)
    logger.info(
        "Prepared docs for TestsetGenerator: %d -> %d",
        len(langchain_docs),
        len(prepared_docs),
    )

    # mistral:7b generates diverse questions
    generator_llm = _ragas_llm(temperature=0.4, model_name=_OLLAMA_GENERATOR_MODEL)
    emb = _ragas_embeddings()

    generator = TestsetGenerator(llm=generator_llm, embedding_model=emb)

    # transforms_llm acts as the critic.
    # Retry plan adapts model, concurrency, and docs count for local stability.
    critic_plan = []
    for model_name in (_OLLAMA_CRITIC_MODEL, _OLLAMA_GENERATOR_MODEL):
        if model_name not in critic_plan:
            critic_plan.append(model_name)

    workers_plan = []
    for workers in (_RAGAS_MAX_WORKERS, 1):
        workers = max(1, workers)
        if workers not in workers_plan:
            workers_plan.append(workers)

    attempt_plan: List[Tuple[str, int, list, str, str]] = []
    for critic_model in critic_plan:
        for workers in workers_plan:
            plan = (critic_model, workers, prepared_docs, "base", "default")
            if plan not in attempt_plan:
                attempt_plan.append(plan)

    parse_doc_caps = [
        max(1, testset_size * 2),
        max(_RAGAS_PARSE_MIN_DOCS, testset_size),
        _RAGAS_PARSE_MIN_DOCS,
    ]
    for cap in parse_doc_caps:
        reduced_docs = _cap_docs_for_retry(prepared_docs, max_docs=cap)
        if len(reduced_docs) >= len(prepared_docs):
            continue
        for critic_model in (_OLLAMA_GENERATOR_MODEL, _OLLAMA_CRITIC_MODEL):
            plan = (
                critic_model,
                1,
                reduced_docs,
                f"parse_fallback_cap_{len(reduced_docs)}",
                "safe",
            )
            if plan not in attempt_plan:
                attempt_plan.append(plan)

    requested_size = max(1, int(testset_size))
    oversample_factor = max(1, int(_RAGAS_TESTSET_OVERSAMPLE_FACTOR))
    generated_size = min(
        requested_size * oversample_factor,
        max(requested_size, int(_RAGAS_MAX_GENERATED_QUESTIONS)),
    )
    logger.info(
        "Testset generation target=%d, generation_size=%d (oversample_factor=%d, max_generated=%d)",
        requested_size,
        generated_size,
        oversample_factor,
        _RAGAS_MAX_GENERATED_QUESTIONS,
    )

    testset = None
    last_err: Exception | None = None
    for attempt_idx, (critic_model, workers, attempt_docs, reason, transform_mode) in enumerate(attempt_plan, start=1):
        critic_llm = _ragas_llm(temperature=0.0, model_name=critic_model)
        run_config = _ragas_run_config(max_workers=workers)
        transforms = (
            _build_safe_transforms(critic_llm, emb)
            if transform_mode == "safe"
            else None
        )
        logger.info(
            "TestsetGenerator attempt %d/%d reason=%s transform=%s docs=%d critic=%s run_config(timeout=%d, max_retries=%d, max_wait=%d, max_workers=%d)",
            attempt_idx,
            len(attempt_plan),
            reason,
            transform_mode,
            len(attempt_docs),
            critic_model,
            _RAGAS_TIMEOUT,
            _RAGAS_MAX_RETRIES,
            _RAGAS_MAX_WAIT,
            workers,
        )
        try:
            testset = _generate_testset_once(
                generator=generator,
                docs=attempt_docs,
                testset_size=generated_size,
                critic_llm=critic_llm,
                emb=emb,
                run_config=run_config,
                transforms=transforms,
            )
            break
        except ValueError as err:
            last_err = err
            # Some ragas builds still raise this if average chunk length is low.
            if "too short" in str(err).lower():
                logger.warning(
                    "Ragas rejected document length (%s). Retrying with more aggressive merge.",
                    err,
                )
                retry_docs = _prepare_langchain_docs_for_testset(
                    langchain_docs=langchain_docs, min_tokens=200, target_tokens=500
                )
                retry_docs = _limit_docs_for_testset(
                    retry_docs, testset_size=max(1, testset_size)
                )
                testset = _generate_testset_once(
                    generator=generator,
                    docs=retry_docs,
                    testset_size=generated_size,
                    critic_llm=critic_llm,
                    emb=emb,
                    run_config=run_config,
                    transforms=transforms,
                )
                break
            if _is_ragas_parse_error(err) and attempt_idx < len(attempt_plan):
                logger.warning(
                    "Ragas parser failed on attempt %d/%d (ValueError path). "
                    "Retrying with stricter fallback profile. Error: %s",
                    attempt_idx,
                    len(attempt_plan),
                    err,
                )
                time.sleep(1)
                continue
            if _is_headline_property_missing_error(err) and attempt_idx < len(attempt_plan):
                logger.warning(
                    "Ragas HeadlineSplitter failed on attempt %d/%d. "
                    "Retrying with safer transform profile. Error: %s",
                    attempt_idx,
                    len(attempt_plan),
                    err,
                )
                time.sleep(1)
                continue
            logger.warning(
                "Unexpected ValueError from testset generation: %s",
                err,
            )
            raise
        except Exception as err:
            last_err = err
            if _is_ollama_502_error(err) and attempt_idx < len(attempt_plan):
                wait_s = 5 * attempt_idx
                logger.warning(
                    "Ollama returned 502 during testset generation. "
                    "Retrying with adjusted critic/concurrency in %d second(s).",
                    wait_s,
                )
                time.sleep(wait_s)
                continue
            if _is_ragas_parse_error(err) and attempt_idx < len(attempt_plan):
                logger.warning(
                    "Ragas parser failed on attempt %d/%d. "
                    "Retrying with stricter fallback profile. Error: %s",
                    attempt_idx,
                    len(attempt_plan),
                    err,
                )
                time.sleep(1)
                continue
            raise

    if testset is None:
        # Final fallback: minimal transforms load to avoid hard failure.
        fallback_docs = _cap_docs_for_retry(prepared_docs, max_docs=max(_RAGAS_PARSE_MIN_DOCS, 1))
        logger.warning(
            "All planned attempts failed. Running final fallback with docs=%d, critic=%s, max_workers=1.",
            len(fallback_docs),
            _OLLAMA_GENERATOR_MODEL,
        )
        try:
            testset = _generate_testset_once(
                generator=generator,
                docs=fallback_docs,
                testset_size=generated_size,
                critic_llm=_ragas_llm(temperature=0.0, model_name=_OLLAMA_GENERATOR_MODEL),
                emb=emb,
                run_config=_ragas_run_config(max_workers=1),
                transforms=_build_safe_transforms(
                    _ragas_llm(temperature=0.0, model_name=_OLLAMA_GENERATOR_MODEL),
                    emb,
                ),
            )
        except Exception as final_err:  # pragma: no cover - defensive guard
            if last_err is not None:
                raise RuntimeError(
                    "Failed to generate testset after retries and final fallback. "
                    f"Last error: {last_err}"
                ) from final_err
            raise RuntimeError("Failed to generate testset after retries and final fallback.") from final_err

    df = testset.to_pandas()
    logger.info("TestsetGenerator produced %d rows; columns: %s", len(df), list(df.columns))

    samples: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        # 0.2.x uses user_input / reference; guard against missing columns
        question = row.get("user_input") or row.get("question", "")
        ground_truth = row.get("reference") or row.get("ground_truth", "")
        if pd.notna(question) and pd.notna(ground_truth) and question and ground_truth:
            samples.append({"user_input": str(question), "reference": str(ground_truth)})

    deduped_samples = _deduplicate_test_samples(samples, target_size=requested_size)
    if len(deduped_samples) < requested_size:
        logger.warning(
            "Unique generated questions are fewer than requested: %d < %d. "
            "Consider increasing RAGAS_MAX_TESTSET_DOCS / RAGAS_MAX_GENERATED_QUESTIONS.",
            len(deduped_samples),
            requested_size,
        )
    else:
        logger.info(
            "Using %d deduplicated test samples (requested=%d).",
            len(deduped_samples),
            requested_size,
        )
    return deduped_samples


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

    judge_llm = _ragas_llm(temperature=0.0, model_name=_OLLAMA_JUDGE_MODEL)
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

    logger.info(
        "Running Ragas 0.2.x evaluation on %d samples with RunConfig(timeout=%d, max_retries=%d, max_wait=%d, max_workers=%d) ...",
        len(samples),
        _RAGAS_TIMEOUT,
        _RAGAS_MAX_RETRIES,
        _RAGAS_MAX_WAIT,
        min(_RAGAS_MAX_WORKERS, 2),
    )
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=_ragas_run_config(max_workers=min(_RAGAS_MAX_WORKERS, 2)),
            raise_exceptions=False,
        )
    except Exception as err:
        if not _is_ollama_502_error(err):
            raise
        logger.warning(
            "Ollama returned 502 during evaluation. Retrying once with single worker."
        )
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=_ragas_run_config(max_workers=1),
            raise_exceptions=False,
        )
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
        rag_model: str = "mistral-7b",
        output_path: str = _DEFAULT_RESULTS_DIR,
        vector_store_dir: str = "./data/eval/ragas_eval_store",
        collection_name: str = "ragas_eval_collection",
        top_k: int = 3,
        testset_size: int = 8,
        chunking_strategy: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        docs_dir          : Folder with PDF / text evaluation files.
        rag_model         : ``"mistral-7b"`` | ``"llama2-7b"`` | ``"t5-base"``
        output_path       : Destination directory or CSV file path.
        vector_store_dir  : Isolated ChromaDB directory (separate from the
                            main application store).
        collection_name   : ChromaDB collection name for this evaluation run.
        top_k             : Documents retrieved per question.
        testset_size      : Number of test questions to auto-generate.
        chunking_strategy : ``"fixed"`` | ``"sentence"`` | ``None``.
                            ``None`` inherits from ``Config.CHUNKING.strategy``
                            (which matches the main RAG system).
        """
        self.docs_dir = docs_dir
        self.rag_model_name = rag_model
        self.output_path = output_path
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.testset_size = testset_size
        self.chunking_strategy = chunking_strategy
        self._retriever = None
        self._response_gen = None
        self._run_timestamp = datetime.now()
        self._source_pdf_files: List[str] = []
        self._sample_failures: List[Dict[str, str]] = []

    # ── RAG system ───────────────────────────────────────────────────────────

    def _build_rag_llm(self):
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

    def _apply_chunking(self, llamaindex_docs: list) -> list:
        """
        Chunk *llamaindex_docs* using the same strategy as the main RAG system.

        Strategy resolution order
        -------------------------
        1. ``self.chunking_strategy`` if explicitly set (``"fixed"`` or ``"sentence"``).
        2. ``Config.CHUNKING.strategy`` — the project-wide default.

        This ensures the evaluation vector store contains the same granularity
        of chunks as the production vector store.
        """
        from src.config import Config, ChunkingConfig  # noqa: PLC0415
        from src.ingestion.chunking import ChunkingFactory  # noqa: PLC0415

        base_cfg = Config.CHUNKING
        if self.chunking_strategy and self.chunking_strategy != base_cfg.strategy:
            # Build an override config that shares chunk_size / overlap from base
            chunking_cfg = ChunkingConfig(
                strategy=self.chunking_strategy,
                chunk_size=base_cfg.chunk_size,
                chunk_overlap=base_cfg.chunk_overlap,
            )
        else:
            chunking_cfg = base_cfg

        strategy = ChunkingFactory.create_strategy(chunking_cfg)
        logger.info(
            "Applying %s chunking (chunk_size=%d, overlap=%d) to %d document(s).",
            chunking_cfg.strategy,
            chunking_cfg.chunk_size,
            chunking_cfg.chunk_overlap,
            len(llamaindex_docs),
        )
        chunked = strategy.chunk_documents(llamaindex_docs)
        logger.info("Chunking produced %d chunks.", len(chunked))
        return chunked

    def _index_documents(self, llamaindex_docs: list) -> None:
        logger.info("Indexing %d document chunk(s) ...", len(llamaindex_docs))
        self._retriever.index_documents(llamaindex_docs)
        logger.info("Indexing complete.")

    def _clear_vector_store(self) -> None:
        """Clear all documents from the vector store before indexing."""
        if self._retriever is not None:
            logger.info("Clearing vector store ...")
            self._retriever.clear_index()
            logger.info("Vector store cleared.")

    # ── Sample collection ────────────────────────────────────────────────────

    def _collect_samples(self, test_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        self._sample_failures = []
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
                logger.info("    -> %d context(s), answer: %d chars", len(contexts), len(answer))
            except Exception as exc:
                error_text = str(exc).strip() or exc.__class__.__name__
                self._sample_failures.append(
                    {
                        "user_input": question,
                        "reference": reference,
                        "error": error_text,
                    }
                )
                logger.error("    [SKIP] %s", error_text)
        return samples

    # ── Output ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_results(df: pd.DataFrame) -> None:
        display_cols = [c for c in _METRIC_COLUMNS if c in df.columns]
        question_col = next((c for c in ("user_input", "question") if c in df.columns), None)
        show_cols = ([question_col] if question_col else []) + display_cols

        sep = "=" * 72
        print(f"\n{sep}")
        print("  RAGAS EVALUATION RESULTS")
        print(sep)
        if show_cols:
            print(df[show_cols].to_string(index=True, max_colwidth=55))
        else:
            print("  No metric columns found in evaluation output.")
        print(f"\n{'-' * 72}")
        print(f"  {'Metric':<32} {'Mean':>8}  {'Min':>8}  {'Max':>8}")
        print(f"{'-' * 72}")
        for col in display_cols:
            data = df[col].dropna()
            if len(data):
                print(f"  {col:<32} {data.mean():>8.4f}  {data.min():>8.4f}  {data.max():>8.4f}")
        print(f"{sep}\n")

    def _export_csv(self, df: pd.DataFrame) -> None:
        out = self._resolve_output_csv_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        export_df = df.copy()
        export_df["evaluation_timestamp"] = self._run_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        export_df["source_pdf_files"] = "; ".join(self._source_pdf_files) if self._source_pdf_files else ""
        export_df["source_pdf_count"] = len(self._source_pdf_files)
        export_df.to_csv(out, index=False, encoding="utf-8")
        print(f"Results saved to: {out.resolve()}\n")

    def _resolve_output_csv_path(self) -> Path:
        """
        Resolve the final CSV output path with a timestamped filename.

        Behaviour
        ---------
        - If ``output_path`` is a directory (or has no suffix), save
          ``ragas_results_YYYYMMDD_HHMMSS.csv`` under that directory.
        - If ``output_path`` is a CSV file path, append the timestamp before
          the suffix, e.g. ``ragas_results_YYYYMMDD_HHMMSS.csv``.
        """
        timestamp = self._run_timestamp.strftime("%Y%m%d_%H%M%S")
        raw_path = Path(self.output_path)

        if raw_path.suffix.lower() == ".csv":
            return raw_path.with_name(f"{raw_path.stem}_{timestamp}.csv")

        return raw_path / f"ragas_results_{timestamp}.csv"

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self, skip_indexing: bool = False, test_data: List[Dict[str, str]] | None = None, clear_vector_store: bool = False) -> pd.DataFrame:
        """
        Execute the full evaluation pipeline.

        Parameters
        ----------
        skip_indexing : Reuse the vector store from a previous run.
        test_data : Pre-generated test data (skips test set generation if provided).
        clear_vector_store : Clear the vector store before indexing.
        """
        # Step 1 - load documents
        print("\nStep 1/5  Loading documents from folder ...")
        print(f"  Folder : {Path(self.docs_dir).resolve()}")
        llamaindex_docs, langchain_docs = load_documents_from_folder(self.docs_dir)
        self._source_pdf_files = _extract_source_file_names(
            llamaindex_docs, suffixes=(".pdf",)
        )
        print(f"  [OK] {len(llamaindex_docs)} page/chunk(s) loaded.\n")
        if self._source_pdf_files:
            print(f"  Source PDFs : {', '.join(self._source_pdf_files)}\n")

        # Step 2 - generate or use provided test dataset
        if test_data is None:
            print("Step 2/5  Generating test dataset with Ragas TestsetGenerator ...")
            print(f"  generator LLM    : {_OLLAMA_GENERATOR_MODEL}  (T=0.4, via Ollama)")
            print(f"  transforms LLM   : {_OLLAMA_CRITIC_MODEL}  (T=0.0, critic role via Ollama)")
            print(f"  Target size      : {self.testset_size} questions\n")

            test_data = generate_testset(langchain_docs=langchain_docs, testset_size=self.testset_size)
            if not test_data:
                raise RuntimeError(
                    "TestsetGenerator returned no samples. "
                    "Check that Ollama is running and the documents have sufficient text content."
                )
            print(f"  [OK] {len(test_data)} question(s) generated.\n")
        else:
            print("Step 2/5  Using provided test dataset ...")
            print(f"  Questions loaded : {len(test_data)}\n")

        # Step 3 - initialise RAG + index
        print("Step 3/5  Initialising RAG pipeline ...")
        self._init_rag_system()
        if clear_vector_store:
            self._clear_vector_store()
        if not skip_indexing:
            from src.config import Config  # noqa: PLC0415
            effective_strategy = self.chunking_strategy or Config.CHUNKING.strategy
            print(f"  Chunking strategy : {effective_strategy} "
                  f"(chunk_size={Config.CHUNKING.chunk_size}, "
                  f"overlap={Config.CHUNKING.chunk_overlap})")
            chunked_docs = self._apply_chunking(llamaindex_docs)
            print(
                f"  [OK] {len(llamaindex_docs)} page(s) to {len(chunked_docs)} chunk(s) "
                f"after {effective_strategy} chunking.\n"
            )
            self._index_documents(chunked_docs)
        else:
            print("  (Skipping indexing – reusing existing vector store)\n")

        # Step 4 - answer all questions
        print(f"Step 4/5  Running RAG pipeline on {len(test_data)} question(s) ...")
        samples = self._collect_samples(test_data)
        if self._sample_failures:
            print(
                f"  [WARN] {len(self._sample_failures)} question(s) failed during "
                "RAG answer generation and were skipped."
            )
        if not samples:
            unique_errors = sorted(
                {
                    failure["error"]
                    for failure in self._sample_failures
                    if failure.get("error")
                }
            )
            error_summary = "; ".join(unique_errors[:3]) if unique_errors else "unknown error"
            raise RuntimeError(
                "No RAG samples collected. All generated questions failed during "
                f"answer generation. Error summary: {error_summary}"
            )

        # Step 5 - evaluate
        print(f"\nStep 5/5  Evaluating {len(samples)} sample(s) with Ragas 0.2.x ...\n")
        results_df = _evaluate_with_ragas(samples)
        self._print_results(results_df)
        self._export_csv(results_df)
        return results_df
