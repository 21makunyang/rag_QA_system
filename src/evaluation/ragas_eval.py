"""
Ragas-based evaluation module for the RAG QA system.

Metrics evaluated
-----------------
- Faithfulness       : Is the answer grounded in the retrieved context?
- Answer Relevancy   : Does the answer address the user's question?
- Context Precision  : Are the retrieved chunks relevant to the question?
- Context Recall     : Does the retrieved context cover the reference answer?

Model configuration
-------------------
- RAG generation LLM : OpenRouter – NVIDIA Nemotron-3 Super 120B (default)
                       OR any existing Ollama/HuggingFace backend.
- Ragas judge LLM    : OpenRouter – NVIDIA Nemotron-3 Super 120B
- Ragas embedding    : sentence-transformers/all-MiniLM-L6-v2
                       (same model used by the project's Retriever)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter LLM backend (compatible with existing OllamaBackend interface)
# ─────────────────────────────────────────────────────────────────────────────

_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLMBackend:
    """
    Thin wrapper around the OpenRouter API that implements the same interface
    as OllamaBackend / HuggingFaceBackend so it can be dropped into the
    existing ResponseGenerator.
    """

    def __init__(
        self,
        model_name: str = _OPENROUTER_MODEL,
        api_key: Optional[str] = None,
        base_url: str = _OPENROUTER_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenRouterLLMBackend. "
                "Install it with:  pip install openai"
            ) from exc

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    # Public interface (mirrors OllamaBackend)
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

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
        return {
            "model_name": self.model_name,
            "backend": "openrouter",
            "base_url": self.base_url,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Built-in test dataset (NLP / RAG domain, self-contained)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TEST_DATA: List[Dict[str, str]] = [
    {
        "user_input": "What is Retrieval-Augmented Generation (RAG)?",
        "reference": (
            "Retrieval-Augmented Generation (RAG) is an AI framework that combines "
            "information retrieval with language model generation. It first retrieves "
            "relevant documents from a knowledge base and then uses those documents as "
            "context when the language model generates an answer, improving factual "
            "accuracy and reducing hallucinations."
        ),
    },
    {
        "user_input": "How does vector similarity search work in a RAG system?",
        "reference": (
            "Vector similarity search converts text into dense numerical vectors called "
            "embeddings using neural networks. When a query arrives, it is embedded into "
            "the same vector space, and the retriever finds the documents whose vectors "
            "are closest to the query vector using distance metrics such as cosine "
            "similarity or dot product. The top-k most similar documents are returned "
            "as context for the language model."
        ),
    },
    {
        "user_input": "What are common document chunking strategies and why is overlap used?",
        "reference": (
            "Common document chunking strategies include fixed-size chunking (splitting by "
            "a fixed number of tokens or characters), sentence-based chunking (splitting at "
            "sentence boundaries to preserve grammatical units), and semantic chunking "
            "(splitting at topic boundaries detected by embedding similarity). Overlap "
            "between adjacent chunks is used to prevent important context from being lost "
            "at chunk boundaries, ensuring that retrieval does not miss relevant information "
            "that spans two consecutive chunks."
        ),
    },
    {
        "user_input": "What is ChromaDB and how is it used in RAG pipelines?",
        "reference": (
            "ChromaDB is an open-source, embeddable vector database designed for storing "
            "and querying high-dimensional embeddings. In RAG pipelines it persists document "
            "embeddings on disk and provides fast approximate nearest-neighbour search. "
            "Documents are inserted with their embeddings; at query time the database "
            "returns the most similar stored vectors, enabling the retriever to supply "
            "relevant context to the language model."
        ),
    },
    {
        "user_input": "What role do embedding models play in a RAG system?",
        "reference": (
            "Embedding models in a RAG system transform both documents and user queries "
            "into dense vector representations that capture semantic meaning. Because "
            "semantically similar texts map to nearby points in the embedding space, the "
            "retriever can find documents that are conceptually relevant to a query even "
            "when the exact words differ. The quality and dimensionality of the embedding "
            "model directly influence retrieval accuracy."
        ),
    },
    {
        "user_input": "What is faithfulness in the context of RAG evaluation?",
        "reference": (
            "Faithfulness measures whether every factual claim made in the generated answer "
            "can be directly supported by the retrieved context. A fully faithful answer "
            "contains no information that contradicts or goes beyond what the retrieved "
            "documents state, preventing hallucination. It is scored between 0 and 1, "
            "where 1 means all claims are grounded in the context."
        ),
    },
    {
        "user_input": "How does context precision differ from context recall in RAG evaluation?",
        "reference": (
            "Context precision measures the proportion of retrieved chunks that are "
            "actually relevant to the question (signal-to-noise ratio of retrieval). "
            "Context recall measures the proportion of information needed to answer the "
            "question that is present in the retrieved chunks. High precision means few "
            "irrelevant chunks were retrieved; high recall means the important facts were "
            "not missed. Both metrics together characterise retrieval quality."
        ),
    },
    {
        "user_input": "What is answer relevancy in RAG evaluation?",
        "reference": (
            "Answer relevancy measures how directly and completely the generated answer "
            "addresses the user's question, regardless of factual correctness. It is "
            "evaluated by checking whether the answer covers the intent of the question "
            "without including excessive off-topic content. The metric is typically scored "
            "between 0 and 1 using embedding similarity between the answer and questions "
            "re-generated from that answer."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator class
# ─────────────────────────────────────────────────────────────────────────────


class RagasEvaluator:
    """
    End-to-end Ragas evaluation for the RAG QA system.

    Workflow
    --------
    1. Initialise the existing RAG pipeline (Retriever + ResponseGenerator).
    2. Optionally index test reference texts as evaluation documents.
    3. Run each test question through the RAG pipeline to collect
       (user_input, retrieved_contexts, response, reference) tuples.
    4. Evaluate with Ragas metrics using OpenRouter as the judge LLM and
       the project's HuggingFace embedding model.
    5. Print results to console and export to CSV.
    """

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        rag_model: str = "openrouter",
        output_path: str = "./ragas_results.csv",
        vector_store_dir: str = "./data/eval/ragas_eval_store",
        collection_name: str = "ragas_eval_collection",
        top_k: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        openrouter_api_key : API key for OpenRouter.  Falls back to
                             the ``OPENROUTER_API_KEY`` environment variable.
        rag_model          : Which backend to use for RAG answer generation.
                             ``"openrouter"`` (default) – NVIDIA Nemotron via
                             OpenRouter.  ``"mistral-7b"`` / ``"llama2-7b"`` –
                             local Ollama.  ``"t5-base"`` – local HuggingFace.
        output_path        : Where to save the CSV results.
        vector_store_dir   : Persistent directory for the evaluation
                             ChromaDB collection (isolated from the main store).
        collection_name    : ChromaDB collection name used for evaluation.
        top_k              : Documents to retrieve per query.
        """
        self.api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.rag_model_name = rag_model
        self.output_path = output_path
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.top_k = top_k

        self._retriever = None
        self._response_gen = None

    # ── RAG system initialisation ────────────────────────────────────────────

    def _build_llm_backend(self):
        """Return a LLM backend appropriate for the configured rag_model."""
        if self.rag_model_name == "openrouter":
            logger.info("Using OpenRouter (NVIDIA Nemotron) as the RAG generation LLM.")
            return OpenRouterLLMBackend(api_key=self.api_key)

        # Delegate to existing model configs for Ollama / HuggingFace
        from src.config import Config  # noqa: PLC0415

        model_config = Config.get_model_config(self.rag_model_name)
        if model_config.backend == "ollama":
            from src.models.ollama_backend import OllamaBackend  # noqa: PLC0415

            return OllamaBackend(model_config)
        else:
            from src.models.huggingface_backend import HuggingFaceBackend  # noqa: PLC0415

            return HuggingFaceBackend(model_config)

    def _init_rag_system(self) -> None:
        """Initialise Retriever and ResponseGenerator with an isolated collection."""
        from src.config import VectorStoreConfig  # noqa: PLC0415
        from src.query.response_gen import ResponseGenerator  # noqa: PLC0415
        from src.query.retriever import Retriever  # noqa: PLC0415

        eval_vs_config = VectorStoreConfig(
            store_type="chroma",
            persist_dir=self.vector_store_dir,
            collection_name=self.collection_name,
        )

        logger.info("Initialising evaluation RAG system ...")
        self._retriever = Retriever(eval_vs_config)
        llm_backend = self._build_llm_backend()
        self._response_gen = ResponseGenerator(llm_backend, self._retriever)
        logger.info("Evaluation RAG system ready.")

    def _index_documents(self, test_data: List[Dict[str, str]]) -> None:
        """Index each reference answer as a searchable document."""
        from llama_index.core import Document  # noqa: PLC0415

        logger.info("Indexing %d evaluation documents …", len(test_data))
        docs = [
            Document(
                text=item["reference"],
                metadata={"source": "ragas_eval", "question": item["user_input"]},
            )
            for item in test_data
        ]
        self._retriever.index_documents(docs)
        logger.info("Indexing complete.")

    # ── Sample collection ────────────────────────────────────────────────────

    def _collect_samples(
        self, test_data: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Run every test question through the RAG pipeline and return a list of
        dicts with keys: user_input, retrieved_contexts, response, reference.
        """
        samples: List[Dict[str, Any]] = []
        total = len(test_data)

        for idx, item in enumerate(test_data, start=1):
            question = item["user_input"]
            reference = item["reference"]
            logger.info("  [%d/%d] %s", idx, total, question[:70])

            try:
                result = self._response_gen.generate_response(
                    query=question,
                    top_k=self.top_k,
                    use_rag=True,
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
                logger.info(
                    "    → %d context(s) retrieved, answer: %d chars",
                    len(contexts),
                    len(answer),
                )
            except Exception as exc:
                logger.error("    ✗ Failed: %s", exc)

        return samples

    # ── Ragas model configuration ────────────────────────────────────────────

    def _ragas_llm(self):
        """Wrap the OpenRouter ChatOpenAI model for use by Ragas metrics."""
        try:
            from langchain_openai import ChatOpenAI  # noqa: PLC0415
            from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-openai and ragas are required. "
                "Run: pip install langchain-openai ragas"
            ) from exc

        chat_model = ChatOpenAI(
            model=_OPENROUTER_MODEL,
            api_key=self.api_key,
            base_url=_OPENROUTER_BASE_URL,
            # Use low temperature for deterministic evaluation judgements
            temperature=0.0,
        )
        return LangchainLLMWrapper(chat_model)

    def _ragas_embeddings(self):
        """
        Wrap the project's HuggingFace embedding model for use by Ragas.
        Uses the same model as the project's Retriever
        (sentence-transformers/all-MiniLM-L6-v2).
        """
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
            from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-huggingface and ragas are required. "
                "Run: pip install langchain-huggingface ragas"
            ) from exc

        from src.config import Config  # noqa: PLC0415

        hf_emb = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        return LangchainEmbeddingsWrapper(hf_emb)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate_with_ragas(
        self, samples: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Build a Ragas EvaluationDataset and run all four metrics."""
        try:
            from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: PLC0415
            from ragas.metrics import (  # noqa: PLC0415
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
        except ImportError as exc:
            raise ImportError(
                "ragas>=0.2.0 is required. Run: pip install 'ragas>=0.2.0'"
            ) from exc

        ragas_llm = self._ragas_llm()
        ragas_emb = self._ragas_embeddings()

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
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ]

        logger.info(
            "Running Ragas evaluation on %d samples with 4 metrics …", len(samples)
        )
        results = evaluate(dataset=dataset, metrics=metrics)
        return results.to_pandas()

    # ── Output ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_results(df: pd.DataFrame) -> None:
        """Pretty-print the per-sample scores and aggregate means."""
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        display_cols = [c for c in metric_cols if c in df.columns]

        sep = "=" * 68
        print(f"\n{sep}")
        print("  RAGAS EVALUATION RESULTS")
        print(sep)

        # Per-sample table
        print(df[["user_input"] + display_cols].to_string(index=True, max_colwidth=50))

        # Aggregate row
        print(f"\n{'─' * 68}")
        print(f"  {'Metric':<30} {'Mean':>8}  {'Min':>8}  {'Max':>8}")
        print(f"{'─' * 68}")
        for col in display_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                print(
                    f"  {col:<30} {col_data.mean():>8.4f}  "
                    f"{col_data.min():>8.4f}  {col_data.max():>8.4f}"
                )
        print(f"{sep}\n")

    def _export_csv(self, df: pd.DataFrame) -> None:
        """Save results DataFrame to CSV."""
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"Results saved to: {out.resolve()}\n")
        logger.info("Results saved to %s", out.resolve())

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        test_data: Optional[List[Dict[str, str]]] = None,
        skip_indexing: bool = False,
    ) -> pd.DataFrame:
        """
        Execute the full evaluation pipeline.

        Parameters
        ----------
        test_data       : List of ``{"user_input": ..., "reference": ...}`` dicts.
                          Defaults to the built-in NLP/RAG domain dataset.
        skip_indexing   : Pass ``True`` to reuse an already-populated vector
                          store from a previous run (speeds up re-runs).

        Returns
        -------
        pandas.DataFrame with one row per test sample and columns for each
        Ragas metric score.
        """
        if test_data is None:
            test_data = DEFAULT_TEST_DATA

        # 1. Boot up the RAG pipeline
        self._init_rag_system()

        # 2. Populate the vector store (unless caller asks to skip)
        if not skip_indexing:
            self._index_documents(test_data)
        else:
            logger.info("Skipping document indexing (--skip-indexing flag set).")

        # 3. Gather RAG outputs for every test question
        logger.info("Collecting RAG outputs for %d questions …", len(test_data))
        samples = self._collect_samples(test_data)

        if not samples:
            raise RuntimeError(
                "No samples were collected.  "
                "Check that the RAG pipeline is correctly initialised and that "
                "documents have been indexed."
            )

        # 4. Run Ragas evaluation
        results_df = self._evaluate_with_ragas(samples)

        # 5. Display and persist
        self._print_results(results_df)
        self._export_csv(results_df)

        return results_df
