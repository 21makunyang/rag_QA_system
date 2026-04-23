import pytest
from src.config import Config
from src.ingestion.chunking import ChunkingFactory
from src.ingestion.connectors import PDFConnector, TextFileConnector
from src.query.retriever import Retriever
from src.query.response_gen import ResponseGenerator
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.test_cases import TestCaseManager
import numpy as np

def load_test_documents():
    """加载测试文档（使用项目中已有的 data/documents 目录）"""
    docs = []
    pdf_connector = PDFConnector()
    text_connector = TextFileConnector()
    for file_path in Config.DOCUMENTS_DIR.glob("*"):
        if pdf_connector.supports(str(file_path)):
            docs.extend(pdf_connector.load(str(file_path)))
        elif text_connector.supports(str(file_path)):
            docs.extend(text_connector.load(str(file_path)))
    return docs

def evaluate_strategy(strategy_name, chunking_config, test_queries, retriever_config):
    """评估特定分块策略的性能"""
    chunking_config.strategy = strategy_name
    chunker = ChunkingFactory.create_strategy(chunking_config)
    docs = load_test_documents()
    chunked_docs = chunker.chunk_documents(docs)

    retriever = Retriever(retriever_config)
    retriever.clear_index()
    retriever.index_documents(chunked_docs)

    from src.models.ollama_backend import OllamaBackend
    llm = OllamaBackend(Config.get_model_config("mistral-7b"))
    response_gen = ResponseGenerator(llm, retriever)

    calculator = MetricsCalculator()
    results = []
    for q in test_queries:
        resp = response_gen.generate_response(q["query"], top_k=5)
        metrics = calculator.calculate_response_metrics(
            query=q["query"],
            response=resp["answer"],
            expected_answer=q.get("expected_answer"),
            retrieved_docs=resp["retrieved_docs"]
        )
        results.append(metrics)
    return {
        "avg_semantic_similarity": np.mean([r.get("semantic_similarity", 0) for r in results]),
        "avg_context_coverage": np.mean([r.get("context_coverage_ratio", 0) for r in results]),
        "avg_retrieval_score": np.mean([r.get("avg_retrieval_score", 0) for r in results])
    }

def test_hierarchical_vs_fixed():
    strategies = ["fixed", "sentence", "semantic", "hierarchical"]
    # 从 TestCaseManager 加载测试用例
    tc_manager = TestCaseManager()
    tc_manager.generate_test_suite()
    test_queries = tc_manager.get_test_cases(max_cases=20)

    results = {}
    for s in strategies:
        print(f"Testing {s}...")
        Config.CHUNKING.strategy = s
        # 如果是层级分块，确保 chunk_sizes 已配置
        if s == "hierarchical":
            Config.CHUNKING.chunk_sizes = [2048, 512, 128]
        metrics = evaluate_strategy(s, Config.CHUNKING, test_queries, Config.VECTOR_STORE)
        results[s] = metrics

    # 打印对比结果
    for s, m in results.items():
        print(f"{s}: SS={m['avg_semantic_similarity']:.3f}, Cov={m['avg_context_coverage']:.3f}")
    return results