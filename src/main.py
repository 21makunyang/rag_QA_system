"""
Main entry point for CS6493 LLM Applications
"""

import argparse
import logging
import sys
from pathlib import Path

from sympy import false

from src import Config
from src.ingestion.connectors import PDFConnector, TextFileConnector
from src.ingestion.chunking import ChunkingFactory
from src.models.ollama_backend import OllamaBackend
from src.models.huggingface_backend import HuggingFaceBackend
from src.query.retriever import Retriever
from src.query.response_gen import ResponseGenerator
from src.evaluation.metrics import MetricsCalculator

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    Config.create_directories()


def initialize_components(model_name: str = "mistral-7b"):
    """
    Initialize all components for the RAG pipeline

    Args:
        model_name: Name of the model to use

    Returns:
        Dictionary of initialized components
    """
    logger.info(f"Initializing components with model: {model_name}")

    # Get model configuration
    model_config = Config.get_model_config(model_name)

    # Initialize LLM backend
    if model_config.backend == "ollama":
        llm_backend = OllamaBackend(model_config)
    elif model_config.backend == "huggingface":
        llm_backend = HuggingFaceBackend(model_config)
    else:
        raise ValueError(f"Unsupported backend: {model_config.backend}")

    # Initialize connectors
    pdf_connector = PDFConnector()
    text_connector = TextFileConnector()

    # Initialize chunking strategy
    chunking = ChunkingFactory.create_strategy(Config.CHUNKING)

    # Initialize retriever and response generator
    # Using ChromaDB-based retriever with optimized configuration
    retriever = Retriever(Config.VECTOR_STORE)

    # Log vector store statistics
    stats = retriever.get_collection_stats()
    logger.info(f"Vector store stats: {stats}")

    response_gen = ResponseGenerator(llm_backend, retriever)

    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()

    return {
        "llm_backend": llm_backend,
        "pdf_connector": pdf_connector,
        "text_connector": text_connector,
        "chunking": chunking,
        "retriever": retriever,
        "response_gen": response_gen,
        "metrics_calc": metrics_calc
    }


def process_documents(components: dict, document_dir: str):
    """
    Process documents in the specified directory

    Args:
        components: Dictionary of initialized components
        document_dir: Directory containing documents
    """
    logger.info(f"Processing documents from: {document_dir}")

    # Process PDFs
    pdf_files = list(Path(document_dir).glob("*.pdf"))
    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF files")
        for pdf_file in pdf_files:
            try:
                documents = components["pdf_connector"].load(pdf_file)
                chunks = components["chunking"].chunk_documents(documents)
                components["retriever"].index_documents(chunks)
                logger.info(f"Successfully processed: {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")

    # Process text files
    text_files = list(Path(document_dir).glob("*.txt"))
    if text_files:
        logger.info(f"Found {len(text_files)} text files")
        for text_file in text_files:
            try:
                documents = components["text_connector"].load(text_file)
                chunks = components["chunking"].chunk_documents(documents)
                components["retriever"].index_documents(chunks)
                logger.info(f"Successfully processed: {text_file.name}")
            except Exception as e:
                logger.error(f"Error processing {text_file.name}: {e}")


def query_pipeline(components: dict, query: str) -> dict:
    """
    Execute query pipeline

    Args:
        components: Dictionary of initialized components
        query: User query

    Returns:
        Response dictionary with metrics
    """
    logger.info(f"Processing query: {query}")

    # Generate response
    response = components["response_gen"].generate_response(query)

    # Calculate metrics
    metrics = components["metrics_calc"].calculate_response_metrics(
        query=query,
        response=response["answer"],
        retrieved_docs=response.get("retrieved_docs", [])
    )

    return {
        "query": query,
        "answer": response["answer"],
        "retrieved_context": response.get("retrieved_docs", []),
        "metrics": metrics
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="CS6493 LLM Applications")
    parser.add_argument(
        "--model",
        type=str,
        default="mistral-7b",
        choices=["mistral-7b", "t5-base", "llama2-7b"],
        help="Model to use for inference"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to process"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="./data/documents",
        help="Directory containing documents to process"
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process documents without querying"
    )
    parser.add_argument(
        "--rechunking",
        action="store_true",
        help="Rechunk the documents if this flag is set"
    )
    args = parser.parse_args()

    # Setup
    setup_directories()

    # Initialize components
    components = initialize_components(args.model)

    # Process documents
    if args.rechunking:
        if Path(args.documents).exists():
            process_documents(components, args.documents)
        else:
            logger.warning(f"Documents directory does not exist: {args.documents}")
    else:
        logger.info("Rechunking is disabled.")

    # Query or interactive mode
    if args.process_only:
        logger.info("Document processing complete")
        return

    if args.query:
        # Single query mode
        result = query_pipeline(components, args.query)
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Metrics: {result['metrics']}")
    else:
        # Interactive mode
        logger.info("Starting interactive mode. Type 'exit' to quit.")
        while True:
            try:
                query = input("\nEnter your query: ")
                if query.lower() in ["exit", "quit"]:
                    break
                if query.strip():
                    result = query_pipeline(components, query)
                    print(f"\nAnswer: {result['answer']}")
                    print(f"Metrics: {result['metrics']}")
            except KeyboardInterrupt:
                logger.info("Exiting...")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")


if __name__ == "__main__":
    main()
