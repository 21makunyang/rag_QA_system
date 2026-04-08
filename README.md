# CS6493 LLM Applications with LlamaIndex

This project demonstrates practical implementations of Large Language Model applications using the LlamaIndex framework, focusing on educational implementation and exploration of core challenges in real-world LLM application development.

## Features

- **Document QA System**: RAG pipelines with adaptive chunking strategies
- **Multi-Backend Support**: Compare Mistral-7B and T5-base models
- **Data Connectors**: PDF and text file ingestion
- **Evaluation Framework**: Comprehensive metrics for response quality and performance
- **Local Deployment**: Use quantized models via Ollama for computational efficiency

## Project Structure

```
project/
├── data/                    # Data sources and documents
├── src/
│   ├── main.py             # Application entry point
│   ├── config.py           # Configuration management
│   ├── ingestion/          # Data ingestion pipeline
│   ├── models/             # LLM backend configurations
│   ├── query/              # RAG query system
│   ├── evaluation/         # Evaluation framework
│   └── utils/              # Utilities
├── tests/                  # Unit and integration tests
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- Ollama (for running quantized models locally)
- Poppler utilities (for PDF processing)

### Installation

1. **Install Poppler**:
```bash
# macOS
brew install poppler

# Linux
apt-get install poppler-utils
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install and configure Ollama**:
```bash
# Download Ollama from https://ollama.ai/download

# Pull recommended models
ollama pull mistral:7b
ollama pull llama2:7b
```

### Quick Start

1. **Prepare your documents**:
Place PDF or text files in the `data/documents/` directory.

2. **Run the application**:
```bash
# Process documents and start interactive mode
python src/main.py

# Run with specific model
python src/main.py --model mistral-7b

# Run a single query
python src/main.py --query "What is the main topic?"

# Process documents only
python src/main.py --process-only
```

### Configuration

The application is configured through `src/config.py`:

- **Model Configurations**: Set up different LLM backends
- **Chunking Strategy**: Configure 256-token chunks with 10% overlap
- **Vector Store**: ChromaDB settings for document storage

## Usage Examples

### Document Question Answering

```python
from src.main import initialize_components, process_documents, query_pipeline

# Initialize components
components = initialize_components("mistral-7b")

# Process documents
process_documents(components, "./data/documents")

# Query the system
result = query_pipeline(components, "What is the main topic of the document?")
print(f"Answer: {result['answer']}")
print(f"Metrics: {result['metrics']}")
```

### Model Comparison

```python
from src.models.ollama_backend import OllamaBackend
from src.models.huggingface_backend import HuggingFaceBackend
from src.config import Config

# Compare different backends
mistral_config = Config.get_model_config("mistral-7b")
t5_config = Config.get_model_config("t5-base")

mistral_backend = OllamaBackend(mistral_config)
t5_backend = HuggingFaceBackend(t5_config)
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_config.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## Evaluation Framework

The evaluation framework includes:

- **Response Quality Metrics**: Lexical diversity, repetition analysis
- **Context Utilization**: Document coverage and relevance scoring
- **Accuracy Metrics**: Semantic similarity with expected answers
- **Performance Metrics**: Latency and computational efficiency

### Running Evaluations

```bash
# Generate test cases
python -c "from src.evaluation.test_cases import TestCaseManager; tc = TestCaseManager(); tc.generate_test_suite()"

# Run evaluation metrics
python -c "from src.evaluation.metrics import MetricsCalculator; calc = MetricsCalculator()"
```

## Development Guidelines

### Code Quality

```bash
# Lint code
python -m flake8 src/

# Format code
python -m black src/

# Type checking
python -m mypy src/
```

### Adding New Features

1. Follow the existing module structure
2. Add unit tests for new functionality
3. Update documentation in README.md
4. Consider computational efficiency for LLM applications

## Performance Considerations

- **Quantized Models**: Use 4-bit quantization for memory efficiency
- **Batch Processing**: Process documents in batches when possible
- **Caching**: Implement caching for frequently accessed data
- **Vector Storage**: Use efficient vector databases like ChromaDB

## Contributing

This is an educational project for CS6493. When contributing:

1. Focus on educational value and clarity
2. Document any new concepts or techniques
3. Include example usage and test cases
4. Consider computational constraints

## License

This project is for educational purposes as part of CS6493 coursework.

## Acknowledgments

- **LlamaIndex**: Data framework for LLM applications
- **Ollama**: Local model deployment
- **HuggingFace**: Transformers and model hub

## Support

For issues and questions related to this project, please refer to:

1. Project documentation in `CLAUDE.md`
2. Course materials and specification
3. Inline code comments and docstrings