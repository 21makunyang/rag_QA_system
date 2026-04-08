# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Natural Language Processing course project (CS6493) focused on building practical LLM applications using the LlamaIndex framework. The project emphasizes educational implementation and explores core challenges in real-world LLM application development.

## Core Requirements

### 1. System Architecture Design
- Implement at least one application type:
  - **Document QA System**: RAG pipelines with adaptive chunking strategies (256-token chunks with 10% overlap)
  - **Conversational Agent**: Chatbot with short-term memory management
  - **Autonomous Agent**: Task-oriented agents with API integration
- Integrate data connectors for PDFs, text files, or web content
- Compare performance of 2+ LLM backends (e.g., Mistral-7B vs. smaller models like T5-base)

### 2. Capability Evaluation
- Design test cases measuring response relevance and task completion rate
- Analyze memory-performance tradeoffs using different chunking strategies

## Getting Started

### Prerequisites
- Python 3.8+
- Ollama (for running quantized models locally)
- Poppler utilities (for PDF processing)

### Initial Setup
```bash
# Install Poppler for PDF processing
brew install poppler  # macOS
# or
apt-get install poppler-utils  # Linux

# Install Python dependencies
pip install -r requirements.txt

# Install Ollama for local model deployment
# See: https://ollama.ai/download

# Pull recommended models
ollama pull mistral:7b
ollama pull llama2:7b
```

### Project Structure Template
```
project/
├── data/                    # Data sources (PDFs, text files)
├── src/
│   ├── __init__.py
│   ├── main.py             # Main application entry point
│   ├── config.py           # Configuration settings
│   ├── models/             # LLM model configurations
│   ├── ingestion/          # Data ingestion pipelines
│   ├── query/              # Query/retrieval logic
│   ├── evaluation/         # Evaluation metrics and tests
│   └── utils/              # Utility functions
├── tests/                  # Unit and integration tests
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── CLAUDE.md              # This file
```

## Common Development Tasks

### Running the Application
```bash
# Run main application
python src/main.py

# Run with different model backends
python src/main.py --model mistral-7b
python src/main.py --model t5-base
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_evaluation.py

# Run with verbose output
python -m pytest -v
```

### Code Quality
```bash
# Lint code
python -m flake8 src/

# Format code
python -m black src/

# Type checking
python -m mypy src/
```

## Key Technical Decisions

### 1. Model Backend Strategy
For computational efficiency, prioritize:
- **Quantized models** (GPTQ-4bit) through Ollama
- **Local deployment** to avoid API costs and latency
- **Model comparison** between Mistral-7B and smaller models (T5-base)

### 2. Chunking Strategy
- **Default**: 256-token chunks with 10% overlap
- **Evaluation**: Test different chunking strategies for performance tradeoffs

### 3. Data Connectors
- Start with **PDF connectors** (most common use case)
- Support for **text files** as fallback
- Consider **web content** for extended functionality

## LLM Backend Configuration

When implementing model backends, consider:

```python
# Example configuration structure
MODEL_CONFIGS = {
    "mistral-7b": {
        "backend": "ollama",
        "model_name": "mistral:7b",
        "temperature": 0.7,
        "max_tokens": 512
    },
    "t5-base": {
        "backend": "huggingface",
        "model_name": "t5-base",
        "temperature": 0.7,
        "max_tokens": 512
    }
}
```

## Evaluation Metrics

Design test cases to measure:
1. **Response Relevance**: Cosine similarity with expected answers
2. **Task Completion Rate**: Success rate on predefined tasks
3. **Memory Usage**: RAM consumption during inference
4. **Latency**: Response time for queries

## Development Workflow

1. **Setup**: Initialize project structure and dependencies
2. **Infrastructure**: Implement core LlamaIndex components
3. **Data Ingestion**: Build PDF/text connectors with chunking
4. **Query System**: Implement RAG or conversational logic
5. **Model Integration**: Connect LLM backends
6. **Evaluation**: Create test suite and metrics
7. **Optimization**: Compare models and fine-tune parameters
8. **Documentation**: Update README with setup and usage

## Computational Constraints

- Use **quantized models** to fit within typical GPU memory constraints
- Implement **efficient chunking** to minimize token usage
- Consider **batch processing** for large document collections
- Monitor **memory usage** during development

## Testing Strategy

- Unit tests for individual components
- Integration tests for pipeline workflows
- Performance benchmarks across different models
- End-to-end evaluation on sample datasets

## Documentation Requirements

- Clear setup instructions in README.md
- Code comments for complex logic
- Performance comparison results
- Evaluation methodology and results