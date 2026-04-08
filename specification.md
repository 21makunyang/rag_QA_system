** Building Practical LLM Applications with LlamaIndex**

LlamaIndex is a cutting-edge data framework for connecting custom data sources to large language models (LLMs). This project focuses on developing production-grade LLM applications with an emphasis on educational implementation rather than enterprise deployment. Students will explore core challenges in real-world LLM application development, including advanced retrieval strategies and automated evaluation, while considering computational constraints.

The project requires addressing two core technical challenges:

1.  **System Architecture Design:** Construct application pipelines using LlamaIndex's data connectors and retrieval modules:
    *   Implement at least one application types from:
        *   Document QA System: Build RAG pipelines with adaptive chunking strategies (e.g., 256-token chunks with 10% overlap)
        *   Conversational Agent: Develop chatbot with short-term memory management
        *   Autonomous Agent: Create simple task-oriented agents with API integration
    *   Integrate data connectors for at least one source type (PDFs, text files, or web content)
    *   Compare performance of 2+ LLM backends (e.g., Mistral-7B vs. smaller models like T5-base)

2.  **Capability Evaluation:** Establish practical evaluation metrics:
    *   Design test cases measuring response relevance and task completion rate
    *   Analyze memory-performance tradeoffs using different chunking strategies

**Advanced suggestions:** Explore quantization techniques for model deployment or implement basic human feedback mechanisms.

**Hint:** For computational efficiency, consider using quantized models (e.g., GPTQ-4bit versions) through Ollama.