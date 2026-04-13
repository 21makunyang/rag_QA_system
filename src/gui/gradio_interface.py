"""
Gradio-based GUI interface for RAG QA System
"""

import gradio as gr
import logging
import psutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

from src import Config
from src.ingestion.connectors import PDFConnector, TextFileConnector
from src.ingestion.chunking import ChunkingFactory
from src.models.ollama_backend import OllamaBackend
from src.models.huggingface_backend import HuggingFaceBackend
from src.query.retriever import Retriever
from src.query.response_gen import ResponseGenerator
from src.evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

# Global components
_components = None
_query_count = 0
_start_time = time.time()


def initialize_components(model_name: str = "mistral-7b") -> Dict:
    """
    Initialize all RAG components

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
        "metrics_calc": metrics_calc,
        "model_name": model_name
    }


def get_system_status() -> str:
    """Get current system status as formatted string"""
    process = psutil.Process()
    memory_info = process.memory_info()

    status = {
        "状态": "运行中",
        "内存": f"{memory_info.rss / 1024 / 1024:.1f}MB",
        "CPU": f"{psutil.cpu_percent()}%",
        "运行时间": f"{int(time.time() - _start_time)}s"
    }

    return "\n".join([f"{k}: {v}" for k, v in status.items()])


def get_query_statistics() -> List[List]:
    """Get query statistics"""
    global _query_count

    # 这里可以计算实际的统计数据
    avg_time = "0.5s"  # 示例数据

    return [
        ["总查询数", _query_count],
        ["平均响应时间", avg_time],
        ["向量文档数", "待统计"],
        ["当前模型", _components.get("model_name", "未初始化") if _components else "未初始化"]
    ]


def process_uploaded_file(file, model: str) -> Tuple[List[List], str]:
    """
    Process uploaded file

    Args:
        file: Uploaded file object
        model: Selected model name

    Returns:
        Tuple of (document_list, status_message)
    """
    if file is None:
        return [], "错误: 未选择文件"

    try:
        global _components
        if _components is None:
            _components = initialize_components(model)

        file_path = Path(file.name)
        file_name = file_path.name
        file_ext = file_path.suffix.lower()

        # Save uploaded file
        upload_dir = Path(Config.DATA_DIR) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        save_path = upload_dir / file_name
        with open(save_path, "wb") as f:
            f.write(file.read())

        # Process based on file type
        if file_ext == ".pdf":
            documents = _components["pdf_connector"].load(save_path)
        elif file_ext == ".txt":
            documents = _components["text_connector"].load(save_path)
        else:
            return [], f"错误: 不支持的文件格式 {file_ext}"

        # Chunk and index documents
        chunks = _components["chunking"].chunk_documents(documents)
        _components["retriever"].index_documents(chunks)

        logger.info(f"Successfully processed: {file_name}")

        # Get updated document list
        doc_list = get_document_list()

        return doc_list, f"文档处理完成: {file_name}"

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return [], f"错误: {str(e)}"


def get_document_list() -> List[List]:
    """Get list of processed documents"""
    try:
        # 获取已处理的文档列表
        # 这里可以连接到实际存储文档信息的数据库或文件
        upload_dir = Path(Config.DATA_DIR) / "uploads"
        if upload_dir.exists():
            files = list(upload_dir.glob("*.pdf")) + list(upload_dir.glob("*.txt"))
            doc_list = [[f.name, "已处理", "N/A"] for f in files]
            return doc_list
        return []
    except Exception as e:
        logger.error(f"Error getting document list: {e}")
        return []


def chat_query(message: str, chat_history: List, model: str, temperature: float) -> Tuple:
    """
    Process user query

    Args:
        message: User's question
        chat_history: Chat history
        model: Selected model name
        temperature: Temperature parameter

    Returns:
        Tuple of (updated_chat_history, status_message, memory_usage, query_stats)
    """
    global _components, _query_count

    if not message:
        return chat_history, "错误: 请输入问题", None, None

    try:
        if _components is None or _components.get("model_name") != model:
            _components = initialize_components(model)
            _components["response_gen"].llm_backend.update_temperature(temperature)

        # Increment query count
        _query_count += 1

        # Process query
        response = _components["response_gen"].generate_response(message)

        # Calculate metrics
        metrics = _components["metrics_calc"].calculate_response_metrics(
            query=message,
            response=response["answer"],
            retrieved_docs=response.get("retrieved_docs", [])
        )

        # Update chat history - Gradio Chatbot需要特定格式
        chat_history.append({
            "role": "user",
            "content": message
        })
        chat_history.append({
            "role": "assistant",
            "content": response["answer"]
        })

        # Get system status
        status = get_system_status()

        # Get query statistics
        stats = get_query_statistics()

        return chat_history, "查询完成", status, stats

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return chat_history, f"错误: {str(e)}", None, None


def clear_chat_history():
    """Clear chat history"""
    return []


def create_rag_interface():
    """Create Gradio interface for RAG QA System"""

    # Pre-initialize components with default model when interface is created
    global _components
    if _components is None:
        logger.info("Pre-loading default model components...")
        _components = initialize_components("mistral-7b")

    with gr.Blocks(title="RAG问答系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 RAG问答系统")

        # Top settings area
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=["mistral-7b", "llama2-7b", "t5-base"],
                    value="mistral-7b",
                    label="选择模型"
                )
            with gr.Column(scale=1):
                temperature = gr.Slider(
                    0.1, 1.0, value=0.7, step=0.1,
                    label="Temperature (控制响应创造性)"
                )

        # Document management tab
        with gr.Tab("📄 文档管理"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="上传文档 (PDF/TXT)",
                        file_types=[".pdf", ".txt"],
                        type="filepath"
                    )
                    upload_btn = gr.Button("🚀 处理文档", variant="primary")
                with gr.Column(scale=1):
                    doc_list = gr.Dataframe(
                        headers=["文件名", "状态", "操作"],
                        label="已上传文档列表",
                        value=get_document_list()
                    )
                    process_all_btn = gr.Button("📦 处理所有文档")

        # QA interaction tab
        with gr.Tab("💬 问答交互"):
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        avatar_images=(None, "🤖"),
                        render_markdown=True
                    )
                    msg = gr.Textbox(
                        label="输入您的问题",
                        lines=3,
                        placeholder="输入您的问题，然后按回车或点击提交..."
                    )
                    with gr.Row():
                        submit_btn = gr.Button("✈️ 提交", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ 清空历史", size="lg")

        # Performance monitoring tab
        with gr.Tab("📊 性能监控"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🖥️ 系统状态")
                    status_indicator = gr.Textbox(
                        value=get_system_status(),
                        label="系统状态",
                        lines=5,
                        interactive=False
                    )
                    memory_usage = gr.Textbox(
                        value="内存: 0MB",
                        label="内存使用",
                        interactive=False
                    )
                with gr.Column():
                    gr.Markdown("### 📈 查询统计")
                    query_stats = gr.Dataframe(
                        headers=["指标", "值"],
                        value=get_query_statistics(),
                        label="系统统计信息"
                    )

        # Event handlers
        upload_btn.click(
            fn=process_uploaded_file,
            inputs=[file_upload, model_selector],
            outputs=[doc_list, status_indicator],
            show_progress=True
        )

        submit_btn.click(
            fn=chat_query,
            inputs=[msg, chatbot, model_selector, temperature],
            outputs=[chatbot, status_indicator, memory_usage, query_stats],
            show_progress=True
        ).then(
            fn=lambda x: "",
            inputs=[msg],
            outputs=[msg]
        )

        clear_btn.click(
            fn=clear_chat_history,
            outputs=[chatbot]
        )

        # Also submit on Enter key
        msg.submit(
            fn=chat_query,
            inputs=[msg, chatbot, model_selector, temperature],
            outputs=[chatbot, status_indicator, memory_usage, query_stats],
            show_progress=True
        ).then(
            fn=lambda x: "",
            inputs=[msg],
            outputs=[msg]
        )

        return demo


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and launch interface
    interface = create_rag_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
        show_error=True,
        debug=True
    )