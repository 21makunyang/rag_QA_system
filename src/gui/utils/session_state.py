"""
Session state management for Streamlit GUI
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from src.config import Config
from src.ingestion.chunking import ChunkingStrategy

def initialize_session_state():
    """Initialize session state variables if they don't exist"""

    # Model and backend state
    if 'llm_backend' not in st.session_state:
        st.session_state.llm_backend = None

    if 'retriever' not in st.session_state:
        st.session_state.retriever = None

    if 'response_gen' not in st.session_state:
        st.session_state.response_gen = None

    # Document processing state
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False

    if 'document_count' not in st.session_state:
        st.session_state.document_count = 0

    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Current model
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "mistral-7b"

    # Processing status
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    # Configuration state
    if 'chunking_config' not in st.session_state:
        st.session_state.chunking_config = Config.CHUNKING

    # Results storage
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

def get_backend_info(llm_backend) -> Dict[str, Any]:
    """Get information about the current LLM backend"""
    if llm_backend is None:
        return {"status": "Not initialized", "model": "None", "backend": "None"}

    try:
        model_info = llm_backend.get_model_info()
        return {
            "status": "Connected",
            "model": model_info.get("model_name", "Unknown"),
            "backend": model_info.get("backend", "Unknown"),
            "temperature": getattr(llm_backend, 'temperature', 'N/A'),
            "max_tokens": getattr(llm_backend, 'max_tokens', 'N/A')
        }
    except Exception as e:
        return {"status": f"Error: {str(e)}", "model": "Unknown", "backend": "Unknown"}

def get_retriever_stats(retriever) -> Dict[str, Any]:
    """Get statistics about the vector store"""
    if retriever is None:
        return {"document_count": 0, "status": "Not initialized"}

    try:
        stats = retriever.get_collection_stats()
        return {
            "document_count": stats.get("document_count", 0),
            "collection_name": stats.get("collection_name", "Unknown"),
            "persist_dir": stats.get("persist_dir", "Unknown"),
            "status": "Ready"
        }
    except Exception as e:
        return {"document_count": 0, "status": f"Error: {str(e)}"}

def add_to_chat_history(role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
    """Add a message to chat history"""
    message = {
        "role": role,
        "content": content,
        "timestamp": st.session_state.get("last_query_time", None),
        "metadata": metadata or {}
    }
    st.session_state.chat_history.append(message)

def clear_chat_history():
    """Clear the chat history"""
    st.session_state.chat_history = []

def update_processing_status(is_processing: bool, message: str = ""):
    """Update processing status with spinner and message"""
    st.session_state.is_processing = is_processing
    if is_processing and message:
        st.spinner(message)

def save_last_result(result: Dict[str, Any]):
    """Save the last query result"""
    st.session_state.last_result = result

def get_last_result() -> Optional[Dict[str, Any]]:
    """Get the last query result"""
    return st.session_state.last_result