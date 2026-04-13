"""
Metrics and statistics display component
"""

import streamlit as st
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def metrics_display_section(components: Dict[str, Any]) -> None:
    """Metrics and statistics display section"""

    st.header("📊 System Metrics & Statistics")

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    # Get backend info
    backend_info = get_backend_info(components)

    # Get retriever stats
    retriever_stats = get_retriever_stats(components)

    # Display metrics
    with col1:
        st.metric(
            "📄 Documents Indexed",
            retriever_stats.get("document_count", 0),
            help="Total number of documents in vector store"
        )

    with col2:
        st.metric(
            "🤖 Backend Status",
            backend_info.get("status", "Unknown"),
            help="Current status of the LLM backend"
        )

    with col3:
        st.metric(
            "📡 Model",
            backend_info.get("model", "Not initialized"),
            help="Currently loaded model"
        )

    with col4:
        collection_name = retriever_stats.get("collection_name", "Unknown")
        st.metric(
            "🗄️ Collection",
            collection_name,
            help="Name of the ChromaDB collection"
        )

    # Detailed stats in expanders
    with st.expander("🔧 Backend Details", expanded=False):
        if backend_info and backend_info.get("status") != "Not initialized":
            st.json({
                "status": backend_info.get("status", "Unknown"),
                "model": backend_info.get("model", "Unknown"),
                "backend": backend_info.get("backend", "Unknown"),
                "temperature": backend_info.get("temperature", "N/A"),
                "max_tokens": backend_info.get("max_tokens", "N/A")
            })
        else:
            st.info("Backend not initialized")

    with st.expander("🗄️ Vector Store Details", expanded=False):
        if retriever_stats and retriever_stats.get("status") != "Not initialized":
            st.json(retriever_stats)
        else:
            st.info("Vector store not initialized")

    # Last query performance
    if st.session_state.get("last_result"):
        with st.expander("📈 Last Query Performance", expanded=False):
            last_result = st.session_state.last_result
            st.json({
                "generation_time": f"{last_result.get('generation_time', 0):.2f}s",
                "retrieved_docs": len(last_result.get("retrieved_docs", [])),
                "context_used": last_result.get("context_used", False),
                "model": last_result.get("model_info", {}).get("model_name", "Unknown")
            })

def get_backend_info(components: Dict[str, Any]) -> Dict[str, Any]:
    """Get information about the current LLM backend"""
    if not components.get("llm_backend"):
        return {"status": "Not initialized", "model": "None", "backend": "None"}

    try:
        llm_backend = components["llm_backend"]
        model_info = llm_backend.get_model_info()
        return {
            "status": "Connected",
            "model": model_info.get("model_name", "Unknown"),
            "backend": model_info.get("backend", "Unknown"),
            "temperature": getattr(llm_backend, 'temperature', 'N/A'),
            "max_tokens": getattr(llm_backend, 'max_tokens', 'N/A')
        }
    except Exception as e:
        logger.error(f"Error getting backend info: {e}")
        return {"status": f"Error: {str(e)}", "model": "Unknown", "backend": "Unknown"}

def get_retriever_stats(components: Dict[str, Any]) -> Dict[str, Any]:
    """Get statistics about the vector store"""
    if not components.get("retriever"):
        return {"document_count": 0, "status": "Not initialized"}

    try:
        retriever = components["retriever"]
        stats = retriever.get_collection_stats()
        return {
            "document_count": stats.get("document_count", 0),
            "collection_name": stats.get("collection_name", "Unknown"),
            "persist_dir": stats.get("persist_dir", "Unknown"),
            "status": "Ready"
        }
    except Exception as e:
        logger.error(f"Error getting retriever stats: {e}")
        return {"document_count": 0, "status": f"Error: {str(e)}"}