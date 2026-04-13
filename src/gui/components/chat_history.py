"""
Chat history component for conversation management
"""

import streamlit as st
from typing import List, Dict, Any
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def chat_history_section() -> None:
    """Chat history display and management section"""

    st.header("💬 Chat History")

    # History controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.info(f"Total messages: {len(st.session_state.chat_history)}")

    with col2:
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            clear_chat_history()
            st.rerun()

    with col3:
        if st.button("📥 Export", type="secondary", use_container_width=True):
            export_chat_history()

    # Display chat history
    if st.session_state.chat_history:
        display_chat_messages()
    else:
        st.info("No chat history yet. Start by asking a question!")

def display_chat_messages() -> None:
    """Display chat messages in a scrollable container"""

    # Create a scrollable container for chat history
    with st.container():
        for idx, message in enumerate(st.session_state.chat_history):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp")
            metadata = message.get("metadata", {})

            # Determine message style based on role
            if role == "user":
                with st.chat_message("user"):
                    st.write(content)

                    # Show metadata if available
                    if metadata and st.checkbox("Show details", key=f"user_meta_{idx}"):
                        st.json({"timestamp": timestamp, **metadata})

            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.write(content)

                    # Show performance metrics for assistant responses
                    if metadata:
                        with st.expander("Performance Details", expanded=False):
                            generation_time = metadata.get("generation_time", 0)
                            retrieved_docs = metadata.get("retrieved_docs_count", 0)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("⏱️ Generation Time", f"{generation_time:.2f}s")
                            with col2:
                                st.metric("📄 Docs Retrieved", retrieved_docs)

                            # Show model info if available
                            model_info = metadata.get("model_info", {})
                            if model_info:
                                st.json(model_info)

            else:
                # Unknown role
                st.warning(f"Unknown message role: {role}")
                st.write(content)

            # Add divider between messages (except for last message)
            if idx < len(st.session_state.chat_history) - 1:
                st.divider()

def clear_chat_history() -> None:
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.session_state.last_result = None
    st.success("✅ Chat history cleared")

def export_chat_history() -> None:
    """Export chat history to JSON"""

    if not st.session_state.chat_history:
        st.warning("⚠️ No chat history to export")
        return

    try:
        # Prepare export data
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_messages": len(st.session_state.chat_history),
            "messages": st.session_state.chat_history
        }

        # Convert to JSON
        json_data = json.dumps(export_data, indent=2, ensure_ascii=False)

        # Create download button
        st.download_button(
            label="📥 Download Chat History",
            data=json_data,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            type="primary"
        )

    except Exception as e:
        st.error(f"❌ Error exporting chat history: {str(e)}")
        logger.error(f"Error exporting chat history: {e}")

def add_to_chat_history(role: str, content: str, metadata: Dict[str, Any] = None) -> None:
    """Add a message to chat history (helper function for other components)"""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    st.session_state.chat_history.append(message)

def get_recent_chat_history(max_messages: int = 5) -> List[Dict[str, Any]]:
    """Get recent chat history for context"""
    if not st.session_state.chat_history:
        return []

    # Return the last N messages
    return st.session_state.chat_history[-max_messages:]