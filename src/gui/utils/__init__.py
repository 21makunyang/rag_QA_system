"""
GUI utilities package for CS6493 LLM Applications
"""

from src.gui.utils.session_state import (
    initialize_session_state,
    get_backend_info,
    get_retriever_stats,
    add_to_chat_history,
    clear_chat_history,
    update_processing_status,
    save_last_result,
    get_last_result
)

__all__ = [
    "initialize_session_state",
    "get_backend_info",
    "get_retriever_stats",
    "add_to_chat_history",
    "clear_chat_history",
    "update_processing_status",
    "save_last_result",
    "get_last_result"
]