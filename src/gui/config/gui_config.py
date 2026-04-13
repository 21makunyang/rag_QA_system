"""
GUI configuration for CS6493 LLM Applications
"""

from dataclasses import dataclass
from typing import List

@dataclass
class GUIConfig:
    """Configuration for GUI interface"""
    port: int = 8501
    host: str = "localhost"
    theme: str = "light"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = None
    default_top_k: int = 5
    enable_dark_mode: bool = True
    page_title: str = "CS6493 RAG QA System"
    page_icon: str = "🤖"
    layout: str = "wide"

    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ["pdf", "txt", "md"]