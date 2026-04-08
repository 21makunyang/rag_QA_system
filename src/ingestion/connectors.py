"""
Data connectors for loading documents from various sources
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

from llama_index.core import Document

logger = logging.getLogger(__name__)

class BaseConnector(ABC):
    """Base class for data connectors"""

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """Load documents from file"""
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Check if connector supports the file type"""
        pass

class PDFConnector(BaseConnector):
    """Connector for PDF files using LlamaIndex"""

    def __init__(self):
        """Initialize PDF connector"""
        self.supported_extensions = {'.pdf'}

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from PDF file

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports(str(file_path)):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        try:
            # Use LlamaIndex's built-in PDF reader
            from llama_index.readers.file import PDFReader

            reader = PDFReader()
            documents = reader.load_data(str(file_path))

            logger.info(f"Successfully loaded {len(documents)} documents from {file_path.name}")
            return documents

        except Exception as e:
            logger.error(f"Error loading PDF from {file_path}: {e}")
            raise

    def supports(self, file_path: str) -> bool:
        """Check if file is a PDF"""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions


class TextFileConnector(BaseConnector):
    """Connector for text files"""

    def __init__(self, encoding: str = 'utf-8'):
        """Initialize text file connector"""
        self.supported_extensions = {'.txt', '.md', '.rst', '.json'}
        self.encoding = encoding

    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from text file

        Args:
            file_path: Path to text file

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports(str(file_path)):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                text = f.read()

            # Create a single document from the file
            document = Document(
                text=text,
                metadata={
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_type': file_path.suffix
                }
            )

            logger.info(f"Successfully loaded text file: {file_path.name}")
            return [document]

        except UnicodeDecodeError:
            # Try different encodings if UTF-8 fails
            encodings = ['latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    document = Document(
                        text=text,
                        metadata={
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'file_type': file_path.suffix,
                            'encoding': encoding
                        }
                    )
                    logger.info(f"Successfully loaded text file: {file_path.name} with encoding {encoding}")
                    return [document]
                except:
                    continue

            raise ValueError(f"Could not decode file {file_path} with any supported encoding")

        except Exception as e:
            logger.error(f"Error loading text file from {file_path}: {e}")
            raise

    def supports(self, file_path: str) -> bool:
        """Check if file is a supported text file"""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions