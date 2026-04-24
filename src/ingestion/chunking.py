"""
Chunking strategies for document processing
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from src.config import Config
#新增
from typing import List, Dict, Any, Optional
from llama_index.core.node_parser import HierarchicalNodeParser, NodeParser
from llama_index.core.schema import BaseNode



logger = logging.getLogger(__name__)

class ChunkingStrategy(ABC):
    """Base class for chunking strategies"""

    def __init__(self, config: Config):
        """Initialize chunking strategy"""
        self.config = config

    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents according to strategy"""
        pass

class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size token-based chunking strategy"""

    def __init__(self, config: Config):
        """Initialize fixed-size chunking"""
        super().__init__(config)
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using fixed-size token splitting

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        logger.info(f"Chunking {len(documents)} documents with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunked_documents = []

        for doc in documents:
            try:
                # Split the document text
                text_chunks = text_splitter.split_text(doc.text)

                # Create new Document objects for each chunk
                for i, chunk_text in enumerate(text_chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'chunk_strategy': 'fixed_size',
                        'chunk_size': self.chunk_size,
                        'chunk_overlap': self.chunk_overlap
                    })

                    chunk_doc = Document(
                        text=chunk_text,
                        metadata=chunk_metadata
                    )
                    chunked_documents.append(chunk_doc)

            except Exception as e:
                logger.error(f"Error chunking document: {e}")
                continue

        logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        return chunked_documents

class SentenceBasedChunking(ChunkingStrategy):
    """Sentence-based chunking strategy"""

    def __init__(self, config: Config):
        """Initialize sentence-based chunking"""
        super().__init__(config)
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using sentence splitting

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        logger.info(f"Chunking {len(documents)} documents using sentence-based strategy")

        sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunked_documents = []

        for doc in documents:
            try:
                # Use LlamaIndex's sentence splitter
                nodes = sentence_splitter.get_nodes_from_documents([doc])

                # Convert nodes back to documents
                for i, node in enumerate(nodes):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(nodes),
                        'chunk_strategy': 'sentence_based',
                        'chunk_size': self.chunk_size,
                        'chunk_overlap': self.chunk_overlap
                    })

                    chunk_doc = Document(
                        text=node.text,
                        metadata=chunk_metadata
                    )
                    chunked_documents.append(chunk_doc)

            except Exception as e:
                logger.error(f"Error chunking document: {e}")
                continue

        logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        return chunked_documents

#新增调用LlamaIndex 的 HierarchicalNodeParser
class HierarchicalChunking(ChunkingStrategy):
    def __init__(self, config: Config, base_parser: Optional[NodeParser] = None):
        super().__init__(config)
        self.chunk_sizes = config.chunk_sizes
        self.include_metadata = config.hierarchical_include_metadata
        self.include_prev_next_rel = config.hierarchical_include_prev_next_rel

        # 不再需要 base_parser，HierarchicalNodeParser 内部会自动处理
        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes,
            include_metadata=self.include_metadata,
            include_prev_next_rel=self.include_prev_next_rel,
            # node_parser=base_parser   # 删除此行
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Hierarchical chunking {len(documents)} docs with sizes {self.chunk_sizes}")

        chunked_documents = []
        for doc in documents:
            try:
                nodes: List[BaseNode] = self.parser.get_nodes_from_documents([doc])
                for i, node in enumerate(nodes):
                    # 构建元数据，保留层级信息
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(nodes),
                        'chunk_strategy': 'hierarchical',
                        'chunk_sizes': self.chunk_sizes,
                        'node_id': node.node_id,
                        # 关键字段：保留父子关系
                        'parent_node_id': getattr(node, 'parent_node', None),
                        'child_node_ids': getattr(node, 'child_nodes', [])
                    })
                    chunk_doc = Document(text=node.get_content(), metadata=chunk_metadata)
                    chunked_documents.append(chunk_doc)
            except Exception as e:
                logger.error(f"Error in hierarchical chunking: {e}")
                continue

        logger.info(f"Created {len(chunked_documents)} hierarchical chunks")
        return chunked_documents

# class ChunkingFactory:
#     """Factory for creating chunking strategies"""
#
#     @staticmethod
#     def create_strategy(config: Config) -> ChunkingStrategy:
#         """
#         Create chunking strategy based on configuration
#
#         Args:
#             config: Chunking configuration
#
#         Returns:
#             ChunkingStrategy instance
#         """
#         if config.strategy == "fixed":
#             return FixedSizeChunking(config)
#         elif config.strategy == "sentence":
#             return SentenceBasedChunking(config)
#         else:
#             logger.warning(f"Unknown chunking strategy: {config.strategy}, using fixed size")
#             return FixedSizeChunking(config)
#新增
class ChunkingFactory:
    @staticmethod
    def create_strategy(config: Config, embed_model=None) -> ChunkingStrategy:
        strategy = config.strategy.lower()
        if strategy == "fixed":
            return FixedSizeChunking(config)
        elif strategy == "sentence":
            return SentenceBasedChunking(config)
        elif strategy == "hierarchical":
            return HierarchicalChunking(config)
        # elif strategy == "semantic":
        #     return SemanticChunking(config, embed_model)
        # elif strategy == "dynamic":
        #     return DynamicChunking(config)
        else:
            logger.warning(f"Unknown strategy {strategy}, fallback to fixed")
            return FixedSizeChunking(config)