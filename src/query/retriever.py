"""
Document retriever for RAG system using ChromaDB vector database
"""

import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from src import Config

logger = logging.getLogger(__name__)


class Retriever:
    """
    Document retriever using ChromaDB vector similarity search

    This class provides a robust RAG retrieval system using ChromaDB as the
    vector database backend. It handles document indexing, vector similarity
    search, and integration with LlamaIndex.
    """

    def __init__(self, vector_store_config: Config):
        """
        Initialize retriever with vector store configuration

        Args:
            vector_store_config: Configuration object containing vector store settings
        """
        self.config = vector_store_config
        self.chroma_client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self.vector_store: Optional[ChromaVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None

        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """
        Initialize ChromaDB vector store and LlamaIndex integration

        This method:
        1. Creates a persistent ChromaDB client
        2. Gets or creates the document collection
        3. Initializes the LlamaIndex vector store
        4. Creates the vector index for retrieval
        """
        try:
            # Disable ChromaDB telemetry to avoid compatibility issues
            import os
            os.environ["ANONYMIZED_TELEMETRY"] = "False"

            # Initialize ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get existing collection or create new one
            try:
                self.collection = self.chroma_client.get_collection(self.config.collection_name)
                logger.info(f"Using existing collection: {self.config.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(self.config.collection_name)
                logger.info(f"Created new collection: {self.config.collection_name}")

            # Initialize LlamaIndex vector store with ChromaDB backend
            self.vector_store = ChromaVectorStore(
                chroma_collection=self.collection
            )

            # Create storage context for persistence
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            # Initialize embedding model (HuggingFace local model to avoid OpenAI dependency)
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.core import Settings as LlamaIndexSettings

            embed_model = HuggingFaceEmbedding(
                model_name=Config.EMBEDDING_MODEL,
                embed_batch_size=16
            )

            # Configure LlamaIndex to use the local embedding model
            LlamaIndexSettings.embed_model = embed_model

            # Initialize vector index from vector store
            # This enables efficient similarity search operations
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
                show_progress=True
            )

            logger.info("Successfully initialized vector store and index")
            logger.info(f"Using embedding model: {Config.EMBEDDING_MODEL}")

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents in the vector store

        This method adds documents to the vector database, creating embeddings
        and storing them for efficient retrieval.

        Args:
            documents: List of LlamaIndex Document objects to index
        """
        try:
            logger.info(f"Indexing {len(documents)} documents")

            # Add documents to the vector index
            # LlamaIndex handles embedding generation automatically
            for doc in documents:
                self.index.insert(doc)

            # Persist the index to disk for future use
            self.storage_context.persist(persist_dir=self.config.persist_dir)

            logger.info(f"Successfully indexed {len(documents)} documents")
            logger.info(f"Total documents in index: {self.get_document_count()}")

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query using vector similarity

        This is the main retrieval method that:
        1. Converts the query to embeddings
        2. Performs similarity search in the vector database
        3. Returns the most relevant documents

        Args:
            query: Query string from user
            top_k: Number of documents to retrieve (default: 5)

        Returns:
            List of retrieved documents with scores and metadata
        """
        try:
            logger.info(f"Retrieving documents for query: {query[:50]}...")

            # Create retriever with specified similarity threshold
            retriever = self.index.as_retriever(similarity_top_k=top_k)

            # Create query bundle with the query string
            query_bundle = QueryBundle(query_str=query)

            # Retrieve relevant document nodes
            nodes_with_scores: List[NodeWithScore] = retriever._retrieve(query_bundle)

            # Format results for response generation
            retrieved_docs = []
            for node_with_score in nodes_with_scores:
                doc_info = {
                    'text': node_with_score.node.text,
                    'score': node_with_score.score,
                    'metadata': node_with_score.node.metadata,
                    'doc_id': node_with_score.node.id_
                }
                retrieved_docs.append(doc_info)

            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    def search_by_vector(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search by vector similarity using pre-computed embeddings

        This method allows direct vector search, useful when you already have
        embeddings from a different source.

        Args:
            query_vector: Query embedding vector (list of floats)
            top_k: Number of results to return (default: 5)

        Returns:
            List of matching documents with similarity scores
        """
        try:
            # Use LlamaIndex's native vector search capabilities
            query_bundle = QueryBundle(embeddings=query_vector)

            # Perform similarity search
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes_with_scores = retriever._retrieve(query_bundle)

            # Format results
            retrieved_docs = []
            for node_with_score in nodes_with_scores:
                doc_info = {
                    'text': node_with_score.node.text,
                    'score': node_with_score.score,
                    'metadata': node_with_score.node.metadata,
                    'doc_id': node_with_score.node.id_
                }
                retrieved_docs.append(doc_info)

            return retrieved_docs

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the index

        Returns:
            Number of documents indexed
        """
        try:
            if self.collection:
                return len(self.collection.get()['ids'])
            return 0
        except Exception as e:
            logger.warning(f"Error getting document count: {e}")
            return 0

    def clear_index(self) -> None:
        """
        Clear all documents from the index

        This method removes all documents from the vector store, useful
        for resetting the system or starting fresh.
        """
        try:
            if self.collection:
                self.collection.delete()
                logger.info("Cleared all documents from index")
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Returns:
            Dictionary containing collection statistics
        """
        try:
            if self.collection:
                collection_data = self.collection.get()
                return {
                    'document_count': len(collection_data['ids']),
                    'collection_name': self.config.collection_name,
                    'persist_dir': self.config.persist_dir
                }
            return {'document_count': 0}
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}