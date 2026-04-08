"""
Response generation module for RAG system
"""

import logging
from typing import Dict, List, Any, Optional
import time
from llama_index.core import PromptTemplate

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Response generator using RAG approach"""

    def __init__(self, llm_backend, retriever):
        """Initialize response generator"""
        self.llm_backend = llm_backend
        self.retriever = retriever

        # Define RAG prompt template
        self.rag_prompt = PromptTemplate(
            """You are a helpful AI assistant. Answer the user's question based on the provided context.

Context information is below:
{context_str}

Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: """
        )

        # Define chat prompt for conversational responses
        self.chat_prompt = PromptTemplate(
            """You are a helpful AI assistant. Use the following conversation history and context to respond.

Conversation History:
{chat_history}

Context:
{context_str}

Current Query: {query_str}
Response: """
        )

    def generate_response(
        self,
        query: str,
        top_k: int = 5,
        use_rag: bool = True,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response for a query

        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_rag: Whether to use RAG or not
            chat_history: Optional conversation history

        Returns:
            Response dictionary with answer and metadata
        """
        try:
            start_time = time.time()

            # Retrieve relevant documents if using RAG
            retrieved_docs = []
            context_str = ""

            if use_rag:
                retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
                context_str = self._format_context(retrieved_docs)
            else:
                context_str = "No additional context provided."

            # Format prompt based on whether we have chat history
            if chat_history:
                chat_history_str = self._format_chat_history(chat_history)
                prompt = self.chat_prompt.format(
                    chat_history=chat_history_str,
                    context_str=context_str,
                    query_str=query
                )
            else:
                prompt = self.rag_prompt.format(
                    context_str=context_str,
                    query_str=query
                )

            # Generate response
            response_text = self.llm_backend.generate(prompt)

            end_time = time.time()
            generation_time = end_time - start_time

            # Prepare response dictionary
            response = {
                "query": query,
                "answer": response_text,
                "retrieved_docs": retrieved_docs,
                "generation_time": generation_time,
                "model_info": self.llm_backend.get_model_info(),
                "context_used": len(retrieved_docs) > 0
            }

            logger.info(f"Response generated in {generation_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_streaming_response(
        self,
        query: str,
        top_k: int = 5,
        use_rag: bool = True,
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Generate streaming response for a query

        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_rag: Whether to use RAG or not
            chat_history: Optional conversation history

        Yields:
            Response chunks
        """
        try:
            # Retrieve relevant documents if using RAG
            retrieved_docs = []
            context_str = ""

            if use_rag:
                retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
                context_str = self._format_context(retrieved_docs)
            else:
                context_str = "No additional context provided."

            # Format prompt
            if chat_history:
                chat_history_str = self._format_chat_history(chat_history)
                prompt = self.chat_prompt.format(
                    chat_history=chat_history_str,
                    context_str=context_str,
                    query_str=query
                )
            else:
                prompt = self.rag_prompt.format(
                    context_str=context_str,
                    query_str=query
                )

            # Generate streaming response
            for chunk in self.llm_backend.generate_stream(prompt):
                yield {
                    "chunk": chunk,
                    "query": query,
                    "retrieved_docs": retrieved_docs
                }

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            raise

    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string"""
        if not retrieved_docs:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc['text'][:500]  # Limit context length
            score = doc.get('score', 0.0)
            context_parts.append(f"[Document {i} (score: {score:.3f})]\n{text}\n")

        return "\n".join(context_parts)

    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history into string"""
        if not chat_history:
            return "No previous conversation."

        history_parts = []
        for i, message in enumerate(chat_history[-5:], 1):  # Last 5 messages
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            history_parts.append(f"{role.capitalize()}: {content}")

        return "\n".join(history_parts)

    def generate_with_few_shot(
        self,
        query: str,
        examples: List[Dict[str, str]],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Generate response with few-shot examples

        Args:
            query: User query
            examples: List of example query-answer pairs
            top_k: Number of documents to retrieve

        Returns:
            Response dictionary
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
            context_str = self._format_context(retrieved_docs)

            # Format examples
            examples_str = ""
            for i, example in enumerate(examples, 1):
                examples_str += f"Example {i}:\nQuery: {example['query']}\nAnswer: {example['answer']}\n\n"

            # Create prompt with examples
            prompt = f"""You are a helpful AI assistant. Here are some examples of how to answer questions:

{examples_str}

Now, use the following context to answer the query:

Context:
{context_str}

Query: {query}
Answer: """

            # Generate response
            response_text = self.llm_backend.generate(prompt)

            return {
                "query": query,
                "answer": response_text,
                "retrieved_docs": retrieved_docs,
                "examples_used": len(examples)
            }

        except Exception as e:
            logger.error(f"Error in few-shot generation: {e}")
            raise

    def generate_comparison_response(
        self,
        query: str,
        other_backend,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate responses using two different backends for comparison

        Args:
            query: User query
            other_backend: Another LLM backend to compare with
            top_k: Number of documents to retrieve

        Returns:
            Comparison results
        """
        try:
            # Retrieve documents (same for both backends)
            retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
            context_str = self._format_context(retrieved_docs)

            # Format prompt
            prompt = self.rag_prompt.format(
                context_str=context_str,
                query_str=query
            )

            # Generate with first backend
            start_time1 = time.time()
            response1 = self.llm_backend.generate(prompt)
            time1 = time.time() - start_time1

            # Generate with second backend
            start_time2 = time.time()
            response2 = other_backend.generate(prompt)
            time2 = time.time() - start_time2

            return {
                "query": query,
                "responses": {
                    "backend1": {
                        "answer": response1,
                        "generation_time": time1,
                        "model_info": self.llm_backend.get_model_info()
                    },
                    "backend2": {
                        "answer": response2,
                        "generation_time": time2,
                        "model_info": other_backend.get_model_info()
                    }
                },
                "retrieved_docs": retrieved_docs
            }

        except Exception as e:
            logger.error(f"Error in comparison generation: {e}")
            raise