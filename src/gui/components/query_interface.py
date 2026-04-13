"""
Query interface component for RAG system
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging

from src.query.response_gen import ResponseGenerator

logger = logging.getLogger(__name__)

def query_interface_section(components: Dict[str, Any]) -> None:
    """Main query interface section"""

    st.header("💬 Query Interface")

    # Query configuration
    with st.expander("Query Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            top_k = st.number_input(
                "Top K Documents",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of relevant documents to retrieve"
            )

            use_rag = st.checkbox(
                "Use RAG",
                value=True,
                help="Enable Retrieval-Augmented Generation"
            )

        with col2:
            streaming = st.checkbox(
                "Streaming Response",
                value=False,
                help="Enable streaming response (may be slower)"
            )

            show_context = st.checkbox(
                "Show Retrieved Context",
                value=True,
                help="Display the documents used for answering"
            )

    # Query input
    query = st.text_area(
        "Enter your query",
        height=100,
        placeholder="Type your question here. You can ask about the uploaded documents or anything else...",
        help="Enter your question or query for the RAG system"
    )

    # Submit button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        submit_button = st.button(
            "🚀 Submit Query",
            type="primary",
            use_container_width=True,
            disabled=not query.strip()
        )

    with col2:
        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    # Handle clear button
    if clear_button:
        st.session_state.last_result = None
        st.rerun()

    # Handle query submission
    if submit_button and query.strip():
        process_query(
            query=query,
            components=components,
            top_k=top_k,
            use_rag=use_rag,
            streaming=streaming,
            show_context=show_context
        )

def process_query(
    query: str,
    components: Dict[str, Any],
    top_k: int,
    use_rag: bool,
    streaming: bool,
    show_context: bool
) -> None:
    """Process user query and display response"""

    # Check if components are initialized
    if not components.get("response_gen"):
        st.error("❌ Response generator not initialized. Please check model configuration.")
        return

    if not components.get("retriever"):
        st.error("❌ Retriever not initialized. Please check vector store configuration.")
        return

    # Update processing status
    with st.spinner("🤔 Generating response..."):
        try:
            # Record start time
            import time
            start_time = time.time()

            # Generate response
            if streaming:
                response = generate_streaming_response(
                    query=query,
                    response_gen=components["response_gen"],
                    top_k=top_k,
                    use_rag=use_rag
                )
            else:
                response = components["response_gen"].generate_response(
                    query=query,
                    top_k=top_k,
                    use_rag=use_rag,
                    chat_history=st.session_state.chat_history[-5:]  # Last 5 messages
                )

            # Calculate generation time
            generation_time = time.time() - start_time
            response["generation_time"] = generation_time

            # Save last result
            st.session_state.last_result = response
            st.session_state.last_query_time = generation_time

            # Add to chat history
            from src.gui.utils.session_state import add_to_chat_history
            add_to_chat_history("user", query)

            # Display response
            display_response(response, show_context)

            # Add assistant response to chat history
            add_to_chat_history("assistant", response["answer"], {
                "generation_time": generation_time,
                "retrieved_docs_count": len(response.get("retrieved_docs", [])),
                "model_info": response.get("model_info", {})
            })

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            logger.error(f"Error generating response: {e}")

def generate_streaming_response(
    query: str,
    response_gen: ResponseGenerator,
    top_k: int,
    use_rag: bool
) -> Dict[str, Any]:
    """Generate streaming response and display in real-time"""

    # Placeholder for response
    response_placeholder = st.empty()
    full_response = ""

    # Streaming container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            for chunk_data in response_gen.generate_streaming_response(
                query=query,
                top_k=top_k,
                use_rag=use_rag
            ):
                chunk = chunk_data.get("chunk", "")
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

        except Exception as e:
            st.error(f"Streaming error: {str(e)}")
            logger.error(f"Streaming error: {e}")

    # Return final response structure
    return {
        "query": query,
        "answer": full_response,
        "retrieved_docs": chunk_data.get("retrieved_docs", []),
        "generation_time": 0.0,  # Will be calculated outside
        "model_info": getattr(response_gen.llm_backend, 'get_model_info', lambda: {})(),
        "context_used": use_rag
    }

def display_response(response: Dict[str, Any], show_context: bool) -> None:
    """Display query response with optional context"""

    # Main response
    st.subheader("📝 Response")
    with st.chat_message("assistant"):
        st.markdown(response["answer"])

    # Metrics in expander
    with st.expander("📊 Performance Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "⏱️ Generation Time",
                f"{response.get('generation_time', 0):.2f}s"
            )

        with col2:
            st.metric(
                "📄 Retrieved Docs",
                len(response.get("retrieved_docs", []))
            )

        with col3:
            model_info = response.get("model_info", {})
            st.metric(
                "🤖 Model",
                model_info.get("model_name", "Unknown")
            )

        with col4:
            st.metric(
                "✅ RAG Enabled",
                "Yes" if response.get("context_used", False) else "No"
            )

    # Retrieved context
    if show_context and response.get("retrieved_docs"):
        st.subheader("📚 Retrieved Context")
        with st.expander("Show retrieved documents", expanded=False):
            for idx, doc in enumerate(response["retrieved_docs"], 1):
                score = doc.get("score", 0.0)
                text = doc.get("text", "")[:500] + "..." if len(doc.get("text", "")) > 500 else doc.get("text", "")

                st.markdown(f"**Document {idx}** (Score: {score:.3f})")
                st.text_area(
                    f"Content of document {idx}",
                    value=text,
                    height=100,
                    disabled=True,
                    key=f"doc_{idx}"
                )
                st.divider()

    # Model info in expander
    with st.expander("🔧 Model Information", expanded=False):
        model_info = response.get("model_info", {})
        if model_info:
            st.json(model_info)

    # Actions
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("📥 Export Response"):
            export_response(response)

    with col2:
        if st.button("🔄 New Query"):
            st.session_state.last_result = None
            st.rerun()

def export_response(response: Dict[str, Any]) -> None:
    """Export query response to markdown"""

    import json
    from datetime import datetime

    # Create export content
    export_content = f"""# Query Response Export

## Query
{response["query"]}

## Response
{response["answer"]}

## Metrics
- Generation Time: {response.get("generation_time", 0):.2f}s
- Model: {response.get("model_info", {}).get("model_name", "Unknown")}
- Retrieved Documents: {len(response.get("retrieved_docs", []))}
- RAG Enabled: {"Yes" if response.get("context_used", False) else "No"}
- Timestamp: {datetime.now().isoformat()}

## Retrieved Documents
"""

    for idx, doc in enumerate(response.get("retrieved_docs", []), 1):
        export_content += f"""
### Document {idx} (Score: {doc.get('score', 0.0):.3f})
{doc.get('text', 'No content available')}
"""

    # Download button
    st.download_button(
        label="📥 Download Response",
        data=export_content,
        file_name=f"query_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )