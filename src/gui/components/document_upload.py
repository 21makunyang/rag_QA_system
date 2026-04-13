"""
Document upload and processing component
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from src.ingestion.connectors import PDFConnector, TextFileConnector
from src.ingestion.chunking import ChunkingFactory
from src.config import Config

logger = logging.getLogger(__name__)

def document_upload_section(components: Dict[str, Any]) -> None:
    """Document upload and processing UI section"""

    st.header("📄 Document Upload & Processing")

    # Upload configuration
    with st.expander("Upload Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # Chunking strategy selection
            chunking_strategy = st.selectbox(
                "Chunking Strategy",
                options=["fixed", "sentence"],
                index=0,
                help="Choose how documents are split into chunks"
            )

            chunk_size = st.number_input(
                "Chunk Size (tokens)",
                min_value=128,
                max_value=1024,
                value=Config.CHUNKING.chunk_size,
                step=32,
                help="Size of each document chunk in tokens"
            )

            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=100,
                value=Config.CHUNKING.chunk_overlap,
                help="Overlap between chunks (10% of chunk size recommended)"
            )

        with col2:
            # Processing options
            rechunking = st.checkbox(
                "Rechunk all documents",
                value=False,
                help="Clear existing index and reprocess all documents"
            )

            show_progress = st.checkbox(
                "Show detailed progress",
                value=True,
                help="Display detailed processing information"
            )

    # File upload area
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Drag and drop files here or click to browse"
    )

    # Process button
    process_button = st.button(
        "🔄 Process Documents",
        type="primary",
        disabled=len(uploaded_files) == 0
    )

    # Processing logic
    if process_button:
        process_documents(
            uploaded_files=uploaded_files,
            components=components,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rechunking=rechunking,
            show_progress=show_progress
        )

def process_documents(
    uploaded_files,
    components: Dict[str, Any],
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    rechunking: bool,
    show_progress: bool
) -> None:
    """Process uploaded documents"""

    # Clear existing index if rechunking
    if rechunking and components.get("retriever"):
        try:
            with st.spinner("Clearing existing index..."):
                components["retriever"].clear_index()
                st.session_state.document_count = 0
                st.success("Index cleared successfully")
        except Exception as e:
            st.error(f"Error clearing index: {str(e)}")
            logger.error(f"Error clearing index: {e}")
            return

    # Update chunking configuration
    if components.get("chunking"):
        components["chunking"].chunk_size = chunk_size
        components["chunking"].chunk_overlap = chunk_overlap

    # Process files
    total_files = len(uploaded_files)
    processed_files = 0
    errors = []

    # Progress bar
    progress_bar = st.progress(0) if show_progress else None
    status_text = st.empty() if show_progress else None

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            if show_progress:
                progress = (idx + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name} ({idx+1}/{total_files})")

            # Save uploaded file to temp location
            file_extension = Path(uploaded_file.name).suffix.lower()
            temp_file_path = Path(Config.DOCUMENTS_DIR) / uploaded_file.name

            # Ensure directory exists
            temp_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save file
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process based on file type
            if file_extension == ".pdf":
                documents = components["pdf_connector"].load(temp_file_path)
            elif file_extension in [".txt", ".md"]:
                documents = components["text_connector"].load(temp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Chunk documents
            chunks = components["chunking"].chunk_documents(documents)

            # Index documents
            components["retriever"].index_documents(chunks)

            processed_files += 1
            st.success(f"✅ Processed {uploaded_file.name} ({len(chunks)} chunks)")

        except Exception as e:
            error_msg = f"❌ Error processing {uploaded_file.name}: {str(e)}"
            st.error(error_msg)
            errors.append(error_msg)
            logger.error(f"Error processing {uploaded_file.name}: {e}")

    # Final status
    if show_progress:
        progress_bar.empty()
        status_text.empty()

    # Summary
    if processed_files > 0:
        st.success(f"✅ Successfully processed {processed_files} out of {total_files} files")

        # Update document count
        if components.get("retriever"):
            try:
                stats = components["retriever"].get_collection_stats()
                st.session_state.document_count = stats.get("document_count", 0)
            except Exception as e:
                logger.error(f"Error updating document count: {e}")

        # Refresh the interface
        st.session_state.documents_processed = True
        st.rerun()

    if errors:
        st.error(f"❌ {len(errors)} files failed to process")
        for error in errors:
            st.write(error)