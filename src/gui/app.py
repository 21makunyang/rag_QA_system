# """
# Main Streamlit application for CS6493 RAG QA System
# """
#
# import sys
# import logging
# from pathlib import Path
#
# # Add src to path for imports
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#
# import streamlit as st
# from src.gui.config.gui_config import GUIConfig
# from src.gui.utils.session_state import initialize_session_state
# from src.gui.components.document_upload import document_upload_section
# from src.gui.components.query_interface import query_interface_section
# from src.gui.components.model_config import model_config_section, display_backend_status
# from src.gui.components.metrics_display import metrics_display_section
# from src.gui.components.chat_history import chat_history_section
#
# from src import Config
# from src.ingestion.connectors import PDFConnector, TextFileConnector
# from src.ingestion.chunking import ChunkingFactory
# from src.query.retriever import Retriever
#
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)
#
# def setup_page_config(config: GUIConfig):
#     """Setup Streamlit page configuration"""
#
#     st.set_page_config(
#         page_title=config.page_title,
#         page_icon=config.page_icon,
#         layout=config.layout,
#         initial_sidebar_state="expanded"
#     )
#
# def setup_sidebar():
#     """Setup sidebar with system information"""
#
#     st.sidebar.title("🔧 System Info")
#
#     # Display backend status
#     display_backend_status(st.session_state.get("components", {}))
#
#     # System info
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("**System Configuration**")
#     st.sidebar.code({
#         "model": st.session_state.get("current_model", "Not selected"),
#         "documents": st.session_state.get("document_count", 0),
#         "messages": len(st.session_state.get("chat_history", []))
#     })
#
#     # Quick actions
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("**Quick Actions**")
#     if st.sidebar.button("🔄 Reset All", type="secondary"):
#         st.session_state.clear()
#         initialize_session_state()
#         st.rerun()
#
# def launch_gui(components=None, args=None):
#     """
#     Launch the Streamlit GUI application
#
#     Args:
#         components: Pre-initialized components (optional)
#         args: Command line arguments (optional)
#     """
#
#     # Initialize configuration
#     config = GUIConfig()
#
#     # Setup page
#     setup_page_config(config)
#
#     # Initialize session state
#     initialize_session_state()
#
#     # Setup sidebar
#     setup_sidebar()
#
#     # Main title
#     st.title(f"{config.page_icon} {config.page_title}")
#     st.markdown("---")
#
#     # Initialize components if not provided
#     if components is None:
#         components = initialize_components(args)
#
#     # Store components in session state
#     st.session_state.components = components
#
#     # Create tabs for different sections
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "📄 Documents",
#         "💬 Query",
#         "🤖 Model",
#         "📊 Metrics",
#         "💬 History"
#     ])
#
#     # Document upload tab
#     with tab1:
#         document_upload_section(components)
#
#     # Query interface tab
#     with tab2:
#         query_interface_section(components)
#
#     # Model configuration tab
#     with tab3:
#         components = model_config_section(components)
#
#     # Metrics tab
#     with tab4:
#         metrics_display_section(components)
#
#     # Chat history tab
#     with tab5:
#         chat_history_section()
#
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         f"**CS6493 LLM Applications Project** | "
#         f"Documents: {st.session_state.get('document_count', 0)} | "
#         f"Model: {st.session_state.get('current_model', 'Not selected')}"
#     )
#
#     return components
#
# def initialize_components(args=None):
#     """
#     Initialize RAG system components
#
#     Args:
#         args: Command line arguments
#
#     Returns:
#         Dictionary of initialized components
#     """
#
#     try:
#         with st.spinner("Initializing RAG components..."):
#             # Determine model to use
#             if args and hasattr(args, 'model'):
#                 model_name = args.model
#             else:
#                 model_name = st.session_state.get("current_model", "mistral-7b")
#
#             # Initialize connectors
#             pdf_connector = PDFConnector()
#             text_connector = TextFileConnector()
#
#             # Initialize chunking strategy
#             chunking_config = Config.CHUNKING
#             chunking = ChunkingFactory.create_strategy(chunking_config)
#
#             # Initialize retriever
#             retriever = Retriever(Config.VECTOR_STORE)
#
#             # Update document count
#             stats = retriever.get_collection_stats()
#             st.session_state.document_count = stats.get("document_count", 0)
#
#             # Initialize model backend if possible
#             llm_backend = None
#             response_gen = None
#
#             try:
#                 from src.models.ollama_backend import OllamaBackend
#                 from src.models.huggingface_backend import HuggingFaceBackend
#                 from src.query.response_gen import ResponseGenerator
#
#                 model_config = Config.get_model_config(model_name)
#                 if model_config.backend == "ollama":
#                     llm_backend = OllamaBackend(model_config)
#                 elif model_config.backend == "huggingface":
#                     llm_backend = HuggingFaceBackend(model_config)
#
#                 if llm_backend and retriever:
#                     response_gen = ResponseGenerator(llm_backend, retriever)
#
#                 st.session_state.current_model = model_name
#
#             except Exception as e:
#                 logger.warning(f"Could not initialize model backend: {e}")
#                 st.warning("Model backend initialization skipped. Please initialize manually in the Model tab.")
#
#             # Create components dictionary
#             components = {
#                 "llm_backend": llm_backend,
#                 "pdf_connector": pdf_connector,
#                 "text_connector": text_connector,
#                 "chunking": chunking,
#                 "retriever": retriever,
#                 "response_gen": response_gen,
#                 "metrics_calc": None  # Will be initialized when needed
#             }
#
#             # Process documents if specified
#             if args and hasattr(args, 'rechunking') and args.rechunking:
#                 if hasattr(args, 'documents') and Path(args.documents).exists():
#                     process_documents_on_startup(components, args.documents)
#
#             st.success("✅ Components initialized successfully")
#
#             return components
#
#     except Exception as e:
#         st.error(f"❌ Error initializing components: {str(e)}")
#         logger.error(f"Error initializing components: {e}")
#         return {}
#
# def process_documents_on_startup(components, document_dir):
#     """Process documents on application startup"""
#
#     try:
#         st.info(f"Processing documents from {document_dir}...")
#
#         from pathlib import Path
#         from src.ingestion.connectors import PDFConnector, TextFileConnector
#
#         # Process PDFs
#         pdf_files = list(Path(document_dir).glob("*.pdf"))
#         for pdf_file in pdf_files:
#             try:
#                 documents = components["pdf_connector"].load(pdf_file)
#                 chunks = components["chunking"].chunk_documents(documents)
#                 components["retriever"].index_documents(chunks)
#                 st.info(f"Processed: {pdf_file.name}")
#             except Exception as e:
#                 logger.error(f"Error processing {pdf_file.name}: {e}")
#
#         # Process text files
#         text_files = list(Path(document_dir).glob("*.txt"))
#         for text_file in text_files:
#             try:
#                 documents = components["text_connector"].load(text_file)
#                 chunks = components["chunking"].chunk_documents(documents)
#                 components["retriever"].index_documents(chunks)
#                 st.info(f"Processed: {text_file.name}")
#             except Exception as e:
#                 logger.error(f"Error processing {text_file.name}: {e}")
#
#         # Update document count
#         stats = components["retriever"].get_collection_stats()
#         st.session_state.document_count = stats.get("document_count", 0)
#
#         st.success(f"✅ Document processing complete. Indexed {st.session_state.document_count} documents.")
#
#     except Exception as e:
#         logger.error(f"Error during document processing: {e}")
#         st.error(f"❌ Error processing documents: {str(e)}")
#
# def main():
#     """Main entry point for GUI application"""
#
#     # Parse command line arguments if needed
#     import argparse
#     parser = argparse.ArgumentParser(description="CS6493 RAG QA System GUI")
#     parser.add_argument("--model", type=str, default="mistral-7b")
#     parser.add_argument("--documents", type=str, default="./data/documents")
#     parser.add_argument("--rechunking", action="store_true")
#     args = parser.parse_args()
#
#     # Launch GUI
#     launch_gui(args=args)
#
# if __name__ == "__main__":
#     main()