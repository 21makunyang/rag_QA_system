"""
Model configuration component
"""

import streamlit as st
from typing import Dict, Any
import logging

from src.models.ollama_backend import OllamaBackend
from src.models.huggingface_backend import HuggingFaceBackend
from src.config import Config

logger = logging.getLogger(__name__)

def model_config_section(components: Dict[str, Any]) -> Dict[str, Any]:
    """Model configuration section"""

    st.header("🤖 Model Configuration")

    # Model selection
    col1, col2 = st.columns(2)

    with col1:
        available_models = list(Config.MODEL_CONFIGS.keys())
        current_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=available_models.index(st.session_state.current_model),
            help="Choose the LLM backend to use"
        )

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses. Lower values are more deterministic."
        )

        # Max tokens
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=64,
            max_value=2048,
            value=512,
            step=64,
            help="Maximum number of tokens to generate in response"
        )

    with col2:
        # Backend info
        backend_info = st.empty()

        # Test connection button
        test_connection = st.button(
            "🔌 Test Connection",
            type="secondary",
            use_container_width=True
        )

    # Initialize/Update backend
    if st.button("🔄 Initialize/Update Model", type="primary"):
        try:
            with st.spinner("Initializing model backend..."):
                components = initialize_model_backend(
                    components=components,
                    model_name=current_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Update session state
                st.session_state.current_model = current_model

                st.success(f"✅ Model {current_model} initialized successfully")

        except Exception as e:
            st.error(f"❌ Error initializing model: {str(e)}")
            logger.error(f"Error initializing model: {e}")

    # Display backend info
    if components.get("llm_backend"):
        backend_info.info("Model backend initialized")

        # Model info display
        with st.expander("Model Details", expanded=False):
            model_info = components["llm_backend"].get_model_info()
            st.json(model_info)

    # Test connection
    if test_connection:
        test_model_connection(components)

    return components

def initialize_model_backend(
    components: Dict[str, Any],
    model_name: str,
    temperature: float,
    max_tokens: int
) -> Dict[str, Any]:
    """Initialize or update model backend"""

    # Get model configuration
    model_config = Config.get_model_config(model_name)
    model_config.temperature = temperature
    model_config.max_tokens = max_tokens

    # Initialize LLM backend
    if model_config.backend == "ollama":
        components["llm_backend"] = OllamaBackend(model_config)
    elif model_config.backend == "huggingface":
        components["llm_backend"] = HuggingFaceBackend(model_config)
    else:
        raise ValueError(f"Unsupported backend: {model_config.backend}")

    # Initialize response generator if retriever exists
    if components.get("retriever") and components.get("llm_backend"):
        from src.query.response_gen import ResponseGenerator
        components["response_gen"] = ResponseGenerator(
            components["llm_backend"],
            components["retriever"]
        )

    logger.info(f"Initialized model backend: {model_name}")
    return components

def test_model_connection(components: Dict[str, Any]) -> None:
    """Test connection to model backend"""

    if not components.get("llm_backend"):
        st.warning("⚠️ No model backend initialized")
        return

    with st.spinner("Testing model connection..."):
        try:
            # Simple test prompt
            test_prompt = "Hello, this is a test message. Please respond briefly."
            response = components["llm_backend"].generate(test_prompt)

            if response and len(response.strip()) > 0:
                st.success("✅ Model connection successful")
                st.write(f"Model response: *{response.strip()}*")
            else:
                st.warning("⚠️ Model returned empty response")

        except Exception as e:
            st.error(f"❌ Model connection failed: {str(e)}")
            logger.error(f"Model connection test failed: {e}")

def display_backend_status(components: Dict[str, Any]) -> None:
    """Display backend status in sidebar"""

    if not components.get("llm_backend"):
        st.sidebar.warning("Backend: Not initialized")
        return

    try:
        model_info = components["llm_backend"].get_model_info()
        backend_status = model_info.get("backend", "Unknown")
        model_name = model_info.get("model_name", "Unknown")

        st.sidebar.success(f"Backend: {backend_status}")
        st.sidebar.info(f"Model: {model_name}")

    except Exception as e:
        st.sidebar.error(f"Backend Error: {str(e)}")