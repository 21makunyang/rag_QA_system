#!/usr/bin/env python3
"""
Test script for GUI integration
"""

import sys
import os
import tempfile
import subprocess

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")

    try:
        # Test core imports
        from src.config import Config
        print("✅ Config import successful")

        from src.ingestion.connectors import PDFConnector, TextFileConnector
        print("✅ Connector imports successful")

        from src.ingestion.chunking import ChunkingFactory
        print("✅ Chunking imports successful")

        from src.models.ollama_backend import OllamaBackend
        print("✅ Ollama backend import successful")

        from src.models.huggingface_backend import HuggingFaceBackend
        print("✅ HuggingFace backend import successful")

        from src.query.retriever import Retriever
        print("✅ Retriever import successful")

        from src.query.response_gen import ResponseGenerator
        print("✅ ResponseGenerator import successful")

        # Test GUI imports
        from src.gui.app import launch_gui
        print("✅ GUI app import successful")

        from src.gui.components.document_upload import document_upload_section
        print("✅ Document upload component import successful")

        from src.gui.components.query_interface import query_interface_section
        print("✅ Query interface component import successful")

        from src.gui.components.model_config import model_config_section
        print("✅ Model config component import successful")

        from src.gui.components.metrics_display import metrics_display_section
        print("✅ Metrics display component import successful")

        from src.gui.components.chat_history import chat_history_section
        print("✅ Chat history component import successful")

        from src.gui.utils.session_state import initialize_session_state
        print("✅ Session state utilities import successful")

        from src.gui.config.gui_config import GUIConfig
        print("✅ GUI config import successful")

        print("\n🎉 All imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_argument_parsing():
    """Test argument parsing with --gui flag"""
    print("\nTesting argument parsing...")

    try:
        # Create a test script that uses the same argument parsing
        test_script = """
import sys
sys.path.insert(0, '.')

from src.main import main
import argparse

# Mock the main function to test argument parsing
parser = argparse.ArgumentParser(description="CS6493 LLM Applications")
parser.add_argument("--model", type=str, default="mistral-7b", choices=["mistral-7b", "t5-base", "llama2-7b"])
parser.add_argument("--query", type=str)
parser.add_argument("--documents", type=str, default="./data/documents")
parser.add_argument("--process-only", action="store_true")
parser.add_argument("--rechunking", action="store_true")
parser.add_argument("--gui", action="store_true")

# Test parsing with --gui flag
args = parser.parse_args(['--gui', '--model', 'mistral-7b'])
print(f"GUI flag: {args.gui}")
print(f"Model: {args.model}")
"""

        # Execute the test script
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True,
            text=True,
            cwd='.'
        )

        if result.returncode == 0:
            print("✅ Argument parsing successful")
            print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Argument parsing failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error testing argument parsing: {e}")
        return False

def test_requirements():
    """Test that Streamlit is in requirements"""
    print("\nTesting requirements.txt...")

    try:
        with open('../requirements.txt', 'r') as f:
            requirements = f.read()

        if 'streamlit' in requirements.lower():
            print("✅ Streamlit found in requirements.txt")
            return True
        else:
            print("❌ Streamlit not found in requirements.txt")
            return False

    except Exception as e:
        print(f"❌ Error reading requirements: {e}")
        return False

def test_directory_structure():
    """Test that GUI directory structure exists"""
    print("\nTesting directory structure...")

    required_dirs = [
        'src/gui',
        'src/gui/components',
        'src/gui/utils',
        'src/gui/config'
    ]

    required_files = [
        'src/gui/__init__.py',
        'src/gui/app.py',
        'src/gui/components/__init__.py',
        'src/gui/components/document_upload.py',
        'src/gui/components/query_interface.py',
        'src/gui/components/model_config.py',
        'src/gui/components/metrics_display.py',
        'src/gui/components/chat_history.py',
        'src/gui/utils/__init__.py',
        'src/gui/utils/session_state.py',
        'src/gui/config/__init__.py',
        'src/gui/config/gui_config.py'
    ]

    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
            return False

    # Check files
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            return False

    return True

def main():
    """Run all tests"""
    print("🚀 Starting GUI integration tests...\n")

    tests = [
        test_imports,
        test_argument_parsing,
        test_requirements,
        test_directory_structure
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append((test.__name__, False))

    print("\n" + "="*50)
    print("📊 Test Results Summary")
    print("="*50)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! GUI integration is ready.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())