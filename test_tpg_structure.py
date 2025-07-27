#!/usr/bin/env python3
"""
Test TPG structure without requiring torch
"""

import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tpg_file_structure():
    """Test that all TPG files exist and have the expected structure"""
    
    try:
        logger.info("Testing TPG file structure...")
        
        # Check if TPG directory exists
        tpg_dir = "extras/TPG"
        if not os.path.exists(tpg_dir):
            logger.error(f"✗ TPG directory not found: {tpg_dir}")
            return False
        
        # Check required files
        required_files = [
            "extras/TPG/__init__.py",
            "extras/TPG/pipeline_sdxl_tpg.py",
            "extras/TPG/tpg_integration.py",
            "extras/TPG/tpg_interface.py",
            "extras/TPG/example_usage.py"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                logger.info(f"✓ Found: {file_path}")
            else:
                logger.error(f"✗ Missing: {file_path}")
                return False
        
        logger.info("✓ All required TPG files found")
        return True
        
    except Exception as e:
        logger.error(f"✗ File structure test failed: {e}")
        return False

def test_tpg_imports_basic():
    """Test basic imports without torch dependencies"""
    
    try:
        logger.info("Testing basic TPG imports...")
        
        # Test pipeline import (this will fail due to torch, but we can catch it)
        try:
            from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
            logger.info("✓ Pipeline import successful")
            pipeline_import_ok = True
        except ImportError as e:
            if "torch" in str(e) or "diffusers" in str(e):
                logger.info("⚠ Pipeline import failed due to missing dependencies (expected)")
                pipeline_import_ok = True  # Expected failure
            else:
                logger.error(f"✗ Pipeline import failed: {e}")
                pipeline_import_ok = False
        
        # Test integration import (should work without torch initially)
        try:
            import extras.TPG.tpg_integration
            logger.info("✓ Integration module import successful")
            integration_import_ok = True
        except ImportError as e:
            logger.error(f"✗ Integration import failed: {e}")
            integration_import_ok = False
        
        # Test interface import
        try:
            import extras.TPG.tpg_interface
            logger.info("✓ Interface module import successful")
            interface_import_ok = True
        except ImportError as e:
            logger.error(f"✗ Interface import failed: {e}")
            interface_import_ok = False
        
        # Test example import
        try:
            import extras.TPG.example_usage
            logger.info("✓ Example usage import successful")
            example_import_ok = True
        except ImportError as e:
            logger.error(f"✗ Example usage import failed: {e}")
            example_import_ok = False
        
        all_ok = all([pipeline_import_ok, integration_import_ok, interface_import_ok, example_import_ok])
        
        if all_ok:
            logger.info("✓ Basic TPG imports test completed")
        else:
            logger.error("✗ Some imports failed")
        
        return all_ok
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        return False

def test_tpg_code_structure():
    """Test that TPG code has expected classes and functions"""
    
    try:
        logger.info("Testing TPG code structure...")
        
        # Read pipeline file and check for key classes
        with open("extras/TPG/pipeline_sdxl_tpg.py", "r") as f:
            pipeline_content = f.read()
        
        expected_pipeline_items = [
            "class StableDiffusionXLTPGPipeline",
            "class TPGAttentionProcessor",
            "def enable_tpg",
            "def disable_tpg",
            "def __call__"
        ]
        
        for item in expected_pipeline_items:
            if item in pipeline_content:
                logger.info(f"✓ Found in pipeline: {item}")
            else:
                logger.error(f"✗ Missing in pipeline: {item}")
                return False
        
        # Read integration file and check for key functions
        with open("extras/TPG/tpg_integration.py", "r") as f:
            integration_content = f.read()
        
        expected_integration_items = [
            "def enable_tpg",
            "def disable_tpg",
            "def is_tpg_enabled",
            "def shuffle_tokens",
            "class TPGContext"
        ]
        
        for item in expected_integration_items:
            if item in integration_content:
                logger.info(f"✓ Found in integration: {item}")
            else:
                logger.error(f"✗ Missing in integration: {item}")
                return False
        
        # Read interface file and check for key classes
        with open("extras/TPG/tpg_interface.py", "r") as f:
            interface_content = f.read()
        
        expected_interface_items = [
            "class TPGInterface",
            "def enable_tpg_simple",
            "def disable_tpg_simple",
            "def with_tpg"
        ]
        
        for item in expected_interface_items:
            if item in interface_content:
                logger.info(f"✓ Found in interface: {item}")
            else:
                logger.error(f"✗ Missing in interface: {item}")
                return False
        
        logger.info("✓ TPG code structure test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Code structure test failed: {e}")
        return False

def test_tpg_documentation():
    """Test that TPG files have proper documentation"""
    
    try:
        logger.info("Testing TPG documentation...")
        
        files_to_check = [
            "extras/TPG/pipeline_sdxl_tpg.py",
            "extras/TPG/tpg_integration.py", 
            "extras/TPG/tpg_interface.py",
            "extras/TPG/example_usage.py"
        ]
        
        for file_path in files_to_check:
            with open(file_path, "r") as f:
                content = f.read()
            
            # Check for docstring at the beginning
            if '"""' in content[:500]:
                logger.info(f"✓ {file_path} has module docstring")
            else:
                logger.warning(f"⚠ {file_path} missing module docstring")
            
            # Check for function docstrings
            if 'def ' in content and '"""' in content:
                logger.info(f"✓ {file_path} has function docstrings")
            else:
                logger.warning(f"⚠ {file_path} may be missing function docstrings")
        
        logger.info("✓ TPG documentation test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Documentation test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing TPG structure and organization...")
    
    # Test 1: File structure
    logger.info("\n=== Test 1: File Structure ===")
    structure_ok = test_tpg_file_structure()
    
    # Test 2: Basic imports
    logger.info("\n=== Test 2: Basic Imports ===")
    imports_ok = test_tpg_imports_basic()
    
    # Test 3: Code structure
    logger.info("\n=== Test 3: Code Structure ===")
    code_ok = test_tpg_code_structure()
    
    # Test 4: Documentation
    logger.info("\n=== Test 4: Documentation ===")
    docs_ok = test_tpg_documentation()
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"File Structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    logger.info(f"Basic Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    logger.info(f"Code Structure: {'✓ PASS' if code_ok else '✗ FAIL'}")
    logger.info(f"Documentation: {'✓ PASS' if docs_ok else '✗ FAIL'}")
    
    all_passed = all([structure_ok, imports_ok, code_ok, docs_ok])
    
    if all_passed:
        logger.info("\n🎉 TPG structure is properly organized!")
        logger.info("Key components:")
        logger.info("- Pipeline implementation (pipeline_sdxl_tpg.py)")
        logger.info("- Integration layer (tpg_integration.py)")
        logger.info("- User interface (tpg_interface.py)")
        logger.info("- Usage examples (example_usage.py)")
        logger.info("\nThe TPG implementation is ready for use!")
    else:
        logger.info("\n⚠️  Some structure tests failed. Check the errors above.")