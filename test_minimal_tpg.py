#!/usr/bin/env python3
"""
Minimal test to isolate TPG freezing issue
"""

import torch
import logging
import sys
import time
import signal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_with_timeout(func, timeout_seconds=30):
    """Run a function with a timeout"""
    
    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        logger.error(f"Function timed out after {timeout_seconds} seconds")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        raise e

def test_import():
    """Test importing the TPG pipeline"""
    logger.info("Testing TPG pipeline import...")
    
    try:
        # Test individual imports first
        logger.info("Importing diffusers components...")
        from diffusers.models.attention import BasicTransformerBlock
        logger.info("✓ BasicTransformerBlock imported")
        
        from diffusers import DiffusionPipeline
        logger.info("✓ DiffusionPipeline imported")
        
        logger.info("Importing TPG pipeline...")
        from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
        logger.info("✓ Pipeline imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_properties():
    """Test basic pipeline properties"""
    logger.info("Testing basic pipeline properties...")
    
    try:
        from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
        
        # Test class properties
        logger.info(f"✓ Pipeline class: {StableDiffusionXLTPGPipeline}")
        logger.info(f"✓ Pipeline MRO: {[cls.__name__ for cls in StableDiffusionXLTPGPipeline.__mro__]}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Properties test failed: {e}")
        return False

def test_make_tpg_block():
    """Test the make_tpg_block function"""
    logger.info("Testing make_tpg_block function...")
    
    try:
        from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
        from diffusers.models.attention import BasicTransformerBlock
        
        # Test creating a TPG block
        modified_class = make_tpg_block(BasicTransformerBlock, do_cfg=True)
        logger.info(f"✓ make_tpg_block created class: {modified_class}")
        
        # Test instantiation with required arguments
        # BasicTransformerBlock requires dim, num_attention_heads, attention_head_dim
        instance = modified_class(
            dim=768,
            num_attention_heads=12,
            attention_head_dim=64
        )
        logger.info(f"✓ TPG block instance created: {type(instance)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ make_tpg_block test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests with timeouts"""
    
    tests = [
        ("Import Test", test_import),
        ("Basic Properties Test", test_basic_properties),
        ("make_tpg_block Test", test_make_tpg_block),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n=== {test_name} ===")
        
        try:
            # Use longer timeout for import test
            timeout = 30 if "Import" in test_name else 10
            result = test_with_timeout(test_func, timeout_seconds=timeout)
            results[test_name] = result is not False
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n=== TEST SUMMARY ===")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    return results

if __name__ == "__main__":
    logger.info("Starting minimal TPG tests with timeout protection...")
    
    try:
        results = run_all_tests()
        
        if all(results.values()):
            logger.info("\n🎉 All tests passed!")
        else:
            logger.info("\n⚠️  Some tests failed. Check the logs above.")
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Tests interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Tests completed")