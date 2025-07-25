#!/usr/bin/env python3
"""
TPG Freeze Diagnostic Script
This script helps identify potential freezing issues in the TPG pipeline.
"""

import sys
import os
import time
import threading
import signal
import traceback
import logging
from contextlib import contextmanager

# Add the project root to the path
sys.path.insert(0, '/content/ChaoticFooocus')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout detection"""
    def timeout_handler():
        logger.error(f"TIMEOUT: Operation took longer than {seconds} seconds!")
        traceback.print_stack()
        os._exit(1)
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()

def test_tpg_imports():
    """Test if TPG imports work without freezing"""
    logger.info("Testing TPG imports...")
    
    with timeout_context(30):
        try:
            from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline, make_tpg_block
            logger.info("✓ TPG imports successful")
            return True
        except Exception as e:
            logger.error(f"✗ TPG import failed: {e}")
            return False

def test_make_tpg_block():
    """Test the make_tpg_block function"""
    logger.info("Testing make_tpg_block function...")
    
    with timeout_context(30):
        try:
            from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
            import torch.nn as nn
            
            # Create a dummy block class
            class DummyBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 10)
                
                def forward(self, x):
                    return self.linear(x)
            
            # Test make_tpg_block
            modified_class = make_tpg_block(DummyBlock, do_cfg=True)
            logger.info(f"✓ make_tpg_block returned: {modified_class}")
            
            # Test instantiation
            instance = modified_class()
            logger.info("✓ Modified class instantiation successful")
            
            return True
        except Exception as e:
            logger.error(f"✗ make_tpg_block test failed: {e}")
            traceback.print_exc()
            return False

def test_pipeline_creation():
    """Test TPG pipeline creation"""
    logger.info("Testing TPG pipeline creation...")
    
    with timeout_context(60):
        try:
            from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
            import torch
            
            # Mock components for testing
            class MockUNet:
                def __init__(self):
                    self.config = type('Config', (), {'in_channels': 4})()
                    self.dtype = torch.float32
                    self.device = torch.device('cpu')
                
                def __call__(self, *args, **kwargs):
                    return torch.randn(1, 4, 64, 64)
            
            class MockVAE:
                def __init__(self):
                    self.config = type('Config', (), {'latent_channels': 4})()
                    self.dtype = torch.float32
                    self.device = torch.device('cpu')
            
            class MockScheduler:
                def __init__(self):
                    self.config = type('Config', (), {})()
            
            class MockTokenizer:
                def __init__(self):
                    pass
            
            class MockTextEncoder:
                def __init__(self):
                    self.dtype = torch.float32
                    self.device = torch.device('cpu')
            
            # Try to create pipeline with mock components
            pipeline = StableDiffusionXLTPGPipeline(
                vae=MockVAE(),
                text_encoder=MockTextEncoder(),
                text_encoder_2=MockTextEncoder(),
                tokenizer=MockTokenizer(),
                tokenizer_2=MockTokenizer(),
                unet=MockUNet(),
                scheduler=MockScheduler()
            )
            
            logger.info("✓ TPG pipeline creation successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ TPG pipeline creation failed: {e}")
            traceback.print_exc()
            return False

def test_layer_modification():
    """Test the layer modification process that might cause freezing"""
    logger.info("Testing layer modification process...")
    
    with timeout_context(30):
        try:
            from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
            import torch.nn as nn
            import torch
            
            # Create a more realistic transformer block
            class MockTransformerBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(768)
                    self.attn = nn.MultiheadAttention(768, 8)
                    self.norm2 = nn.LayerNorm(768)
                    self.mlp = nn.Sequential(
                        nn.Linear(768, 3072),
                        nn.GELU(),
                        nn.Linear(3072, 768)
                    )
                
                def forward(self, x, context=None, transformer_options={}):
                    # Simulate ldm_patched signature
                    return x + torch.randn_like(x) * 0.1
            
            # Test the modification process
            original_block = MockTransformerBlock()
            logger.info("✓ Created mock transformer block")
            
            # Test make_tpg_block
            modified_class = make_tpg_block(MockTransformerBlock, do_cfg=True)
            logger.info("✓ make_tpg_block completed")
            
            # Test instance creation (this is where freezing might occur)
            modified_instance = modified_class.__new__(modified_class)
            logger.info("✓ Modified instance created with __new__")
            
            # Test dict update (another potential freeze point)
            modified_instance.__dict__.update(original_block.__dict__)
            logger.info("✓ Dict update completed")
            
            # Test shuffle_tokens method addition
            def dummy_shuffle(x):
                return x
            modified_instance.shuffle_tokens = dummy_shuffle
            logger.info("✓ shuffle_tokens method added")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Layer modification test failed: {e}")
            traceback.print_exc()
            return False

def test_threading_issues():
    """Test for potential threading issues"""
    logger.info("Testing for threading issues...")
    
    with timeout_context(30):
        try:
            import threading
            import time
            
            # Test if we can create threads without issues
            def worker():
                time.sleep(0.1)
                logger.info("Thread worker completed")
            
            threads = []
            for i in range(5):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)
            
            for t in threads:
                t.join(timeout=5)
                if t.is_alive():
                    logger.error("Thread did not complete in time")
                    return False
            
            logger.info("✓ Threading test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Threading test failed: {e}")
            return False

def test_memory_usage():
    """Test for memory-related issues"""
    logger.info("Testing memory usage...")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        after_gc_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory after GC: {after_gc_memory:.2f} MB")
        
        return True
        
    except ImportError:
        logger.warning("psutil not available, skipping memory test")
        return True
    except Exception as e:
        logger.error(f"Memory test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    logger.info("Starting TPG Freeze Diagnostic Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_tpg_imports),
        ("make_tpg_block Test", test_make_tpg_block),
        ("Pipeline Creation Test", test_pipeline_creation),
        ("Layer Modification Test", test_layer_modification),
        ("Threading Test", test_threading_issues),
        ("Memory Test", test_memory_usage),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            results[test_name] = {
                'passed': result,
                'duration': end_time - start_time
            }
            logger.info(f"{test_name} completed in {end_time - start_time:.2f}s")
        except Exception as e:
            logger.error(f"{test_name} crashed: {e}")
            results[test_name] = {
                'passed': False,
                'duration': None,
                'error': str(e)
            }
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        duration = f"({result['duration']:.2f}s)" if result['duration'] else "(crashed)"
        logger.info(f"{test_name}: {status} {duration}")
        if result['passed']:
            passed += 1
        elif 'error' in result:
            logger.info(f"  Error: {result['error']}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! TPG should work without freezing.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        logger.info("\nTroubleshooting suggestions:")
        logger.info("1. Check if the issue is in layer modification")
        logger.info("2. Look for threading conflicts")
        logger.info("3. Monitor memory usage during generation")
        logger.info("4. Add more logging to the TPG pipeline")

if __name__ == "__main__":
    main()