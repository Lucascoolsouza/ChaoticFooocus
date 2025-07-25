#!/usr/bin/env python3
"""
Layer Modification Debug Script
Focuses specifically on the layer modification process that might cause freezing.
"""

import sys
import os
import time
import threading
import logging
import traceback

# Add the project root to the path
sys.path.insert(0, '/content/ChaoticFooocus')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_layer_modification_detailed():
    """Test each step of layer modification in detail"""
    logger.info("Testing layer modification process in detail...")
    
    try:
        # Import required modules
        logger.info("Step 1: Importing modules...")
        from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
        import torch
        import torch.nn as nn
        logger.info("✓ Modules imported")
        
        # Create a realistic mock transformer block
        logger.info("Step 2: Creating mock transformer block...")
        class MockLDMTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(768)
                self.attn1 = nn.MultiheadAttention(768, 8, batch_first=True)
                self.norm2 = nn.LayerNorm(768)
                self.attn2 = nn.MultiheadAttention(768, 8, batch_first=True)
                self.norm3 = nn.LayerNorm(768)
                self.ff = nn.Sequential(
                    nn.Linear(768, 3072),
                    nn.GELU(),
                    nn.Linear(3072, 768)
                )
                
            def forward(self, x, context=None, transformer_options={}):
                # Simulate ldm_patched BasicTransformerBlock signature
                residual = x
                x = self.norm1(x)
                x, _ = self.attn1(x, x, x)
                x = x + residual
                
                if context is not None:
                    residual = x
                    x = self.norm2(x)
                    x, _ = self.attn2(x, context, context)
                    x = x + residual
                
                residual = x
                x = self.norm3(x)
                x = self.ff(x)
                x = x + residual
                
                return x
        
        original_block = MockLDMTransformerBlock()
        logger.info("✓ Mock transformer block created")
        
        # Test make_tpg_block function
        logger.info("Step 3: Testing make_tpg_block...")
        start_time = time.time()
        modified_class = make_tpg_block(MockLDMTransformerBlock, do_cfg=True)
        end_time = time.time()
        logger.info(f"✓ make_tpg_block completed in {end_time - start_time:.4f}s")
        logger.info(f"Modified class: {modified_class}")
        logger.info(f"Modified class MRO: {modified_class.__mro__}")
        
        # Test instance creation - this is where freezing might occur
        logger.info("Step 4: Testing instance creation...")
        start_time = time.time()
        
        # Method 1: Using __new__ (current approach)
        logger.info("  4a: Testing __new__ approach...")
        modified_instance = modified_class.__new__(modified_class)
        logger.info(f"  ✓ __new__ completed in {time.time() - start_time:.4f}s")
        
        # Test dict update - another potential freeze point
        logger.info("  4b: Testing __dict__ update...")
        start_time = time.time()
        modified_instance.__dict__.update(original_block.__dict__)
        logger.info(f"  ✓ __dict__ update completed in {time.time() - start_time:.4f}s")
        
        # Test method addition
        logger.info("  4c: Testing method addition...")
        start_time = time.time()
        
        def shuffle_tokens(x):
            """Test shuffle tokens method"""
            try:
                if len(x.shape) >= 2:
                    b, n = x.shape[:2]
                    permutation = torch.randperm(n, device=x.device)
                    return x[:, permutation]
                return x
            except Exception as e:
                logger.warning(f"Token shuffling failed: {e}")
                return x
        
        modified_instance.shuffle_tokens = shuffle_tokens
        logger.info(f"  ✓ Method addition completed in {time.time() - start_time:.4f}s")
        
        # Test alternative approach: Direct instantiation
        logger.info("Step 5: Testing alternative instantiation...")
        start_time = time.time()
        try:
            # Try direct instantiation instead of __new__ + __dict__ update
            alternative_instance = modified_class()
            # Copy attributes manually
            for attr_name, attr_value in original_block.__dict__.items():
                if not attr_name.startswith('_'):
                    setattr(alternative_instance, attr_name, attr_value)
            alternative_instance.shuffle_tokens = shuffle_tokens
            logger.info(f"✓ Alternative instantiation completed in {time.time() - start_time:.4f}s")
        except Exception as e:
            logger.error(f"✗ Alternative instantiation failed: {e}")
        
        # Test forward pass
        logger.info("Step 6: Testing forward pass...")
        start_time = time.time()
        
        # Create test input
        test_input = torch.randn(2, 77, 768)  # batch_size=2, seq_len=77, dim=768
        test_context = torch.randn(2, 77, 768)
        
        try:
            with torch.no_grad():
                output = modified_instance.forward(test_input, context=test_context, transformer_options={})
            logger.info(f"✓ Forward pass completed in {time.time() - start_time:.4f}s")
            logger.info(f"Output shape: {output.shape}")
        except Exception as e:
            logger.error(f"✗ Forward pass failed: {e}")
            traceback.print_exc()
        
        # Test multiple instances (potential memory leak check)
        logger.info("Step 7: Testing multiple instances...")
        start_time = time.time()
        instances = []
        for i in range(10):
            instance = modified_class.__new__(modified_class)
            instance.__dict__.update(original_block.__dict__)
            instance.shuffle_tokens = shuffle_tokens
            instances.append(instance)
        logger.info(f"✓ Created 10 instances in {time.time() - start_time:.4f}s")
        
        # Cleanup
        del instances
        logger.info("✓ Cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Layer modification test failed: {e}")
        traceback.print_exc()
        return False

def test_threading_with_layer_modification():
    """Test layer modification in a threaded environment"""
    logger.info("Testing layer modification with threading...")
    
    def worker(worker_id):
        try:
            logger.info(f"Worker {worker_id}: Starting layer modification")
            result = test_layer_modification_detailed()
            logger.info(f"Worker {worker_id}: {'Success' if result else 'Failed'}")
            return result
        except Exception as e:
            logger.error(f"Worker {worker_id}: Exception - {e}")
            return False
    
    # Test with multiple threads
    threads = []
    results = []
    
    def thread_wrapper(worker_id):
        result = worker(worker_id)
        results.append(result)
    
    for i in range(3):
        t = threading.Thread(target=thread_wrapper, args=(i,))
        t.start()
        threads.append(t)
    
    # Wait for all threads with timeout
    for t in threads:
        t.join(timeout=60)
        if t.is_alive():
            logger.error("Thread did not complete in time - possible freeze")
            return False
    
    success_count = sum(results)
    logger.info(f"Threading test: {success_count}/{len(results)} workers succeeded")
    return success_count == len(results)

def main():
    """Run all layer modification tests"""
    logger.info("Starting Layer Modification Debug Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Detailed Layer Modification", test_layer_modification_detailed),
        ("Threading with Layer Modification", test_threading_with_layer_modification),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        start_time = time.time()
        
        try:
            result = test_func()
            end_time = time.time()
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name}: {status} (took {end_time - start_time:.2f}s)")
        except Exception as e:
            end_time = time.time()
            logger.error(f"{test_name}: ✗ CRASHED after {end_time - start_time:.2f}s - {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("Layer modification tests completed.")
    logger.info("If any test hangs or takes too long, that's where the freeze occurs.")

if __name__ == "__main__":
    main()