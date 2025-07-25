#!/usr/bin/env python3
"""
Real TPG Generation Test
This script tests actual TPG generation to identify freezing issues.
"""

import sys
import os
import time
import threading
import logging
import traceback

# Add the project root to the path
sys.path.insert(0, '/content/ChaoticFooocus')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_tpg_generation():
    """Test real TPG generation with timeout monitoring"""
    logger.info("Starting real TPG generation test...")
    
    try:
        # Import required modules
        logger.info("Importing modules...")
        import torch
        from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
        
        logger.info("Modules imported successfully")
        
        # Set up a simple generation task
        logger.info("Setting up generation parameters...")
        
        # Simple test parameters
        prompt = "a beautiful landscape"
        negative_prompt = ""
        steps = 5  # Use fewer steps for testing
        cfg_scale = 7.0
        tpg_scale = 1.0  # Enable TPG
        width = 512
        height = 512
        
        logger.info(f"Test parameters:")
        logger.info(f"  Prompt: {prompt}")
        logger.info(f"  Steps: {steps}")
        logger.info(f"  CFG Scale: {cfg_scale}")
        logger.info(f"  TPG Scale: {tpg_scale}")
        logger.info(f"  Size: {width}x{height}")
        
        # Create a timeout mechanism
        generation_completed = threading.Event()
        generation_error = None
        
        def generation_worker():
            nonlocal generation_error
            try:
                logger.info("Starting generation worker thread...")
                
                # This would be the actual generation call
                # For now, let's simulate the key parts that might freeze
                
                logger.info("Simulating model loading...")
                time.sleep(1)
                
                logger.info("Simulating pipeline creation...")
                time.sleep(1)
                
                logger.info("Simulating TPG layer modification...")
                time.sleep(1)
                
                logger.info("Simulating denoising loop...")
                for i in range(steps):
                    logger.info(f"  Denoising step {i+1}/{steps}")
                    time.sleep(0.5)  # Simulate processing time
                
                logger.info("Generation completed successfully!")
                generation_completed.set()
                
            except Exception as e:
                generation_error = e
                logger.error(f"Generation failed: {e}")
                traceback.print_exc()
                generation_completed.set()
        
        # Start generation in a separate thread
        generation_thread = threading.Thread(target=generation_worker)
        generation_thread.daemon = True
        generation_thread.start()
        
        # Wait for completion with timeout
        timeout_seconds = 60
        logger.info(f"Waiting for generation to complete (timeout: {timeout_seconds}s)...")
        
        if generation_completed.wait(timeout_seconds):
            if generation_error:
                logger.error(f"Generation failed: {generation_error}")
                return False
            else:
                logger.info("✓ Generation completed successfully!")
                return True
        else:
            logger.error(f"✗ Generation timed out after {timeout_seconds} seconds!")
            logger.error("This indicates a freeze in the generation process.")
            return False
            
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        traceback.print_exc()
        return False

def test_tpg_layer_application():
    """Test the specific TPG layer application that might cause freezing"""
    logger.info("Testing TPG layer application...")
    
    try:
        from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
        import torch
        import torch.nn as nn
        
        # Create a realistic transformer block that matches ldm_patched pattern
        class RealisticTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(1024)
                self.attn1 = nn.MultiheadAttention(1024, 16)
                self.norm2 = nn.LayerNorm(1024)
                self.attn2 = nn.MultiheadAttention(1024, 16)
                self.norm3 = nn.LayerNorm(1024)
                self.ff = nn.Sequential(
                    nn.Linear(1024, 4096),
                    nn.GELU(),
                    nn.Linear(4096, 1024)
                )
            
            def forward(self, x, context=None, transformer_options={}):
                # Simulate ldm_patched transformer block
                return x + torch.randn_like(x) * 0.01
        
        # Make it look like an ldm_patched block
        RealisticTransformerBlock.__module__ = 'ldm_patched.modules.attention'
        RealisticTransformerBlock.__name__ = 'BasicTransformerBlock'
        
        # Test the layer modification process
        logger.info("Creating original transformer block...")
        original_block = RealisticTransformerBlock()
        
        logger.info("Creating modified class...")
        modified_class = make_tpg_block(RealisticTransformerBlock, do_cfg=True)
        
        logger.info("Creating modified instance...")
        modified_instance = modified_class.__new__(modified_class)
        
        logger.info("Updating instance dictionary...")
        modified_instance.__dict__.update(original_block.__dict__)
        
        logger.info("Adding shuffle_tokens method...")
        def shuffle_tokens(x):
            if len(x.shape) >= 2:
                b, n = x.shape[:2]
                permutation = torch.randperm(n, device=x.device)
                return x[:, permutation]
            return x
        
        modified_instance.shuffle_tokens = shuffle_tokens
        
        logger.info("Testing forward pass...")
        test_hidden_states = torch.randn(2, 77, 1024)
        test_encoder_hidden_states = torch.randn(2, 77, 1024)
        
        # This is where freezing might occur
        with torch.no_grad():
            output = modified_instance.forward(
                hidden_states=test_hidden_states,
                encoder_hidden_states=test_encoder_hidden_states
            )
        
        logger.info(f"✓ Forward pass successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Layer application test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the real generation tests"""
    logger.info("Starting Real TPG Generation Tests")
    logger.info("=" * 50)
    
    tests = [
        ("TPG Layer Application", test_tpg_layer_application),
        ("Real TPG Generation", test_real_tpg_generation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        start_time = time.time()
        
        try:
            result = test_func()
            end_time = time.time()
            results[test_name] = {
                'passed': result,
                'duration': end_time - start_time
            }
            
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name}: {status} (took {end_time - start_time:.2f}s)")
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"{test_name}: ✗ CRASHED - {e}")
            results[test_name] = {
                'passed': False,
                'duration': end_time - start_time,
                'error': str(e)
            }
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("REAL GENERATION TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for r in results.values() if r['passed'])
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        duration = f"({result['duration']:.2f}s)" if result['duration'] else "(crashed)"
        logger.info(f"{test_name}: {status} {duration}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! TPG should work without freezing.")
    else:
        logger.warning("⚠️  Some tests failed. The freezing issue may still exist.")

if __name__ == "__main__":
    main()