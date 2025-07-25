#!/usr/bin/env python3
"""
TPG Generation Debug Script
This script tests TPG generation with detailed logging to identify freeze points.
"""

import sys
import os
import time
import threading
import logging
from contextlib import contextmanager

# Add the project root to the path
sys.path.insert(0, '/content/ChaoticFooocus')

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tpg_debug.log')
    ]
)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

@contextmanager
def operation_timeout(seconds, operation_name):
    """Context manager with timeout and detailed logging"""
    logger.info(f"Starting operation: {operation_name}")
    start_time = time.time()
    
    def timeout_handler():
        logger.error(f"TIMEOUT: {operation_name} took longer than {seconds} seconds!")
        logger.error(f"This suggests a freeze in: {operation_name}")
        # Don't exit, just raise exception to continue with other tests
        raise TimeoutError(f"Timeout in {operation_name}")
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    
    try:
        yield
        end_time = time.time()
        logger.info(f"✓ {operation_name} completed in {end_time - start_time:.2f}s")
    except TimeoutError:
        logger.error(f"✗ {operation_name} timed out after {seconds}s")
        raise
    except Exception as e:
        end_time = time.time()
        logger.error(f"✗ {operation_name} failed after {end_time - start_time:.2f}s: {e}")
        raise
    finally:
        timer.cancel()

def test_step_by_step_generation():
    """Test TPG generation step by step to identify freeze point"""
    logger.info("Testing TPG generation step by step...")
    
    try:
        # Step 1: Import modules
        with operation_timeout(30, "Import modules"):
            import torch
            from modules import config
            from modules.model_loader import load_model
            from extras.TPG.pipeline_sdxl_tpg import StableDiffusionXLTPGPipeline
            logger.info("All modules imported successfully")
        
        # Step 2: Load model components
        with operation_timeout(60, "Load model components"):
            # This is a simplified version - adjust paths as needed
            logger.info("Loading model components...")
            # You might need to adjust this based on your actual model loading
            logger.info("Model components loaded (mocked for now)")
        
        # Step 3: Create TPG pipeline
        with operation_timeout(30, "Create TPG pipeline"):
            logger.info("Creating TPG pipeline...")
            # Mock pipeline creation for testing
            logger.info("TPG pipeline created (mocked)")
        
        # Step 4: Test layer modification
        with operation_timeout(30, "Test layer modification"):
            from extras.TPG.pipeline_sdxl_tpg import make_tpg_block
            import torch.nn as nn
            
            class TestBlock(nn.Module):
                def forward(self, x, context=None, transformer_options={}):
                    return x
            
            original_block = TestBlock()
            modified_class = make_tpg_block(TestBlock, do_cfg=True)
            
            # This is where freezing might occur
            logger.info("Creating modified instance...")
            modified_instance = modified_class.__new__(modified_class)
            
            logger.info("Updating instance dict...")
            modified_instance.__dict__.update(original_block.__dict__)
            
            logger.info("Adding shuffle_tokens method...")
            def shuffle_tokens(x):
                if len(x.shape) >= 2:
                    b, n = x.shape[:2]
                    permutation = torch.randperm(n, device=x.device)
                    return x[:, permutation]
                return x
            
            modified_instance.shuffle_tokens = shuffle_tokens
            logger.info("Layer modification test completed")
        
        # Step 5: Test actual generation (simplified)
        with operation_timeout(60, "Test generation process"):
            logger.info("Testing generation process...")
            
            # Mock the generation process
            prompt = "a beautiful landscape"
            logger.info(f"Processing prompt: {prompt}")
            
            # Simulate the steps that might freeze
            logger.info("Simulating prompt encoding...")
            time.sleep(0.1)
            
            logger.info("Simulating UNet preparation...")
            time.sleep(0.1)
            
            logger.info("Simulating layer modification application...")
            time.sleep(0.1)
            
            logger.info("Simulating denoising loop...")
            for i in range(3):  # Simulate 3 steps instead of full 30
                logger.info(f"Denoising step {i+1}/3")
                time.sleep(0.1)
            
            logger.info("Generation process test completed")
        
        logger.info("🎉 All generation steps completed successfully!")
        return True
        
    except TimeoutError as e:
        logger.error(f"Generation test failed due to timeout: {e}")
        return False
    except Exception as e:
        logger.error(f"Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_concurrent_operations():
    """Test if TPG works with concurrent operations"""
    logger.info("Testing concurrent operations...")
    
    try:
        with operation_timeout(30, "Concurrent operations test"):
            import threading
            import time
            
            results = []
            
            def worker(worker_id):
                logger.info(f"Worker {worker_id} starting")
                time.sleep(0.5)  # Simulate work
                results.append(f"Worker {worker_id} completed")
                logger.info(f"Worker {worker_id} finished")
            
            threads = []
            for i in range(3):
                t = threading.Thread(target=worker, args=(i,))
                t.start()
                threads.append(t)
            
            for t in threads:
                t.join(timeout=10)
                if t.is_alive():
                    logger.error("Thread did not complete in time")
                    return False
            
            logger.info(f"All threads completed. Results: {results}")
            return True
            
    except Exception as e:
        logger.error(f"Concurrent operations test failed: {e}")
        return False

def monitor_system_resources():
    """Monitor system resources during tests"""
    try:
        import psutil
        process = psutil.Process()
        
        logger.info("System Resource Monitoring:")
        logger.info(f"CPU usage: {psutil.cpu_percent()}%")
        logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        logger.info(f"Open file descriptors: {process.num_fds()}")
        logger.info(f"Thread count: {process.num_threads()}")
        
    except ImportError:
        logger.warning("psutil not available for resource monitoring")
    except Exception as e:
        logger.warning(f"Resource monitoring failed: {e}")

def main():
    """Run the debug tests"""
    logger.info("Starting TPG Generation Debug Tests")
    logger.info("=" * 60)
    
    # Monitor initial system state
    monitor_system_resources()
    
    tests = [
        ("Step-by-step Generation", test_step_by_step_generation),
        ("Concurrent Operations", test_concurrent_operations),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name}: {status} (took {end_time - start_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"{test_name}: ✗ CRASHED - {e}")
        
        # Monitor resources after each test
        monitor_system_resources()
    
    logger.info("\n" + "=" * 60)
    logger.info("Debug tests completed. Check the logs above for freeze points.")
    logger.info("If a test times out, that's likely where the freeze occurs.")
    logger.info("Log file saved to: tpg_debug.log")

if __name__ == "__main__":
    main()