#!/usr/bin/env python3
"""
Debug TPG Generation Freeze
This script adds detailed logging to identify exactly where TPG generation freezes.
"""

import sys
import os
import time
import threading
import logging

# Add the project root to the path
sys.path.insert(0, '/content/ChaoticFooocus')

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_debug_logging_to_tpg():
    """Add debug logging to the TPG pipeline to track where it freezes"""
    logger.info("Adding debug logging to TPG pipeline...")
    
    try:
        # Import and patch the TPG pipeline with debug logging
        from extras.TPG import pipeline_sdxl_tpg
        
        # Store original methods
        original_call = pipeline_sdxl_tpg.StableDiffusionXLTPGPipeline.__call__
        
        def debug_call(self, *args, **kwargs):
            logger.info("=== TPG PIPELINE CALL STARTED ===")
            logger.info(f"Args: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
            
            try:
                # Add timeout monitoring
                start_time = time.time()
                
                def monitor_progress():
                    while True:
                        elapsed = time.time() - start_time
                        logger.info(f"TPG generation running for {elapsed:.1f}s...")
                        if elapsed > 300:  # 5 minutes timeout
                            logger.error("TPG generation has been running for over 5 minutes - likely frozen!")
                            break
                        time.sleep(10)  # Check every 10 seconds
                
                monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
                monitor_thread.start()
                
                result = original_call(self, *args, **kwargs)
                logger.info("=== TPG PIPELINE CALL COMPLETED ===")
                return result
                
            except Exception as e:
                logger.error(f"TPG pipeline call failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Patch the method
        pipeline_sdxl_tpg.StableDiffusionXLTPGPipeline.__call__ = debug_call
        logger.info("✓ Debug logging added to TPG pipeline")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to add debug logging: {e}")
        return False

def add_unet_debug_logging():
    """Add debug logging to UNet calls to see where it freezes"""
    logger.info("Adding debug logging to UNet calls...")
    
    try:
        from extras.TPG import pipeline_sdxl_tpg
        
        # Find the UNet call in the denoising loop
        original_file_content = None
        tpg_file_path = '/content/ChaoticFooocus/extras/TPG/pipeline_sdxl_tpg.py'
        
        # Read the current file
        with open(tpg_file_path, 'r') as f:
            content = f.read()
        
        # Look for the UNet call pattern and add logging around it
        if 'noise_pred = self.unet(' in content:
            logger.info("Found UNet call in TPG pipeline")
            
            # Add logging before and after UNet call
            modified_content = content.replace(
                'noise_pred = self.unet(',
                '''logger.info(f"About to call UNet at step {i+1}")
                sys.stdout.flush()
                noise_pred = self.unet('''
            )
            
            # Also add logging after the UNet call
            modified_content = modified_content.replace(
                'logger.info(f"UNet call completed successfully at step {i+1}',
                '''logger.info(f"UNet call returned, processing result...")
                sys.stdout.flush()
                logger.info(f"UNet call completed successfully at step {i+1}'''
            )
            
            # Write back the modified content
            with open(tpg_file_path, 'w') as f:
                f.write(modified_content)
            
            logger.info("✓ Added UNet debug logging")
            return True
        else:
            logger.warning("Could not find UNet call pattern in TPG pipeline")
            return False
            
    except Exception as e:
        logger.error(f"Failed to add UNet debug logging: {e}")
        return False

def create_freeze_detector():
    """Create a script that can detect when TPG generation freezes"""
    
    freeze_detector_script = '''#!/usr/bin/env python3
"""
TPG Freeze Detector - Run this alongside TPG generation to detect freezes
"""

import time
import psutil
import logging
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_process():
    """Monitor the ChaoticFooocus process for signs of freezing"""
    logger.info("Starting TPG freeze monitoring...")
    
    # Find the main process
    main_process = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and any('webui.py' in cmd for cmd in proc.info['cmdline']):
                main_process = proc
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not main_process:
        logger.error("Could not find ChaoticFooocus main process")
        return
    
    logger.info(f"Monitoring process {main_process.pid}")
    
    last_cpu_time = main_process.cpu_times()
    freeze_count = 0
    
    while True:
        try:
            time.sleep(5)  # Check every 5 seconds
            
            current_cpu_time = main_process.cpu_times()
            cpu_usage = main_process.cpu_percent()
            memory_info = main_process.memory_info()
            
            # Check if CPU time is advancing (process is doing work)
            cpu_time_diff = current_cpu_time.user - last_cpu_time.user
            
            logger.info(f"CPU: {cpu_usage:.1f}%, Memory: {memory_info.rss/1024/1024:.1f}MB, CPU time diff: {cpu_time_diff:.3f}")
            
            if cpu_time_diff < 0.01 and cpu_usage < 1.0:
                freeze_count += 1
                logger.warning(f"Possible freeze detected (count: {freeze_count})")
                
                if freeze_count >= 6:  # 30 seconds of no activity
                    logger.error("FREEZE DETECTED: Process appears to be frozen!")
                    logger.error("Check the main process logs for the last operation before freeze")
                    break
            else:
                freeze_count = 0  # Reset counter if activity detected
            
            last_cpu_time = current_cpu_time
            
        except psutil.NoSuchProcess:
            logger.info("Process ended")
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            break

if __name__ == "__main__":
    monitor_process()
'''
    
    with open('tpg_freeze_detector.py', 'w') as f:
        f.write(freeze_detector_script)
    
    logger.info("✓ Created tpg_freeze_detector.py")
    logger.info("Run this in a separate terminal: python tpg_freeze_detector.py")

def main():
    """Set up debugging for TPG generation freeze"""
    logger.info("Setting up TPG Generation Freeze Debugging")
    logger.info("=" * 60)
    
    # Add debug logging
    if add_debug_logging_to_tpg():
        logger.info("✓ TPG pipeline debug logging enabled")
    
    if add_unet_debug_logging():
        logger.info("✓ UNet debug logging enabled")
    
    # Create freeze detector
    create_freeze_detector()
    
    logger.info("\n" + "=" * 60)
    logger.info("DEBUG SETUP COMPLETE")
    logger.info("=" * 60)
    logger.info("Now run your TPG generation and watch for:")
    logger.info("1. Where exactly the freeze occurs")
    logger.info("2. The last logged operation before freeze")
    logger.info("3. Run 'python tpg_freeze_detector.py' in another terminal")
    logger.info("4. Check if it freezes at UNet call or layer modification")

if __name__ == "__main__":
    main()