#!/usr/bin/env python3
"""
Test script for LFL (Latent Feedback Loop) Neural Echo Sampler integration
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_neural_echo_sampler():
    """Test the Neural Echo Sampler module"""
    print("Testing Neural Echo Sampler module...")
    
    try:
        from modules.neural_echo_sampler import (
            NeuralEchoSampler, 
            initialize_neural_echo, 
            get_neural_echo_sampler,
            reset_neural_echo,
            apply_neural_echo,
            is_neural_echo_enabled,
            setup_neural_echo_for_task
        )
        print("✓ Successfully imported Neural Echo Sampler module")
        
        # Test initialization
        sampler = initialize_neural_echo(echo_strength=0.1, decay_factor=0.8, max_memory=10)
        print(f"✓ Initialized sampler: strength={sampler.echo_strength}, decay={sampler.decay_factor}, memory={sampler.max_memory}")
        
        # Test parameter updates
        sampler.update_parameters(echo_strength=0.05, max_memory=15)
        print(f"✓ Updated parameters: strength={sampler.echo_strength}, memory={sampler.max_memory}")
        
        # Test reset
        sampler.reset()
        print(f"✓ Reset sampler, history length: {len(sampler.history)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Neural Echo Sampler: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_async_task_integration():
    """Test that async task can handle LFL parameters"""
    print("\nTesting async task integration...")
    
    try:
        # Create a mock async task with LFL parameters
        class MockAsyncTask:
            def __init__(self):
                self.lfl_enabled = True
                self.lfl_echo_strength = 0.05
                self.lfl_decay_factor = 0.9
                self.lfl_max_memory = 20
        
        task = MockAsyncTask()
        
        from modules.neural_echo_sampler import is_neural_echo_enabled, setup_neural_echo_for_task
        
        # Test enabled check
        enabled = is_neural_echo_enabled(task)
        print(f"✓ LFL enabled check: {enabled}")
        
        # Test setup
        sampler = setup_neural_echo_for_task(task)
        if sampler:
            print(f"✓ Setup successful: strength={sampler.echo_strength}, decay={sampler.decay_factor}, memory={sampler.max_memory}")
        else:
            print("✗ Setup failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing async task integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_webui_controls():
    """Test that webui controls are properly integrated"""
    print("\nTesting webui controls integration...")
    
    try:
        # Check if webui.py has the LFL controls in ctrls list
        with open('webui.py', 'r', encoding='utf-8') as f:
            webui_content = f.read()
        
        # Check for LFL controls in ctrls list
        if "ctrls += [lfl_enabled, lfl_echo_strength, lfl_decay_factor, lfl_max_memory]" in webui_content:
            print("✓ LFL controls found in webui.py ctrls list")
        else:
            print("✗ LFL controls not found in webui.py ctrls list")
            return False
            
        # Check for LFL UI elements
        lfl_ui_elements = [
            "lfl_enabled = gr.Checkbox",
            "lfl_echo_strength = gr.Slider",
            "lfl_decay_factor = gr.Slider", 
            "lfl_max_memory = gr.Slider"
        ]
        
        for element in lfl_ui_elements:
            if element in webui_content:
                print(f"✓ Found UI element: {element}")
            else:
                print(f"✗ Missing UI element: {element}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Error testing webui controls: {e}")
        return False

def test_async_worker_parameters():
    """Test that async worker handles LFL parameters"""
    print("\nTesting async worker parameter handling...")
    
    try:
        # Check if async_worker.py has LFL parameter extraction
        with open('modules/async_worker.py', 'r', encoding='utf-8') as f:
            async_worker_content = f.read()
        
        # Check for LFL parameter extraction
        lfl_params_check = [
            "self.lfl_enabled",
            "self.lfl_echo_strength", 
            "self.lfl_decay_factor",
            "self.lfl_max_memory"
        ]
        
        for param in lfl_params_check:
            if param in async_worker_content:
                print(f"✓ Found parameter: {param}")
            else:
                print(f"✗ Missing parameter: {param}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Error testing async worker parameters: {e}")
        return False

def test_pipeline_integration():
    """Test that pipeline integration is correct"""
    print("\nTesting pipeline integration...")
    
    try:
        # Check if default_pipeline.py has LFL integration
        with open('modules/default_pipeline.py', 'r', encoding='utf-8') as f:
            pipeline_content = f.read()
        
        # Check for LFL parameters in function signature
        if "lfl_enabled=False, lfl_echo_strength=0.05, lfl_decay_factor=0.9, lfl_max_memory=20" in pipeline_content:
            print("✓ LFL parameters found in process_diffusion signature")
        else:
            print("✗ LFL parameters not found in process_diffusion signature")
            return False
            
        # Check for Neural Echo Sampler initialization
        if "from modules.neural_echo_sampler import initialize_neural_echo" in pipeline_content:
            print("✓ Neural Echo Sampler import found")
        else:
            print("✗ Neural Echo Sampler import not found")
            return False
            
        # Check for callback enhancement
        if "neural_echo_callback" in pipeline_content:
            print("✓ Neural Echo callback integration found")
        else:
            print("✗ Neural Echo callback integration not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing pipeline integration: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("LFL (Latent Feedback Loop) Integration Test")
    print("=" * 60)
    
    tests = [
        test_neural_echo_sampler,
        test_async_task_integration,
        test_webui_controls,
        test_async_worker_parameters,
        test_pipeline_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LFL integration is complete.")
        return True
    else:
        print("❌ Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)