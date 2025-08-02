#!/usr/bin/env python3
"""
Test script for Neural Echo Sampler integration
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_neural_echo_sampler_in_list():
    """Test that neural_echo is in the KSAMPLER_NAMES list"""
    print("Testing neural_echo sampler in KSAMPLER_NAMES...")
    
    try:
        from ldm_patched.modules.samplers import KSAMPLER_NAMES
        
        if "neural_echo" in KSAMPLER_NAMES:
            print("✓ neural_echo found in KSAMPLER_NAMES")
            return True
        else:
            print("✗ neural_echo not found in KSAMPLER_NAMES")
            return False
            
    except Exception as e:
        print(f"✗ Error checking KSAMPLER_NAMES: {e}")
        return False

def test_neural_echo_sampler_implementation():
    """Test that neural_echo sampler is implemented in ksampler function"""
    print("\nTesting neural_echo sampler implementation...")
    
    try:
        # Check if the implementation exists in the code
        with open('ldm_patched/modules/samplers.py', 'r', encoding='utf-8') as f:
            samplers_content = f.read()
        
        if 'elif sampler_name == "neural_echo":' in samplers_content:
            print("✓ neural_echo sampler implementation found")
        else:
            print("✗ neural_echo sampler implementation not found")
            return False
            
        if "neural_echo_function" in samplers_content:
            print("✓ neural_echo_function defined")
        else:
            print("✗ neural_echo_function not defined")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error checking sampler implementation: {e}")
        return False

def test_neural_echo_module():
    """Test that the neural echo sampler module is properly structured"""
    print("\nTesting neural echo sampler module...")
    
    try:
        # Check if the module file exists and has the right structure
        with open('modules/neural_echo_sampler.py', 'r', encoding='utf-8') as f:
            module_content = f.read()
        
        # Check for key components
        components = [
            "class NeuralEchoSampler:",
            "def compute_echo(self)",
            "def __call__(self, x: torch.Tensor, denoised: torch.Tensor)",
            "def initialize_neural_echo",
            "def get_neural_echo_sampler",
            "def apply_neural_echo"
        ]
        
        for component in components:
            if component in module_content:
                print(f"✓ Found: {component}")
            else:
                print(f"✗ Missing: {component}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Error checking neural echo module: {e}")
        return False

def test_webui_integration():
    """Test that webui has the LFL controls properly integrated"""
    print("\nTesting webui LFL controls...")
    
    try:
        # Check if webui.py has LFL controls
        with open('webui.py', 'r', encoding='utf-8') as f:
            webui_content = f.read()
        
        lfl_controls = [
            "lfl_enabled = gr.Checkbox",
            "lfl_echo_strength = gr.Slider",
            "lfl_decay_factor = gr.Slider",
            "lfl_max_memory = gr.Slider"
        ]
        
        for control in lfl_controls:
            if control in webui_content:
                print(f"✓ Found LFL control: {control}")
            else:
                print(f"✗ Missing LFL control: {control}")
                return False
        
        # Check if controls are in ctrls list
        if "ctrls += [lfl_enabled, lfl_echo_strength, lfl_decay_factor, lfl_max_memory]" in webui_content:
            print("✓ LFL controls added to ctrls list")
        else:
            print("✗ LFL controls not in ctrls list")
            return False
                
        return True
        
    except Exception as e:
        print(f"✗ Error checking webui integration: {e}")
        return False

def test_async_worker_integration():
    """Test that async worker handles LFL parameters"""
    print("\nTesting async worker LFL integration...")
    
    try:
        # Check if async_worker.py has LFL parameter handling
        with open('modules/async_worker.py', 'r', encoding='utf-8') as f:
            async_worker_content = f.read()
        
        lfl_params = [
            "self.lfl_enabled",
            "self.lfl_echo_strength",
            "self.lfl_decay_factor", 
            "self.lfl_max_memory"
        ]
        
        for param in lfl_params:
            if param in async_worker_content:
                print(f"✓ Found LFL parameter: {param}")
            else:
                print(f"✗ Missing LFL parameter: {param}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Error checking async worker integration: {e}")
        return False

def test_pipeline_integration():
    """Test that pipeline integration includes LFL parameters"""
    print("\nTesting pipeline LFL integration...")
    
    try:
        # Check if default_pipeline.py has LFL integration
        with open('modules/default_pipeline.py', 'r', encoding='utf-8') as f:
            pipeline_content = f.read()
        
        # Check for LFL parameters in function signature
        if "lfl_enabled=False" in pipeline_content:
            print("✓ LFL parameters in process_diffusion signature")
        else:
            print("✗ LFL parameters not in process_diffusion signature")
            return False
            
        # Check for Neural Echo Sampler initialization
        if "neural_echo_sampler" in pipeline_content:
            print("✓ Neural Echo Sampler integration found")
        else:
            print("✗ Neural Echo Sampler integration not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error checking pipeline integration: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Neural Echo Sampler Integration Test")
    print("=" * 60)
    
    tests = [
        test_neural_echo_sampler_in_list,
        test_neural_echo_sampler_implementation,
        test_neural_echo_module,
        test_webui_integration,
        test_async_worker_integration,
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
        print("🎉 All tests passed! Neural Echo Sampler is fully integrated.")
        print("\nIntegration includes:")
        print("• Neural Echo Sampler added to KSAMPLER_NAMES")
        print("• Custom sampler implementation with echo callback")
        print("• LFL controls in webui (enabled, strength, decay, memory)")
        print("• Parameter handling in async worker")
        print("• Pipeline integration for neural echo functionality")
        return True
    else:
        print("❌ Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)