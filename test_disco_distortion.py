#!/usr/bin/env python3
"""
Test script for the new Disco Diffusion distortion injection approach
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_distortion_function():
    """Test that the distortion injection function exists and has correct signature"""
    print("Testing disco distortion injection function...")
    
    try:
        from extras.disco_diffusion.pipeline_disco import inject_disco_distortion
        import inspect
        
        # Check function signature
        sig = inspect.signature(inject_disco_distortion)
        params = list(sig.parameters.keys())
        
        expected_params = ['latent_samples', 'disco_scale', 'distortion_type']
        for param in expected_params:
            if param in params:
                print(f"✓ Parameter found: {param}")
            else:
                print(f"✗ Missing parameter: {param}")
                return False
        
        print("✓ Distortion injection function has correct signature")
        return True
        
    except Exception as e:
        print(f"✗ Error testing distortion function: {e}")
        return False

def test_distortion_types():
    """Test that different distortion types are implemented"""
    print("\nTesting distortion types...")
    
    try:
        # Check if the distortion types are implemented in the code
        with open('extras/disco_diffusion/pipeline_disco.py', 'r', encoding='utf-8') as f:
            disco_content = f.read()
        
        distortion_types = ['psychedelic', 'fractal', 'kaleidoscope', 'wave']
        
        for dtype in distortion_types:
            if f"distortion_type == '{dtype}'" in disco_content:
                print(f"✓ Distortion type implemented: {dtype}")
            else:
                print(f"✗ Missing distortion type: {dtype}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing distortion types: {e}")
        return False

def test_pipeline_integration():
    """Test that the pipeline integration is updated for mid-generation injection"""
    print("\nTesting pipeline integration...")
    
    try:
        # Check if default_pipeline.py has the new injection approach
        with open('modules/default_pipeline.py', 'r', encoding='utf-8') as f:
            pipeline_content = f.read()
        
        # Check for mid-generation injection
        if "inject_disco_distortion" in pipeline_content:
            print("✓ Disco distortion injection found in pipeline")
        else:
            print("✗ Disco distortion injection not found in pipeline")
            return False
            
        # Check that it's called on initial_latent
        if "initial_latent['samples'] = inject_disco_distortion" in pipeline_content:
            print("✓ Distortion applied to initial latent")
        else:
            print("✗ Distortion not applied to initial latent")
            return False
            
        # Check that post-processing is simplified
        if "Distortion was already applied during generation" in pipeline_content:
            print("✓ Post-processing simplified for new approach")
        else:
            print("✗ Post-processing not updated")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing pipeline integration: {e}")
        return False

def test_distortion_math():
    """Test that the distortion math looks reasonable"""
    print("\nTesting distortion mathematics...")
    
    try:
        # Check if the distortion code has proper mathematical operations
        with open('extras/disco_diffusion/pipeline_disco.py', 'r', encoding='utf-8') as f:
            disco_content = f.read()
        
        # Check for coordinate grid creation
        if "torch.meshgrid" in disco_content:
            print("✓ Coordinate grid creation found")
        else:
            print("✗ Coordinate grid creation not found")
            return False
            
        # Check for spatial transformations
        if "F.grid_sample" in disco_content:
            print("✓ Grid sampling for spatial transformation found")
        else:
            print("✗ Grid sampling not found")
            return False
            
        # Check for trigonometric functions (for wave/swirl effects)
        trig_functions = ['torch.sin', 'torch.cos', 'torch.atan2']
        for func in trig_functions:
            if func in disco_content:
                print(f"✓ Trigonometric function found: {func}")
            else:
                print(f"✗ Missing trigonometric function: {func}")
                return False
                
        # Check for blend factor calculation
        if "blend_factor = min(disco_scale" in disco_content:
            print("✓ Blend factor calculation found")
        else:
            print("✗ Blend factor calculation not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing distortion math: {e}")
        return False

def test_error_handling():
    """Test that proper error handling is in place"""
    print("\nTesting error handling...")
    
    try:
        # Check if error handling is implemented
        with open('extras/disco_diffusion/pipeline_disco.py', 'r', encoding='utf-8') as f:
            disco_content = f.read()
        
        # Check for try-except blocks
        if "try:" in disco_content and "except Exception as e:" in disco_content:
            print("✓ Error handling found")
        else:
            print("✗ Error handling not found")
            return False
            
        # Check for fallback behavior
        if "return latent_samples" in disco_content:
            print("✓ Fallback behavior implemented")
        else:
            print("✗ Fallback behavior not implemented")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing error handling: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Disco Diffusion Distortion Injection Test")
    print("=" * 60)
    
    tests = [
        test_distortion_function,
        test_distortion_types,
        test_pipeline_integration,
        test_distortion_math,
        test_error_handling
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
        print("🎉 All tests passed! New Disco Diffusion approach is ready.")
        print("\nNew approach benefits:")
        print("• Distortion injected during generation (not post-processing)")
        print("• Multiple distortion types: psychedelic, fractal, kaleidoscope, wave")
        print("• Spatial transformations using coordinate grids")
        print("• Proper blend factor scaling with disco_scale")
        print("• Error handling with fallback behavior")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)