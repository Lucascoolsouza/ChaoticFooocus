#!/usr/bin/env python3
"""
Simple test script for Force Grid integration (no PyTorch dependencies)
"""

def test_force_grid_structure():
    """Test that Force Grid files exist and have the expected structure"""
    print("Testing Force Grid file structure...")
    
    import os
    
    files_to_check = [
        "extensions/force_grid.py",
        "extensions/force_grid_pipeline.py", 
        "extensions/force_grid_integration.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    return True

def test_webui_integration():
    """Test that webui.py has force_grid_checkbox in ctrls"""
    print("\nTesting webui.py integration...")
    
    try:
        with open("webui.py", "r") as f:
            content = f.read()
        
        if "force_grid_checkbox" in content:
            print("✓ force_grid_checkbox found in webui.py")
        else:
            print("✗ force_grid_checkbox not found in webui.py")
            return False
        
        if "ctrls = [currentTask, generate_image_grid, force_grid_checkbox]" in content:
            print("✓ force_grid_checkbox properly added to ctrls")
        else:
            print("✗ force_grid_checkbox not properly added to ctrls")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading webui.py: {e}")
        return False

def test_async_worker_integration():
    """Test that async_worker.py has force_grid_checkbox handling"""
    print("\nTesting async_worker.py integration...")
    
    try:
        with open("modules/async_worker.py", "r") as f:
            content = f.read()
        
        if "self.force_grid_checkbox = args.pop()" in content:
            print("✓ force_grid_checkbox properly popped in AsyncTask")
        else:
            print("✗ force_grid_checkbox not properly popped in AsyncTask")
            return False
        
        if "force_grid_checkbox=async_task.force_grid_checkbox" in content:
            print("✓ force_grid_checkbox passed to process_diffusion")
        else:
            print("✗ force_grid_checkbox not passed to process_diffusion")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading async_worker.py: {e}")
        return False

def test_default_pipeline_integration():
    """Test that default_pipeline.py handles force_grid_checkbox"""
    print("\nTesting default_pipeline.py integration...")
    
    try:
        with open("modules/default_pipeline.py", "r") as f:
            content = f.read()
        
        if "force_grid_checkbox=False" in content:
            print("✓ force_grid_checkbox parameter in process_diffusion")
        else:
            print("✗ force_grid_checkbox parameter not in process_diffusion")
            return False
        
        if "if force_grid_checkbox:" in content:
            print("✓ force_grid_checkbox logic in process_diffusion")
        else:
            print("✗ force_grid_checkbox logic not in process_diffusion")
            return False
        
        if "ForceGridContext" in content:
            print("✓ ForceGridContext usage in process_diffusion")
        else:
            print("✗ ForceGridContext not used in process_diffusion")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading default_pipeline.py: {e}")
        return False

def test_force_grid_code_structure():
    """Test the structure of Force Grid code files"""
    print("\nTesting Force Grid code structure...")
    
    try:
        # Test force_grid_pipeline.py
        with open("extensions/force_grid_pipeline.py", "r") as f:
            pipeline_content = f.read()
        
        if "class ForceGridSampler:" in pipeline_content:
            print("✓ ForceGridSampler class found")
        else:
            print("✗ ForceGridSampler class not found")
            return False
        
        if "force_grid_sampler = ForceGridSampler()" in pipeline_content:
            print("✓ Global force_grid_sampler instance found")
        else:
            print("✗ Global force_grid_sampler instance not found")
            return False
        
        # Test force_grid_integration.py
        with open("extensions/force_grid_integration.py", "r") as f:
            integration_content = f.read()
        
        if "class ForceGridInterface:" in integration_content:
            print("✓ ForceGridInterface class found")
        else:
            print("✗ ForceGridInterface class not found")
            return False
        
        if "force_grid = ForceGridInterface()" in integration_content:
            print("✓ Global force_grid instance found")
        else:
            print("✗ Global force_grid instance not found")
            return False
        
        if "class ForceGridContext:" in integration_content:
            print("✓ ForceGridContext class found")
        else:
            print("✗ ForceGridContext class not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading Force Grid files: {e}")
        return False

def main():
    """Run all Force Grid integration tests"""
    print("Force Grid Integration Test (Simple)")
    print("=" * 50)
    
    tests = [
        test_force_grid_structure,
        test_webui_integration,
        test_async_worker_integration,
        test_default_pipeline_integration,
        test_force_grid_code_structure,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All Force Grid integration tests passed!")
        print("\nForce Grid integration appears to be complete!")
        print("\nTo use Force Grid:")
        print("1. Check the 'Generate Grid Image (Experimental)' checkbox in the UI")
        print("2. Generate images as normal")
        print("3. The output will be a single grid image instead of individual images")
        return True
    else:
        print("✗ Some Force Grid integration tests failed!")
        print("\nPlease fix the failing tests before using Force Grid.")
        return False

if __name__ == "__main__":
    main()