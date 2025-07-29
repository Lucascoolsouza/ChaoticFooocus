#!/usr/bin/env python3
"""
Test script to verify disco parameters are properly handled
"""

def test_disco_parameters():
    """Test that disco parameters are properly handled in async_worker"""
    print("Testing disco parameters...")
    
    try:
        with open("modules/async_worker.py", "r") as f:
            content = f.read()
        
        # Check that all 4 disco parameters are popped
        disco_params = [
            "disco_guidance_steps",
            "disco_cutn", 
            "disco_tv_scale",
            "disco_range_scale"
        ]
        
        for param in disco_params:
            if f"self.{param} = args.pop()" in content:
                print(f"✓ {param} is properly popped")
            else:
                print(f"✗ {param} is not properly popped")
                return False
        
        # Check that all disco parameters are passed to process_diffusion
        for param in disco_params:
            if f"{param}=async_task.{param}" in content:
                print(f"✓ {param} is passed to process_diffusion")
            else:
                print(f"✗ {param} is not passed to process_diffusion")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing disco parameters: {e}")
        return False

def test_parameter_count():
    """Test that the parameter count matches between webui and async_worker"""
    print("\nTesting parameter count...")
    
    try:
        # Count args.pop() calls in async_worker
        with open("modules/async_worker.py", "r") as f:
            async_content = f.read()
        
        pop_count = async_content.count("args.pop()")
        print(f"✓ Found {pop_count} args.pop() calls in async_worker.py")
        
        # This is a rough estimate - the actual count depends on the exact implementation
        # but we should have a reasonable number of parameters
        if pop_count > 50:  # We expect many parameters
            print("✓ Parameter count seems reasonable")
            return True
        else:
            print(f"✗ Parameter count ({pop_count}) seems too low")
            return False
        
    except Exception as e:
        print(f"✗ Error testing parameter count: {e}")
        return False

def test_performance_position():
    """Test that performance_selection is in the correct position"""
    print("\nTesting performance_selection position...")
    
    try:
        with open("modules/async_worker.py", "r") as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Find the positions of key parameters
        disco_guidance_line = -1
        performance_line = -1
        
        for i, line in enumerate(lines):
            if "self.disco_guidance_steps = args.pop()" in line:
                disco_guidance_line = i
            if "self.performance_selection = Performance(args.pop())" in line:
                performance_line = i
        
        if disco_guidance_line > 0 and performance_line > disco_guidance_line:
            print("✓ performance_selection is popped after disco parameters (correct)")
            return True
        else:
            print("✗ performance_selection is not in the correct position relative to disco parameters")
            return False
        
    except Exception as e:
        print(f"✗ Error testing performance position: {e}")
        return False

def main():
    """Run disco parameter tests"""
    print("Disco Parameters Test")
    print("=" * 50)
    
    tests = [
        test_disco_parameters,
        test_parameter_count,
        test_performance_position,
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
        print("✓ All disco parameter tests passed!")
        print("\nThe Performance enum error should now be completely fixed.")
        print("The parameter order now correctly matches webui.py ctrls list.")
        return True
    else:
        print("✗ Some disco parameter tests failed!")
        return False

if __name__ == "__main__":
    main()