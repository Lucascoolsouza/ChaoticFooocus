#!/usr/bin/env python3
"""
Test script to verify parameter order between webui.py and async_worker.py
"""

def test_parameter_order():
    """Test that the parameter order matches between webui and async_worker"""
    print("Testing parameter order...")
    
    try:
        # Read webui.py to extract ctrls order
        with open("webui.py", "r") as f:
            webui_content = f.read()
        
        # Read async_worker.py to extract args.pop() order
        with open("modules/async_worker.py", "r") as f:
            async_content = f.read()
        
        # Check key parameters are in correct positions
        checks = [
            ("force_grid_checkbox", "force_grid_checkbox should be early in the parameter list"),
            ("performance_selection", "performance_selection should be at the end"),
        ]
        
        for param, description in checks:
            if param in webui_content and param in async_content:
                print(f"✓ {param} found in both files - {description}")
            else:
                print(f"✗ {param} missing from one or both files")
                return False
        
        # Verify performance_selection is popped at the end
        if "self.performance_selection = Performance(args.pop())" in async_content:
            # Check it's after enhance_stats
            async_lines = async_content.split('\n')
            performance_line = -1
            enhance_stats_line = -1
            
            for i, line in enumerate(async_lines):
                if "self.enhance_stats = {}" in line:
                    enhance_stats_line = i
                if "self.performance_selection = Performance(args.pop())" in line:
                    performance_line = i
            
            if performance_line > enhance_stats_line > 0:
                print("✓ performance_selection is popped after enhance_stats (correct order)")
            else:
                print("✗ performance_selection is not in the correct position")
                return False
        else:
            print("✗ performance_selection pop not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing parameter order: {e}")
        return False

def test_performance_enum_usage():
    """Test that Performance enum is used correctly"""
    print("\nTesting Performance enum usage...")
    
    try:
        with open("modules/async_worker.py", "r") as f:
            content = f.read()
        
        # Check that Performance is imported
        if "from modules.flags import Performance" in content:
            print("✓ Performance enum is imported")
        else:
            print("✗ Performance enum is not imported")
            return False
        
        # Check that Performance constructor is used correctly
        if "self.performance_selection = Performance(args.pop())" in content:
            print("✓ Performance constructor is used correctly")
        else:
            print("✗ Performance constructor is not used correctly")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Performance enum usage: {e}")
        return False

def main():
    """Run parameter order tests"""
    print("Parameter Order Test")
    print("=" * 50)
    
    tests = [
        test_parameter_order,
        test_performance_enum_usage,
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
        print("✓ All parameter order tests passed!")
        print("\nThe Performance enum error should now be fixed.")
        return True
    else:
        print("✗ Some parameter order tests failed!")
        return False

if __name__ == "__main__":
    main()