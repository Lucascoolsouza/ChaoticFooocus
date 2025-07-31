#!/usr/bin/env python3
"""
Test script to verify canvas integration in webui.py
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_canvas_imports():
    """Test that all canvas-related imports work"""
    print("Testing canvas imports...")
    
    try:
        from modules.canvas_ui import canvas_interface
        print("✅ Canvas interface imported successfully")
        
        from modules.canvas_html import get_canvas_html_with_dark_theme
        print("✅ Canvas HTML imported successfully")
        
        # Test canvas interface methods
        html_content = canvas_interface.get_canvas_html()
        print(f"✅ Canvas HTML generated: {len(html_content)} characters")
        
        # Test button methods
        clear_result = canvas_interface.clear_canvas()
        print(f"✅ Clear canvas method works: {type(clear_result)}")
        
        fit_result = canvas_interface.fit_to_screen()
        print(f"✅ Fit to screen method works: {type(fit_result)}")
        
        # Test image addition
        test_image = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjU2IiBoZWlnaHQ9IjI1NiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjU2IiBoZWlnaHQ9IjI1NiIgZmlsbD0iIzQyODVmNCIvPjx0ZXh0IHg9IjEyOCIgeT0iMTI4IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMjQiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+VGVzdCBJbWFnZTwvdGV4dD48L3N2Zz4="
        canvas_interface.canvas_mode = True
        add_result = canvas_interface.add_image_to_canvas(test_image, "Test image", {"test": True})
        print(f"✅ Add image method works: {len(add_result) if add_result else 0} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_canvas_html_structure():
    """Test that the canvas HTML has the required structure"""
    print("\nTesting canvas HTML structure...")
    
    try:
        from modules.canvas_ui import canvas_interface
        html_content = canvas_interface.get_canvas_html()
        
        # Check for required elements
        required_elements = [
            'fooocus-canvas',
            'canvas-controls',
            'canvas-status-bar',
            'canvas-zoom-controls',
            'setCanvasTool',
            'fitCanvasToScreen',
            'clearCanvas',
            'saveCanvas'
        ]
        
        for element in required_elements:
            if element in html_content:
                print(f"✅ Found required element: {element}")
            else:
                print(f"❌ Missing required element: {element}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ HTML structure test failed: {e}")
        return False

def test_canvas_integration_script():
    """Test that the canvas integration script is generated"""
    print("\nTesting canvas integration script...")
    
    try:
        from modules.canvas_ui import canvas_interface
        script = canvas_interface.get_canvas_integration_script()
        
        # Check for required functions
        required_functions = [
            'addImageToCanvas',
            'regenerateWithPrompt',
            'updateCanvasStatus'
        ]
        
        for func in required_functions:
            if func in script:
                print(f"✅ Found required function: {func}")
            else:
                print(f"❌ Missing required function: {func}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration script test failed: {e}")
        return False

def main():
    """Run all canvas tests"""
    print("Canvas WebUI Integration Test")
    print("=" * 50)
    
    tests = [
        test_canvas_imports,
        test_canvas_html_structure,
        test_canvas_integration_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Canvas integration should work.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())