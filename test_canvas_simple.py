#!/usr/bin/env python3
"""
Simple test script for the canvas UI functionality
"""

def test_javascript_canvas():
    """Test that the JavaScript canvas file is properly structured"""
    print("Testing Canvas JavaScript...")
    
    try:
        with open("javascript/canvas.js", "r") as f:
            js_content = f.read()
        
        # Check for key components
        assert "class FooocusCanvas" in js_content, "Canvas class should be defined"
        assert "addImage" in js_content, "addImage method should exist"
        assert "onMouseDown" in js_content, "Mouse handling should exist"
        assert "render" in js_content, "Render method should exist"
        assert "contextmenu" in js_content, "Context menu should be supported"
        
        print("✓ JavaScript canvas structure is correct")
        
    except FileNotFoundError:
        print("❌ Canvas JavaScript file not found")
        return False
    except Exception as e:
        print(f"❌ JavaScript test failed: {e}")
        return False
    
    return True

def test_css_canvas():
    """Test that the CSS canvas file is properly structured"""
    print("Testing Canvas CSS...")
    
    try:
        with open("css/canvas.css", "r") as f:
            css_content = f.read()
        
        # Check for key styles
        assert ".canvas-container" in css_content, "Canvas container style should exist"
        assert "#fooocus-canvas" in css_content, "Canvas element style should exist"
        assert ".canvas-control-btn" in css_content, "Control button styles should exist"
        assert ".zoom-btn" in css_content, "Zoom button styles should exist"
        
        print("✓ CSS canvas structure is correct")
        
    except FileNotFoundError:
        print("❌ Canvas CSS file not found")
        return False
    except Exception as e:
        print(f"❌ CSS test failed: {e}")
        return False
    
    return True

def test_canvas_ui_module():
    """Test that the canvas UI module is properly structured"""
    print("Testing Canvas UI Module...")
    
    try:
        with open("modules/canvas_ui.py", "r") as f:
            py_content = f.read()
        
        # Check for key components
        assert "class CanvasInterface" in py_content, "CanvasInterface class should exist"
        assert "get_canvas_html" in py_content, "HTML generation method should exist"
        assert "add_image_to_canvas" in py_content, "Image addition method should exist"
        assert "handle_canvas_generation" in py_content, "Generation handler should exist"
        assert "canvas_interface = CanvasInterface()" in py_content, "Global instance should exist"
        
        print("✓ Canvas UI module structure is correct")
        
    except FileNotFoundError:
        print("❌ Canvas UI module file not found")
        return False
    except Exception as e:
        print(f"❌ Canvas UI module test failed: {e}")
        return False
    
    return True

def test_webui_integration():
    """Test that webui.py has been properly modified"""
    print("Testing WebUI Integration...")
    
    try:
        with open("webui.py", "r") as f:
            webui_content = f.read()
        
        # Check for key integrations
        assert "from modules.canvas_ui import canvas_interface" in webui_content, "Canvas import should exist"
        assert "canvas_mode_state" in webui_content, "Canvas mode state should exist"
        assert "canvas_html" in webui_content, "Canvas HTML component should exist"
        assert "canvas_mode_btn" in webui_content, "Canvas mode button should exist"
        assert "canvas_css" in webui_content, "Canvas CSS should be embedded"
        
        print("✓ WebUI integration is correct")
        
    except FileNotFoundError:
        print("❌ WebUI file not found")
        return False
    except Exception as e:
        print(f"❌ WebUI integration test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Canvas UI Implementation...")
    print("=" * 50)
    
    tests = [
        test_javascript_canvas,
        test_css_canvas,
        test_canvas_ui_module,
        test_webui_integration
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
        print("🎉 All tests passed! Canvas UI is ready to use.")
        print("\nTo use the canvas:")
        print("1. Start Fooocus normally: python launch.py")
        print("2. Click 'Switch to Canvas Mode' button")
        print("3. Generate images - they'll appear on the canvas")
        print("4. Use right-click for context menu options")
        print("5. Drag images to move them around")
        print("6. Use mouse wheel to zoom in/out")
        print("7. Use keyboard shortcuts:")
        print("   - Delete: Remove selected images")
        print("   - Ctrl+A: Select all images")
        print("   - Ctrl+S: Save canvas state")
        print("   - Space+drag: Pan the canvas")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        exit(1)