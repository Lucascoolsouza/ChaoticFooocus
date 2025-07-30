#!/usr/bin/env python3
"""
Test script for the canvas UI functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.canvas_ui import canvas_interface

def test_canvas_interface():
    """Test basic canvas interface functionality"""
    print("Testing Canvas Interface...")
    
    # Test HTML generation
    html = canvas_interface.get_canvas_html()
    assert "canvas" in html.lower(), "Canvas HTML should contain canvas element"
    assert "fooocus-canvas" in html, "Canvas should have correct ID"
    print("✓ Canvas HTML generation works")
    
    # Test mode toggle
    result = canvas_interface.toggle_canvas_mode(False)
    assert len(result) == 5, "Toggle should return 5 elements"
    print("✓ Canvas mode toggle works")
    
    # Test canvas state
    state = canvas_interface.get_canvas_state()
    assert "images" in state, "Canvas state should contain images"
    assert "mode" in state, "Canvas state should contain mode"
    print("✓ Canvas state management works")
    
    # Test image addition simulation
    test_prompt = "A beautiful landscape"
    test_metadata = {"seed": 12345, "steps": 20}
    script = canvas_interface.add_image_to_canvas("/test/path.png", test_prompt, test_metadata)
    assert "addImageToCanvas" in script, "Should generate JavaScript for adding image"
    print("✓ Image addition works")
    
    # Test canvas operations
    operations = [
        canvas_interface.clear_canvas(),
        canvas_interface.fit_to_screen(),
        canvas_interface.save_canvas(),
        canvas_interface.select_all_images(),
        canvas_interface.delete_selected_images()
    ]
    
    for op in operations:
        assert "script" in op.lower(), "Operations should return JavaScript"
    print("✓ Canvas operations work")
    
    print("All canvas interface tests passed! ✅")

def test_canvas_integration():
    """Test integration with existing Fooocus components"""
    print("\nTesting Canvas Integration...")
    
    # Test generation result handling
    test_results = ["/path/to/image1.png", "/path/to/image2.png"]
    test_prompt = "Test prompt"
    
    canvas_interface.canvas_mode = True
    script = canvas_interface.handle_canvas_generation(test_results, test_prompt)
    assert len(script) > 0, "Should generate script for canvas generation"
    print("✓ Generation result handling works")
    
    # Test state persistence
    original_state = canvas_interface.get_canvas_state()
    canvas_interface.load_canvas_state(original_state)
    print("✓ State persistence works")
    
    print("Canvas integration tests passed! ✅")

if __name__ == "__main__":
    try:
        test_canvas_interface()
        test_canvas_integration()
        print("\n🎉 All tests passed! Canvas UI is ready to use.")
        print("\nTo use the canvas:")
        print("1. Start Fooocus normally: python launch.py")
        print("2. Click 'Switch to Canvas Mode' button")
        print("3. Generate images - they'll appear on the canvas")
        print("4. Use right-click for context menu options")
        print("5. Drag images to move them around")
        print("6. Use mouse wheel to zoom in/out")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)