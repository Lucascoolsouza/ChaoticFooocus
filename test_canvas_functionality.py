#!/usr/bin/env python3
"""
Test script to verify canvas functionality
"""

import gradio as gr
import os
import sys
from modules.canvas_ui import CanvasInterface

def test_canvas_ui():
    """Test the canvas UI components"""
    print("Testing Canvas UI...")
    
    # Create canvas interface
    canvas_interface = CanvasInterface()
    
    # Test canvas mode toggle
    print("Testing canvas mode toggle...")
    result = canvas_interface.toggle_canvas_mode(False)
    print(f"Toggle result: {len(result)} components returned")
    
    # Test canvas UI creation
    print("Testing canvas UI creation...")
    ui_components = canvas_interface.create_canvas_ui()
    print(f"UI components created: {len(ui_components)} components")
    
    # Test button functions
    print("Testing button functions...")
    
    # Test clear canvas
    clear_result = canvas_interface.clear_canvas()
    print(f"Clear canvas result type: {type(clear_result)}")
    
    # Test fit to screen
    fit_result = canvas_interface.fit_to_screen()
    print(f"Fit to screen result type: {type(fit_result)}")
    
    # Test save canvas
    save_result = canvas_interface.save_canvas()
    print(f"Save canvas result type: {type(save_result)}")
    
    # Test image addition
    print("Testing image addition...")
    canvas_interface.canvas_mode = True
    test_image_path = "test_image.png"
    test_prompt = "A beautiful landscape"
    
    add_result = canvas_interface.add_image_to_canvas(test_image_path, test_prompt)
    print(f"Add image result: {len(add_result) if add_result else 0} characters")
    
    print("Canvas UI test completed successfully!")
    return True

def create_test_interface():
    """Create a test Gradio interface"""
    canvas_interface = CanvasInterface()
    
    with gr.Blocks(title="Canvas UI Test") as demo:
        gr.Markdown("# Canvas UI Test Interface")
        
        # Create canvas UI
        canvas_components = canvas_interface.create_canvas_ui()
        (canvas_container, canvas_mode_btn, gallery_mode_btn, canvas_html, 
         canvas_info, fit_screen_btn, clear_canvas_btn, save_canvas_btn, 
         load_canvas_btn, export_selected_btn, regenerate_selected_btn, 
         delete_selected_btn, select_all_btn, deselect_all_btn) = canvas_components
        
        # Test controls
        with gr.Row():
            test_add_btn = gr.Button("Add Test Image")
            test_clear_btn = gr.Button("Clear Canvas")
            test_fit_btn = gr.Button("Fit to Screen")
        
        # Status display
        status_output = gr.HTML()
        
        # Button event handlers
        canvas_mode_btn.click(
            fn=lambda: canvas_interface.toggle_canvas_mode(False),
            outputs=[canvas_container, canvas_container, canvas_mode_btn, gallery_mode_btn]
        )
        
        test_add_btn.click(
            fn=lambda: canvas_interface.add_image_to_canvas(
                "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjU2IiBoZWlnaHQ9IjI1NiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjU2IiBoZWlnaHQ9IjI1NiIgZmlsbD0iIzQyODVmNCIvPjx0ZXh0IHg9IjEyOCIgeT0iMTI4IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMjQiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+VGVzdCBJbWFnZTwvdGV4dD48L3N2Zz4=",
                "Test image for canvas",
                {"test": True}
            ),
            outputs=[status_output]
        )
        
        test_clear_btn.click(
            fn=canvas_interface.clear_canvas,
            outputs=[status_output]
        )
        
        test_fit_btn.click(
            fn=canvas_interface.fit_to_screen,
            outputs=[status_output]
        )
        
        # Wire up canvas buttons
        clear_canvas_btn.click(
            fn=canvas_interface.clear_canvas,
            outputs=[status_output]
        )
        
        fit_screen_btn.click(
            fn=canvas_interface.fit_to_screen,
            outputs=[status_output]
        )
        
        save_canvas_btn.click(
            fn=canvas_interface.save_canvas,
            outputs=[status_output]
        )
        
        export_selected_btn.click(
            fn=canvas_interface.export_selected_images,
            outputs=[status_output]
        )
        
        regenerate_selected_btn.click(
            fn=canvas_interface.regenerate_selected_images,
            outputs=[status_output]
        )
        
        delete_selected_btn.click(
            fn=canvas_interface.delete_selected_images,
            outputs=[status_output]
        )
        
        select_all_btn.click(
            fn=canvas_interface.select_all_images,
            outputs=[status_output]
        )
        
        deselect_all_btn.click(
            fn=canvas_interface.deselect_all_images,
            outputs=[status_output]
        )
    
    return demo

if __name__ == "__main__":
    print("Canvas UI Test Script")
    print("=" * 50)
    
    # Run basic tests
    try:
        test_canvas_ui()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    
    # Launch test interface if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("\nLaunching test interface...")
        demo = create_test_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )