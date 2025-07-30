import gradio as gr
import json
import base64
from io import BytesIO
from PIL import Image
import modules.config
import modules.async_worker as worker
import modules.html


class CanvasInterface:
    def __init__(self):
        self.canvas_mode = False
        self.canvas_images = {}
        self.next_image_id = 1
        
    def create_canvas_ui(self):
        """Create the canvas-based UI components"""
        with gr.Column(visible=False, elem_id="canvas-mode-container") as canvas_container:
            # Mode toggle button
            with gr.Row():
                canvas_mode_btn = gr.Button(
                    "Switch to Canvas Mode", 
                    elem_id="canvas-mode-toggle",
                    elem_classes=["mode-toggle-btn"]
                )
                gallery_mode_btn = gr.Button(
                    "Switch to Gallery Mode", 
                    elem_id="gallery-mode-toggle",
                    elem_classes=["mode-toggle-btn"],
                    visible=False
                )
            
            # Canvas container
            with gr.Row():
                with gr.Column(scale=4):
                    # Canvas HTML component
                    canvas_html = gr.HTML(
                        value=self.get_canvas_html(),
                        elem_id="canvas-container",
                        elem_classes=["canvas-container"]
                    )
                    
                    # Canvas controls
                    with gr.Row():
                        fit_screen_btn = gr.Button("Fit to Screen", size="sm")
                        clear_canvas_btn = gr.Button("Clear Canvas", size="sm")
                        save_canvas_btn = gr.Button("Save Canvas", size="sm")
                        load_canvas_btn = gr.Button("Load Canvas", size="sm")
                        export_selected_btn = gr.Button("Export Selected", size="sm")
                
                with gr.Column(scale=1, min_width=200):
                    # Canvas info panel
                    canvas_info = gr.HTML(
                        value="<div><h4>Canvas Info</h4><p>Images: 0</p><p>Selected: 0</p></div>",
                        elem_id="canvas-info"
                    )
                    
                    # Quick actions
                    with gr.Group():
                        gr.HTML("<h5>Quick Actions</h5>")
                        regenerate_selected_btn = gr.Button("Regenerate Selected", size="sm")
                        delete_selected_btn = gr.Button("Delete Selected", size="sm")
                        select_all_btn = gr.Button("Select All", size="sm")
                        deselect_all_btn = gr.Button("Deselect All", size="sm")
        
        return (canvas_container, canvas_mode_btn, gallery_mode_btn, canvas_html, 
                canvas_info, fit_screen_btn, clear_canvas_btn, save_canvas_btn, 
                load_canvas_btn, export_selected_btn, regenerate_selected_btn, 
                delete_selected_btn, select_all_btn, deselect_all_btn)
    
    def get_canvas_html(self):
        """Generate the HTML for the canvas interface"""
        return f"""
        <div class="canvas-container">
            <canvas id="fooocus-canvas" width="800" height="600"></canvas>
            
            <div class="canvas-controls" id="canvas-controls">
                <div class="canvas-toolbar-group">
                    <button class="canvas-tool-btn active" data-tool="select" onclick="window.fooocusCanvas?.setTool('select')" title="Select Tool">
                        🔍
                    </button>
                    <button class="canvas-tool-btn" data-tool="frame" onclick="window.fooocusCanvas?.setTool('frame')" title="Frame Tool">
                        🖼️
                    </button>
                    <button class="canvas-tool-btn" data-tool="outpaint" onclick="window.fooocusCanvas?.setTool('outpaint')" title="Outpaint Tool">
                        🎨
                    </button>
                </div>
                
                <div class="canvas-toolbar-group">
                    <button class="canvas-control-btn" onclick="window.fooocusCanvas?.fitToScreen()">
                        📐 Fit Screen
                    </button>
                    <button class="canvas-control-btn" onclick="window.fooocusCanvas?.clear()">
                        🗑️ Clear
                    </button>
                    <button class="canvas-control-btn" onclick="window.fooocusCanvas?.saveCanvas()">
                        💾 Save
                    </button>
                </div>
            </div>
            
            <div class="canvas-zoom-controls">
                <button class="zoom-btn" onclick="window.fooocusCanvas?.zoom *= 1.2; window.fooocusCanvas?.render()">+</button>
                <button class="zoom-btn" onclick="window.fooocusCanvas?.zoom *= 0.8; window.fooocusCanvas?.render()">−</button>
                <button class="zoom-btn" onclick="window.fooocusCanvas?.fitToScreen()">⌂</button>
            </div>
            
            <div class="canvas-status-bar">
                <span id="canvas-zoom-level">Zoom: 100%</span>
                <span id="canvas-image-count">Images: 0</span>
                <span id="canvas-selected-count">Selected: 0</span>
            </div>
        </div>
        
        <script>
            // Initialize canvas integration
            if (typeof window.initCanvasIntegration === 'undefined') {{
                window.initCanvasIntegration = function() {{
                    // Connect canvas to generation system
                    window.addImageToCanvas = function(imageData, prompt, metadata) {{
                        if (window.fooocusCanvas) {{
                            window.fooocusCanvas.addImage(imageData, prompt, metadata);
                        }}
                    }};
                    
                    // Connect regeneration to Gradio
                    window.regenerateWithPrompt = function(prompt, metadata) {{
                        // Find the prompt input and set it
                        const promptInput = document.querySelector('#positive_prompt textarea');
                        if (promptInput) {{
                            promptInput.value = prompt;
                            promptInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                        
                        // Trigger generation
                        const generateBtn = document.querySelector('#generate_button');
                        if (generateBtn) {{
                            generateBtn.click();
                        }}
                    }};
                    
                    // Connect outpainting to Gradio
                    window.triggerOutpaint = function(sourceImage, outpaintRect, direction) {{
                        console.log('Triggering outpaint:', sourceImage, outpaintRect, direction);
                        
                        // Set outpaint mode in the interface
                        const outpaintTab = document.querySelector('#inpaint_tab');
                        if (outpaintTab) {{
                            outpaintTab.click();
                        }}
                        
                        // Set outpaint direction
                        const directionMap = {{
                            'left': ['Left'],
                            'right': ['Right'], 
                            'top': ['Top'],
                            'bottom': ['Bottom']
                        }};
                        
                        const outpaintSelections = document.querySelector('input[type="checkbox"][value="' + direction.charAt(0).toUpperCase() + direction.slice(1) + '"]');
                        if (outpaintSelections) {{
                            outpaintSelections.checked = true;
                            outpaintSelections.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                        
                        // Use the source image prompt
                        const promptInput = document.querySelector('#positive_prompt textarea');
                        if (promptInput && sourceImage.prompt) {{
                            promptInput.value = sourceImage.prompt;
                            promptInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                        
                        // Trigger generation
                        const generateBtn = document.querySelector('#generate_button');
                        if (generateBtn) {{
                            generateBtn.click();
                        }}
                    }};
                    
                    // Connect inpainting to Gradio
                    window.triggerInpaint = function(rect, imageIds) {{
                        console.log('Triggering inpaint:', rect, imageIds);
                        
                        // Switch to inpaint tab
                        const inpaintTab = document.querySelector('#inpaint_tab');
                        if (inpaintTab) {{
                            inpaintTab.click();
                        }}
                        
                        // Trigger generation
                        const generateBtn = document.querySelector('#generate_button');
                        if (generateBtn) {{
                            generateBtn.click();
                        }}
                    }};
                    
                    // Connect upscaling to Gradio
                    window.triggerUpscale = function(rect, imageIds) {{
                        console.log('Triggering upscale:', rect, imageIds);
                        
                        // Switch to upscale tab
                        const uovTab = document.querySelector('#uov_tab');
                        if (uovTab) {{
                            uovTab.click();
                        }}
                        
                        // Trigger generation
                        const generateBtn = document.querySelector('#generate_button');
                        if (generateBtn) {{
                            generateBtn.click();
                        }}
                    }};
                    
                    // Connect general generation to Gradio
                    window.triggerGeneration = function(rect) {{
                        console.log('Triggering generation for area:', rect);
                        
                        // Trigger generation
                        const generateBtn = document.querySelector('#generate_button');
                        if (generateBtn) {{
                            generateBtn.click();
                        }}
                    }};
                    
                    // Update status bar
                    window.updateCanvasStatus = function() {{
                        if (window.fooocusCanvas) {{
                            const zoomLevel = document.getElementById('canvas-zoom-level');
                            const imageCount = document.getElementById('canvas-image-count');
                            const selectedCount = document.getElementById('canvas-selected-count');
                            
                            if (zoomLevel) zoomLevel.textContent = `Zoom: ${{Math.round(window.fooocusCanvas.zoom * 100)}}%`;
                            if (imageCount) imageCount.textContent = `Images: ${{window.fooocusCanvas.images.size}}`;
                            if (selectedCount) selectedCount.textContent = `Selected: ${{window.fooocusCanvas.selectedImages.size}}`;
                        }}
                    }};
                    
                    // Set up periodic status updates
                    setInterval(window.updateCanvasStatus, 1000);
                }};
                
                // Initialize when DOM is ready
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', window.initCanvasIntegration);
                }} else {{
                    window.initCanvasIntegration();
                }}
            }}
        </script>
        """
    
    def toggle_canvas_mode(self, current_mode):
        """Toggle between canvas and gallery modes"""
        new_mode = not current_mode
        return (
            gr.update(visible=new_mode),  # canvas container
            gr.update(visible=not new_mode),  # gallery container
            gr.update(visible=new_mode, value="Switch to Gallery Mode" if new_mode else "Switch to Canvas Mode"),  # canvas mode btn
            gr.update(visible=not new_mode, value="Switch to Canvas Mode" if not new_mode else "Switch to Gallery Mode"),  # gallery mode btn
            new_mode
        )
    
    def add_image_to_canvas(self, image_path, prompt, metadata=None):
        """Add a generated image to the canvas"""
        if not self.canvas_mode:
            return
            
        image_id = self.next_image_id
        self.next_image_id += 1
        
        # Store image data
        self.canvas_images[image_id] = {
            'path': image_path,
            'prompt': prompt,
            'metadata': metadata or {},
            'id': image_id
        }
        
        # Return JavaScript to add image to canvas
        return f"""
        <script>
            if (window.addImageToCanvas) {{
                window.addImageToCanvas('{image_path}', '{prompt}', {json.dumps(metadata or {})});
            }}
        </script>
        """
    
    def handle_canvas_generation(self, task_results, prompt, metadata=None):
        """Handle generation results for canvas mode"""
        if not self.canvas_mode or not task_results:
            return ""
            
        scripts = []
        for image_path in task_results:
            script = self.add_image_to_canvas(image_path, prompt, metadata)
            scripts.append(script)
        
        return "".join(scripts)
    
    def get_canvas_state(self):
        """Get current canvas state for saving"""
        return {
            'images': self.canvas_images,
            'next_id': self.next_image_id,
            'mode': self.canvas_mode
        }
    
    def load_canvas_state(self, state_data):
        """Load canvas state from saved data"""
        try:
            state = json.loads(state_data) if isinstance(state_data, str) else state_data
            self.canvas_images = state.get('images', {})
            self.next_image_id = state.get('next_id', 1)
            self.canvas_mode = state.get('mode', False)
            return True
        except Exception as e:
            print(f"Failed to load canvas state: {e}")
            return False
    
    def clear_canvas(self):
        """Clear all images from canvas"""
        self.canvas_images.clear()
        return """
        <script>
            if (window.fooocusCanvas) {
                window.fooocusCanvas.clear();
            }
        </script>
        """
    
    def fit_to_screen(self):
        """Fit canvas content to screen"""
        return """
        <script>
            if (window.fooocusCanvas) {
                window.fooocusCanvas.fitToScreen();
            }
        </script>
        """
    
    def export_selected_images(self):
        """Export selected images from canvas"""
        return """
        <script>
            if (window.fooocusCanvas && window.fooocusCanvas.selectedImages.size > 0) {
                // Create download links for selected images
                window.fooocusCanvas.selectedImages.forEach(id => {
                    const img = window.fooocusCanvas.images.get(id);
                    if (img) {
                        const link = document.createElement('a');
                        link.download = `fooocus_canvas_${id}.png`;
                        link.href = img.element.src;
                        link.click();
                    }
                });
            }
        </script>
        """
    
    def regenerate_selected_images(self):
        """Regenerate selected images with their original prompts"""
        return """
        <script>
            if (window.fooocusCanvas && window.fooocusCanvas.selectedImages.size > 0) {
                // Get first selected image and regenerate with its prompt
                const firstId = Array.from(window.fooocusCanvas.selectedImages)[0];
                const img = window.fooocusCanvas.images.get(firstId);
                if (img && window.regenerateWithPrompt) {
                    window.regenerateWithPrompt(img.prompt, img.metadata);
                }
            }
        </script>
        """
    
    def delete_selected_images(self):
        """Delete selected images from canvas"""
        return """
        <script>
            if (window.fooocusCanvas) {
                window.fooocusCanvas.deleteSelectedImages();
            }
        </script>
        """
    
    def select_all_images(self):
        """Select all images on canvas"""
        return """
        <script>
            if (window.fooocusCanvas) {
                window.fooocusCanvas.selectAllImages();
            }
        </script>
        """
    
    def deselect_all_images(self):
        """Deselect all images on canvas"""
        return """
        <script>
            if (window.fooocusCanvas) {
                window.fooocusCanvas.selectedImages.clear();
                window.fooocusCanvas.render();
            }
        </script>
        """
    
    def save_canvas(self):
        """Save canvas state"""
        return """
        <script>
            if (window.fooocusCanvas) {
                window.fooocusCanvas.saveCanvas();
                alert('Canvas saved successfully!');
            }
        </script>
        """


# Global canvas interface instance
canvas_interface = CanvasInterface()