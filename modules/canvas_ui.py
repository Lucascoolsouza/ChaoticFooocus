import gradio as gr
import json
import base64
from io import BytesIO
from PIL import Image
import modules.config
import modules.async_worker as worker
import modules.html
from modules.canvas_html import get_canvas_html_with_dark_theme


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
                        value=get_canvas_html_with_dark_theme(),
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
        # Use the working HTML from canvas_html.py
        return get_canvas_html_with_dark_theme()
    
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
            return ""
            
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
        escaped_prompt = prompt.replace("'", "\\'").replace('"', '\\"') if prompt else ""
        return f"""
        <script>
            console.log('Adding image to canvas:', '{image_path}');
            if (window.fooocusCanvas) {{
                window.fooocusCanvas.addImage('{image_path}', '{escaped_prompt}', {json.dumps(metadata or {})});
            }} else {{
                console.log('Canvas not ready, storing for later...');
                window.pendingCanvasImages = window.pendingCanvasImages || [];
                window.pendingCanvasImages.push({{
                    path: '{image_path}',
                    prompt: '{escaped_prompt}',
                    metadata: {json.dumps(metadata or {})}
                }});
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
            if script:
                scripts.append(script)
        
        return "".join(scripts)
    
    def get_canvas_integration_script(self):
        """Get the JavaScript integration script for canvas mode"""
        return """
        <script>
            // Canvas integration functions
            window.addImageToCanvas = function(imageData, prompt, metadata) {
                console.log('Adding image to canvas:', imageData, prompt);
                if (window.fooocusCanvas) {
                    window.fooocusCanvas.addImage(imageData, prompt, metadata);
                } else {
                    console.log('Canvas not ready, storing for later...');
                    window.pendingCanvasImages = window.pendingCanvasImages || [];
                    window.pendingCanvasImages.push({
                        path: imageData,
                        prompt: prompt,
                        metadata: metadata
                    });
                }
            };
            
            // Connect regeneration to Gradio
            window.regenerateWithPrompt = function(prompt, metadata) {
                console.log('Regenerating with prompt:', prompt);
                const promptInput = document.querySelector('#positive_prompt textarea');
                if (promptInput) {
                    promptInput.value = prompt;
                    promptInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
                
                const generateBtn = document.querySelector('#generate_button');
                if (generateBtn) {
                    generateBtn.click();
                }
            };
            
            // Update canvas status
            window.updateCanvasStatus = function() {
                if (window.fooocusCanvas) {
                    const zoomLevel = document.getElementById('canvas-zoom-level');
                    const imageCount = document.getElementById('canvas-image-count');
                    const selectedCount = document.getElementById('canvas-selected-count');
                    
                    if (zoomLevel) zoomLevel.textContent = `Zoom: ${Math.round(window.fooocusCanvas.zoom * 100)}%`;
                    if (imageCount) imageCount.textContent = `Images: ${window.fooocusCanvas.images ? window.fooocusCanvas.images.size : 0}`;
                    if (selectedCount) selectedCount.textContent = `Selected: ${window.fooocusCanvas.selectedImages ? window.fooocusCanvas.selectedImages.size : 0}`;
                }
            };
            
            // Set up periodic status updates
            if (!window.canvasStatusInterval) {
                window.canvasStatusInterval = setInterval(window.updateCanvasStatus, 1000);
            }
        </script>
        """
    
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
        return gr.HTML("""
        <script>
            console.log('Clearing canvas...');
            if (window.fooocusCanvas) {
                if (confirm('Are you sure you want to clear all images from the canvas?')) {
                    window.fooocusCanvas.clear();
                    console.log('Canvas cleared');
                }
            } else {
                console.log('Canvas not ready');
            }
        </script>
        """)
    
    def fit_to_screen(self):
        """Fit canvas content to screen"""
        return gr.HTML("""
        <script>
            console.log('Fitting canvas to screen...');
            if (window.fooocusCanvas) {
                window.fooocusCanvas.fitToScreen();
                console.log('Canvas fitted to screen');
            } else {
                console.log('Canvas not ready');
            }
        </script>
        """)
    
    def export_selected_images(self):
        """Export selected images from canvas"""
        return gr.HTML("""
        <script>
            console.log('Exporting selected images...');
            if (window.fooocusCanvas && window.fooocusCanvas.selectedImages && window.fooocusCanvas.selectedImages.size > 0) {
                // Create download links for selected images
                window.fooocusCanvas.selectedImages.forEach(id => {
                    const img = window.fooocusCanvas.images.get(id);
                    if (img && img.element) {
                        const link = document.createElement('a');
                        link.download = `fooocus_canvas_${id}.png`;
                        link.href = img.element.src;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    }
                });
                console.log('Selected images exported');
            } else {
                alert('No images selected for export');
                console.log('No images selected');
            }
        </script>
        """)
    
    def regenerate_selected_images(self):
        """Regenerate selected images with their original prompts"""
        return gr.HTML("""
        <script>
            console.log('Regenerating selected images...');
            if (window.fooocusCanvas && window.fooocusCanvas.selectedImages && window.fooocusCanvas.selectedImages.size > 0) {
                // Get first selected image and regenerate with its prompt
                const firstId = Array.from(window.fooocusCanvas.selectedImages)[0];
                const img = window.fooocusCanvas.images.get(firstId);
                if (img && window.regenerateWithPrompt) {
                    window.regenerateWithPrompt(img.prompt, img.metadata);
                    console.log('Regeneration triggered with prompt:', img.prompt);
                } else {
                    console.log('No regeneration function available');
                }
            } else {
                alert('No images selected for regeneration');
                console.log('No images selected');
            }
        </script>
        """)
    
    def delete_selected_images(self):
        """Delete selected images from canvas"""
        return gr.HTML("""
        <script>
            console.log('Deleting selected images...');
            if (window.fooocusCanvas && window.fooocusCanvas.selectedImages && window.fooocusCanvas.selectedImages.size > 0) {
                if (confirm('Are you sure you want to delete the selected images?')) {
                    if (window.fooocusCanvas.deleteSelectedImages) {
                        window.fooocusCanvas.deleteSelectedImages();
                        console.log('Selected images deleted');
                    } else {
                        // Fallback deletion
                        window.fooocusCanvas.selectedImages.forEach(id => {
                            window.fooocusCanvas.images.delete(id);
                        });
                        window.fooocusCanvas.selectedImages.clear();
                        if (window.fooocusCanvas.render) {
                            window.fooocusCanvas.render();
                        }
                        console.log('Selected images deleted (fallback)');
                    }
                }
            } else {
                alert('No images selected for deletion');
                console.log('No images selected');
            }
        </script>
        """)
    
    def select_all_images(self):
        """Select all images on canvas"""
        return gr.HTML("""
        <script>
            console.log('Selecting all images...');
            if (window.fooocusCanvas) {
                if (window.fooocusCanvas.selectAllImages) {
                    window.fooocusCanvas.selectAllImages();
                } else {
                    // Fallback selection
                    window.fooocusCanvas.selectedImages = window.fooocusCanvas.selectedImages || new Set();
                    window.fooocusCanvas.selectedImages.clear();
                    if (window.fooocusCanvas.images) {
                        window.fooocusCanvas.images.forEach((img, id) => {
                            window.fooocusCanvas.selectedImages.add(id);
                        });
                    }
                    if (window.fooocusCanvas.render) {
                        window.fooocusCanvas.render();
                    }
                }
                console.log('All images selected');
            } else {
                console.log('Canvas not ready');
            }
        </script>
        """)
    
    def deselect_all_images(self):
        """Deselect all images on canvas"""
        return gr.HTML("""
        <script>
            console.log('Deselecting all images...');
            if (window.fooocusCanvas) {
                if (window.fooocusCanvas.selectedImages) {
                    window.fooocusCanvas.selectedImages.clear();
                }
                if (window.fooocusCanvas.render) {
                    window.fooocusCanvas.render();
                }
                console.log('All images deselected');
            } else {
                console.log('Canvas not ready');
            }
        </script>
        """)
    
    def save_canvas(self):
        """Save canvas state"""
        return gr.HTML("""
        <script>
            console.log('Saving canvas...');
            if (window.fooocusCanvas) {
                if (window.fooocusCanvas.saveCanvas) {
                    window.fooocusCanvas.saveCanvas();
                }
                alert('Canvas saved successfully!');
                console.log('Canvas saved');
            } else {
                console.log('Canvas not ready');
            }
        </script>
        """)


# Global canvas interface instance
canvas_interface = CanvasInterface()