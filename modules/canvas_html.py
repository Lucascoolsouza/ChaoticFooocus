"""
Canvas HTML generation with working buttons and dark blue dotted background
"""

def get_canvas_html_with_dark_theme():
    """Generate the HTML for the canvas interface with dark blue dotted background"""
    return """
    <div class="canvas-container" style="background: #1a2332; border: 1px solid #2d3748; border-radius: 8px; overflow: hidden; box-shadow: inset 0 0 20px rgba(0,0,0,0.3);">
        <canvas id="fooocus-canvas" width="800" height="600" style="
            display: block;
            cursor: default;
            background-color: #1a2332;
            background-image: radial-gradient(circle, #4a5568 1px, transparent 1px);
            background-size: 20px 20px;
            background-position: 0 0;
        "></canvas>
        
        <div class="canvas-controls" id="canvas-controls" style="
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            z-index: 100;
        ">
            <div class="canvas-toolbar-group" style="
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
                backdrop-filter: blur(5px);
                margin-bottom: 8px;
            ">
                <button class="canvas-tool-btn active" data-tool="select" onclick="setCanvasTool('select')" title="Select Tool" style="
                    display: inline-block;
                    width: 36px;
                    height: 36px;
                    border: none;
                    background: #007bff;
                    color: white;
                    cursor: pointer;
                    border-radius: 4px;
                    transition: all 0.2s;
                    font-size: 16px;
                    margin: 2px;
                ">
                    🔍
                </button>
                <button class="canvas-tool-btn" data-tool="frame" onclick="setCanvasTool('frame')" title="Frame Tool" style="
                    display: inline-block;
                    width: 36px;
                    height: 36px;
                    border: none;
                    background: transparent;
                    color: #495057;
                    cursor: pointer;
                    border-radius: 4px;
                    transition: all 0.2s;
                    font-size: 16px;
                    margin: 2px;
                ">
                    🖼️
                </button>
                <button class="canvas-tool-btn" data-tool="outpaint" onclick="setCanvasTool('outpaint')" title="Outpaint Tool" style="
                    display: inline-block;
                    width: 36px;
                    height: 36px;
                    border: none;
                    background: transparent;
                    color: #495057;
                    cursor: pointer;
                    border-radius: 4px;
                    transition: all 0.2s;
                    font-size: 16px;
                    margin: 2px;
                ">
                    🎨
                </button>
            </div>
            
            <div class="canvas-toolbar-group" style="
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
                backdrop-filter: blur(5px);
            ">
                <button class="canvas-control-btn" onclick="fitCanvasToScreen()" style="
                    padding: 8px 12px;
                    background: rgba(255, 255, 255, 0.9);
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: all 0.2s;
                    backdrop-filter: blur(5px);
                    display: block;
                    width: 100%;
                    margin-bottom: 4px;
                ">
                    📐 Fit Screen
                </button>
                <button class="canvas-control-btn" onclick="clearCanvas()" style="
                    padding: 8px 12px;
                    background: rgba(255, 255, 255, 0.9);
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: all 0.2s;
                    backdrop-filter: blur(5px);
                    display: block;
                    width: 100%;
                    margin-bottom: 4px;
                ">
                    🗑️ Clear
                </button>
                <button class="canvas-control-btn" onclick="saveCanvas()" style="
                    padding: 8px 12px;
                    background: rgba(255, 255, 255, 0.9);
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    transition: all 0.2s;
                    backdrop-filter: blur(5px);
                    display: block;
                    width: 100%;
                ">
                    💾 Save
                </button>
            </div>
        </div>
        
        <div class="canvas-zoom-controls" style="
            position: absolute;
            bottom: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
            z-index: 100;
        ">
            <button class="zoom-btn" onclick="zoomCanvas(1.2)" style="
                width: 32px;
                height: 32px;
                border: 1px solid #dee2e6;
                background: rgba(255, 255, 255, 0.9);
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                font-weight: bold;
                color: #495057;
                transition: all 0.2s;
            ">+</button>
            <button class="zoom-btn" onclick="zoomCanvas(0.8)" style="
                width: 32px;
                height: 32px;
                border: 1px solid #dee2e6;
                background: rgba(255, 255, 255, 0.9);
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                font-weight: bold;
                color: #495057;
                transition: all 0.2s;
            ">−</button>
            <button class="zoom-btn" onclick="fitCanvasToScreen()" style="
                width: 32px;
                height: 32px;
                border: 1px solid #dee2e6;
                background: rgba(255, 255, 255, 0.9);
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                font-weight: bold;
                color: #495057;
                transition: all 0.2s;
            ">⌂</button>
        </div>
        
        <div class="canvas-status-bar" style="
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 30px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            display: flex;
            align-items: center;
            padding: 0 15px;
            font-size: 12px;
            gap: 20px;
        ">
            <span id="canvas-zoom-level">Zoom: 100%</span>
            <span id="canvas-image-count">Images: 0</span>
            <span id="canvas-selected-count">Selected: 0</span>
            <span id="canvas-tool-status">Tool: Select</span>
        </div>
    </div>
    
    <script>
        // Canvas utility functions that work immediately
        function setCanvasTool(tool) {
            console.log('Setting canvas tool to:', tool);
            
            // Update button states
            document.querySelectorAll('.canvas-tool-btn').forEach(btn => {
                btn.style.background = 'transparent';
                btn.style.color = '#495057';
            });
            
            const activeBtn = document.querySelector(`[data-tool="${tool}"]`);
            if (activeBtn) {
                activeBtn.style.background = '#007bff';
                activeBtn.style.color = 'white';
            }
            
            // Update status
            const statusEl = document.getElementById('canvas-tool-status');
            if (statusEl) {
                statusEl.textContent = `Tool: ${tool.charAt(0).toUpperCase() + tool.slice(1)}`;
            }
            
            // Set tool on canvas if available
            if (window.fooocusCanvas) {
                window.fooocusCanvas.setTool(tool);
            } else {
                // Store tool selection for when canvas loads
                window.pendingCanvasTool = tool;
            }
        }
        
        function fitCanvasToScreen() {
            console.log('Fitting canvas to screen');
            if (window.fooocusCanvas) {
                window.fooocusCanvas.fitToScreen();
            } else {
                console.log('Canvas not ready yet');
            }
        }
        
        function clearCanvas() {
            console.log('Clearing canvas');
            if (window.fooocusCanvas) {
                if (confirm('Are you sure you want to clear all images from the canvas?')) {
                    window.fooocusCanvas.clear();
                }
            } else {
                console.log('Canvas not ready yet');
            }
        }
        
        function saveCanvas() {
            console.log('Saving canvas');
            if (window.fooocusCanvas) {
                window.fooocusCanvas.saveCanvas();
                alert('Canvas saved successfully!');
            } else {
                console.log('Canvas not ready yet');
            }
        }
        
        function zoomCanvas(factor) {
            console.log('Zooming canvas by factor:', factor);
            if (window.fooocusCanvas) {
                window.fooocusCanvas.zoom *= factor;
                window.fooocusCanvas.zoom = Math.max(0.1, Math.min(5, window.fooocusCanvas.zoom));
                window.fooocusCanvas.render();
                
                // Update zoom display
                const zoomEl = document.getElementById('canvas-zoom-level');
                if (zoomEl) {
                    zoomEl.textContent = `Zoom: ${Math.round(window.fooocusCanvas.zoom * 100)}%`;
                }
            } else {
                console.log('Canvas not ready yet');
            }
        }
        
        // Initialize canvas when ready
        function initializeCanvas() {
            console.log('Initializing canvas...');
            
            // Load canvas script if not already loaded
            if (!window.FooocusCanvas) {
                console.log('Loading canvas script...');
                const script = document.createElement('script');
                script.src = 'javascript/canvas.js';
                script.onload = function() {
                    console.log('Canvas script loaded, creating canvas instance...');
                    setTimeout(() => {
                        createCanvasInstance();
                    }, 100);
                };
                script.onerror = function() {
                    console.error('Failed to load canvas script, creating fallback...');
                    createFallbackCanvas();
                };
                document.head.appendChild(script);
            } else if (document.getElementById('fooocus-canvas') && !window.fooocusCanvas) {
                console.log('Canvas class exists, creating instance...');
                createCanvasInstance();
            }
        }
        
        function createCanvasInstance() {
            if (document.getElementById('fooocus-canvas') && !window.fooocusCanvas) {
                try {
                    window.fooocusCanvas = new FooocusCanvas('fooocus-canvas', 'canvas-controls');
                    
                    // Apply pending tool selection
                    if (window.pendingCanvasTool) {
                        window.fooocusCanvas.setTool(window.pendingCanvasTool);
                        delete window.pendingCanvasTool;
                    }
                    
                    // Add any pending images
                    if (window.pendingCanvasImages && window.pendingCanvasImages.length > 0) {
                        console.log('Adding pending images:', window.pendingCanvasImages.length);
                        window.pendingCanvasImages.forEach(img => {
                            window.fooocusCanvas.addImage(img.path, img.prompt, img.metadata);
                        });
                        window.pendingCanvasImages = [];
                    }
                    
                    // Set up file input handler
                    const fileInput = document.getElementById('canvas-file-input');
                    if (fileInput) {
                        fileInput.addEventListener('change', function(e) {
                            Array.from(e.target.files).forEach(file => {
                                if (file.type.startsWith('image/')) {
                                    window.fooocusCanvas.addImageFromFile(file, 400, 300);
                                }
                            });
                            e.target.value = ''; // Reset input
                        });
                    }
                    
                    console.log('Canvas initialized successfully!');
                } catch (error) {
                    console.error('Failed to create canvas instance:', error);
                    createFallbackCanvas();
                }
            }
            
            // Set up status updates
            if (!window.canvasStatusInterval) {
                window.canvasStatusInterval = setInterval(() => {
                    if (window.fooocusCanvas) {
                        const zoomEl = document.getElementById('canvas-zoom-level');
                        const imageEl = document.getElementById('canvas-image-count');
                        const selectedEl = document.getElementById('canvas-selected-count');
                        
                        if (zoomEl) zoomEl.textContent = `Zoom: ${Math.round(window.fooocusCanvas.zoom * 100)}%`;
                        if (imageEl) imageEl.textContent = `Images: ${window.fooocusCanvas.images ? window.fooocusCanvas.images.size : 0}`;
                        if (selectedEl) selectedEl.textContent = `Selected: ${window.fooocusCanvas.selectedImages ? window.fooocusCanvas.selectedImages.size : 0}`;
                    }
                }, 1000);
            }
        }
        
        function createFallbackCanvas() {
            console.log('Creating fallback canvas implementation...');
            
            // Create a minimal canvas implementation
            window.fooocusCanvas = {
                images: new Map(),
                selectedImages: new Set(),
                zoom: 1,
                panX: 0,
                panY: 0,
                
                addImage: function(imagePath, prompt, metadata) {
                    console.log('Fallback: Adding image', imagePath, prompt);
                    const img = new Image();
                    img.onload = () => {
                        const canvas = document.getElementById('fooocus-canvas');
                        const ctx = canvas.getContext('2d');
                        
                        // Simple positioning
                        const x = (this.images.size % 3) * 270 + 10;
                        const y = Math.floor(this.images.size / 3) * 270 + 10;
                        
                        // Draw image
                        ctx.drawImage(img, x, y, 256, 256);
                        
                        // Store image data
                        this.images.set(this.images.size + 1, {
                            id: this.images.size + 1,
                            element: img,
                            x: x,
                            y: y,
                            width: 256,
                            height: 256,
                            prompt: prompt,
                            metadata: metadata
                        });
                        
                        console.log('Fallback: Image added successfully');
                    };
                    img.onerror = () => {
                        console.error('Fallback: Failed to load image:', imagePath);
                    };
                    img.src = imagePath;
                },
                
                setTool: function(tool) {
                    console.log('Fallback: Setting tool to', tool);
                },
                
                fitToScreen: function() {
                    console.log('Fallback: Fit to screen');
                },
                
                clear: function() {
                    console.log('Fallback: Clearing canvas');
                    const canvas = document.getElementById('fooocus-canvas');
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    this.images.clear();
                    this.selectedImages.clear();
                },
                
                saveCanvas: function() {
                    console.log('Fallback: Saving canvas');
                },
                
                addImageFromFile: function(file, x, y) {
                    console.log('Fallback: Adding image from file', file.name);
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        this.addImage(e.target.result, `Imported: ${file.name}`, {
                            source: 'file',
                            filename: file.name
                        });
                    };
                    reader.readAsDataURL(file);
                }
            };
            
            // Add any pending images
            if (window.pendingCanvasImages && window.pendingCanvasImages.length > 0) {
                console.log('Fallback: Adding pending images:', window.pendingCanvasImages.length);
                window.pendingCanvasImages.forEach(img => {
                    window.fooocusCanvas.addImage(img.path, img.prompt, img.metadata);
                });
                window.pendingCanvasImages = [];
            }
            
            console.log('Fallback canvas created successfully!');
        }
        
        // Initialize immediately and on DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeCanvas);
        } else {
            initializeCanvas();
        }
        
        // Also try to initialize after a short delay in case DOM changes
        setTimeout(initializeCanvas, 500);
    </script>
    """