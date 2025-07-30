// Canvas-based image management system for Fooocus
class FooocusCanvas {
    constructor(canvasId, controlsId) {
        this.canvas = document.getElementById(canvasId);
        this.controls = document.getElementById(controlsId);
        this.ctx = this.canvas.getContext('2d');
        this.images = new Map(); // Store canvas images with metadata
        this.selectedImages = new Set();
        this.isDragging = false;
        this.dragStartPos = { x: 0, y: 0 };
        this.dragTarget = null;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.nextImageId = 1;
        this.gridSpacing = 20;
        this.imageSize = { width: 256, height: 256 };
        
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
        this.resizeCanvas();
        this.render();
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('wheel', this.onWheel.bind(this));
        this.canvas.addEventListener('contextmenu', this.onContextMenu.bind(this));
        this.canvas.addEventListener('dblclick', this.onDoubleClick.bind(this));
        
        // Window resize
        window.addEventListener('resize', this.resizeCanvas.bind(this));
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch(e.key) {
                case 'Delete':
                    this.deleteSelectedImages();
                    break;
                case 'a':
                    if (e.ctrlKey) {
                        e.preventDefault();
                        this.selectAllImages();
                    }
                    break;
                case 's':
                    if (e.ctrlKey) {
                        e.preventDefault();
                        this.saveCanvas();
                    }
                    break;
                case 'z':
                    if (e.ctrlKey) {
                        e.preventDefault();
                        this.undo();
                    }
                    break;
                case ' ':
                    e.preventDefault();
                    this.isPanning = true;
                    this.canvas.style.cursor = 'grab';
                    break;
            }
        });

        document.addEventListener('keyup', (e) => {
            if (e.key === ' ') {
                this.isPanning = false;
                this.canvas.style.cursor = 'default';
            }
        });
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.render();
    }

    screenToCanvas(screenX, screenY) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (screenX - rect.left - this.panX) / this.zoom,
            y: (screenY - rect.top - this.panY) / this.zoom
        };
    }

    canvasToScreen(canvasX, canvasY) {
        return {
            x: canvasX * this.zoom + this.panX,
            y: canvasY * this.zoom + this.panY
        };
    }

    onMouseDown(e) {
        const canvasPos = this.screenToCanvas(e.clientX, e.clientY);
        const clickedImage = this.getImageAt(canvasPos.x, canvasPos.y);

        if (this.isPanning || e.button === 1) { // Middle mouse or space+click
            this.isDragging = true;
            this.dragStartPos = { x: e.clientX, y: e.clientY };
            this.canvas.style.cursor = 'grabbing';
            return;
        }

        if (clickedImage) {
            if (!e.ctrlKey) {
                this.selectedImages.clear();
            }
            this.selectedImages.add(clickedImage.id);
            this.dragTarget = clickedImage;
            this.isDragging = true;
            this.dragStartPos = { x: e.clientX - clickedImage.x * this.zoom, y: e.clientY - clickedImage.y * this.zoom };
        } else {
            this.selectedImages.clear();
        }
        
        this.render();
    }

    onMouseMove(e) {
        if (!this.isDragging) return;

        if (this.isPanning || this.dragTarget === null) {
            // Pan the canvas
            this.panX = e.clientX - this.dragStartPos.x;
            this.panY = e.clientY - this.dragStartPos.y;
        } else if (this.dragTarget) {
            // Move selected images
            const newX = (e.clientX - this.dragStartPos.x) / this.zoom;
            const newY = (e.clientY - this.dragStartPos.y) / this.zoom;
            
            const deltaX = newX - this.dragTarget.x;
            const deltaY = newY - this.dragTarget.y;
            
            // Move all selected images
            this.selectedImages.forEach(id => {
                const img = this.images.get(id);
                if (img) {
                    img.x += deltaX;
                    img.y += deltaY;
                }
            });
        }
        
        this.render();
    }

    onMouseUp(e) {
        this.isDragging = false;
        this.dragTarget = null;
        this.canvas.style.cursor = this.isPanning ? 'grab' : 'default';
    }

    onWheel(e) {
        e.preventDefault();
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(0.1, Math.min(5, this.zoom * zoomFactor));
        
        // Zoom towards mouse position
        this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
        this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
        this.zoom = newZoom;
        
        this.render();
    }

    onContextMenu(e) {
        e.preventDefault();
        const canvasPos = this.screenToCanvas(e.clientX, e.clientY);
        const clickedImage = this.getImageAt(canvasPos.x, canvasPos.y);
        
        if (clickedImage) {
            this.showContextMenu(e.clientX, e.clientY, clickedImage);
        }
    }

    onDoubleClick(e) {
        const canvasPos = this.screenToCanvas(e.clientX, e.clientY);
        const clickedImage = this.getImageAt(canvasPos.x, canvasPos.y);
        
        if (clickedImage) {
            this.showFullResolution(clickedImage);
        }
    }

    getImageAt(x, y) {
        for (let [id, img] of this.images) {
            if (x >= img.x && x <= img.x + img.width &&
                y >= img.y && y <= img.y + img.height) {
                return img;
            }
        }
        return null;
    }

    addImage(imageData, prompt, metadata = {}) {
        const img = new Image();
        img.onload = () => {
            const position = this.findNextPosition();
            const canvasImage = {
                id: this.nextImageId++,
                element: img,
                x: position.x,
                y: position.y,
                width: this.imageSize.width,
                height: this.imageSize.height,
                prompt: prompt,
                metadata: metadata,
                selected: false
            };
            
            this.images.set(canvasImage.id, canvasImage);
            this.render();
            
            // Notify parent about new image
            this.onImageAdded?.(canvasImage);
        };
        
        if (typeof imageData === 'string') {
            img.src = imageData;
        } else {
            // Handle File or Blob
            img.src = URL.createObjectURL(imageData);
        }
    }

    findNextPosition() {
        const cols = Math.floor((this.canvas.width / this.zoom - this.gridSpacing) / (this.imageSize.width + this.gridSpacing));
        const row = Math.floor(this.images.size / Math.max(1, cols));
        const col = this.images.size % Math.max(1, cols);
        
        return {
            x: col * (this.imageSize.width + this.gridSpacing) + this.gridSpacing,
            y: row * (this.imageSize.height + this.gridSpacing) + this.gridSpacing
        };
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Save context for transformations
        this.ctx.save();
        this.ctx.translate(this.panX, this.panY);
        this.ctx.scale(this.zoom, this.zoom);
        
        // Draw grid
        this.drawGrid();
        
        // Draw images
        for (let [id, img] of this.images) {
            this.drawImage(img);
        }
        
        this.ctx.restore();
        
        // Draw UI elements (zoom level, etc.)
        this.drawUI();
    }

    drawGrid() {
        const gridSize = 50;
        this.ctx.strokeStyle = '#f0f0f0';
        this.ctx.lineWidth = 1 / this.zoom;
        
        const startX = Math.floor(-this.panX / this.zoom / gridSize) * gridSize;
        const startY = Math.floor(-this.panY / this.zoom / gridSize) * gridSize;
        const endX = startX + (this.canvas.width / this.zoom) + gridSize;
        const endY = startY + (this.canvas.height / this.zoom) + gridSize;
        
        this.ctx.beginPath();
        for (let x = startX; x < endX; x += gridSize) {
            this.ctx.moveTo(x, startY);
            this.ctx.lineTo(x, endY);
        }
        for (let y = startY; y < endY; y += gridSize) {
            this.ctx.moveTo(startX, y);
            this.ctx.lineTo(endX, y);
        }
        this.ctx.stroke();
    }

    drawImage(img) {
        // Draw the image
        this.ctx.drawImage(img.element, img.x, img.y, img.width, img.height);
        
        // Draw selection border
        if (this.selectedImages.has(img.id)) {
            this.ctx.strokeStyle = '#007bff';
            this.ctx.lineWidth = 3 / this.zoom;
            this.ctx.strokeRect(img.x, img.y, img.width, img.height);
            
            // Draw selection handles
            const handleSize = 8 / this.zoom;
            this.ctx.fillStyle = '#007bff';
            this.ctx.fillRect(img.x - handleSize/2, img.y - handleSize/2, handleSize, handleSize);
            this.ctx.fillRect(img.x + img.width - handleSize/2, img.y - handleSize/2, handleSize, handleSize);
            this.ctx.fillRect(img.x - handleSize/2, img.y + img.height - handleSize/2, handleSize, handleSize);
            this.ctx.fillRect(img.x + img.width - handleSize/2, img.y + img.height - handleSize/2, handleSize, handleSize);
        }
    }

    drawUI() {
        // Draw zoom level
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(10, 10, 100, 30);
        this.ctx.fillStyle = 'white';
        this.ctx.font = '14px Arial';
        this.ctx.fillText(`Zoom: ${Math.round(this.zoom * 100)}%`, 15, 30);
        
        // Draw selection count
        if (this.selectedImages.size > 0) {
            this.ctx.fillStyle = 'rgba(0, 123, 255, 0.8)';
            this.ctx.fillRect(10, 50, 120, 30);
            this.ctx.fillStyle = 'white';
            this.ctx.fillText(`Selected: ${this.selectedImages.size}`, 15, 70);
        }
    }

    showContextMenu(x, y, image) {
        // Remove existing context menu
        const existingMenu = document.querySelector('.canvas-context-menu');
        if (existingMenu) {
            existingMenu.remove();
        }
        
        const menu = document.createElement('div');
        menu.className = 'canvas-context-menu';
        menu.style.cssText = `
            position: fixed;
            left: ${x}px;
            top: ${y}px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            min-width: 150px;
        `;
        
        const menuItems = [
            { text: 'Regenerate', action: () => this.regenerateImage(image) },
            { text: 'Delete', action: () => this.deleteImage(image.id) },
            { text: 'Export', action: () => this.exportImage(image) },
            { text: 'Copy Prompt', action: () => this.copyPrompt(image) },
            { text: 'Show Details', action: () => this.showImageDetails(image) }
        ];
        
        menuItems.forEach(item => {
            const menuItem = document.createElement('div');
            menuItem.textContent = item.text;
            menuItem.style.cssText = `
                padding: 8px 12px;
                cursor: pointer;
                border-bottom: 1px solid #eee;
            `;
            menuItem.addEventListener('click', () => {
                item.action();
                menu.remove();
            });
            menuItem.addEventListener('mouseenter', () => {
                menuItem.style.backgroundColor = '#f5f5f5';
            });
            menuItem.addEventListener('mouseleave', () => {
                menuItem.style.backgroundColor = 'white';
            });
            menu.appendChild(menuItem);
        });
        
        document.body.appendChild(menu);
        
        // Remove menu when clicking elsewhere
        setTimeout(() => {
            document.addEventListener('click', function removeMenu() {
                menu.remove();
                document.removeEventListener('click', removeMenu);
            });
        }, 0);
    }

    regenerateImage(image) {
        // Trigger regeneration with stored prompt and metadata
        if (window.regenerateWithPrompt) {
            window.regenerateWithPrompt(image.prompt, image.metadata);
        }
    }

    deleteImage(imageId) {
        this.images.delete(imageId);
        this.selectedImages.delete(imageId);
        this.render();
    }

    deleteSelectedImages() {
        this.selectedImages.forEach(id => {
            this.images.delete(id);
        });
        this.selectedImages.clear();
        this.render();
    }

    selectAllImages() {
        this.selectedImages.clear();
        this.images.forEach((img, id) => {
            this.selectedImages.add(id);
        });
        this.render();
    }

    exportImage(image) {
        const link = document.createElement('a');
        link.download = `fooocus_${image.id}.png`;
        link.href = image.element.src;
        link.click();
    }

    copyPrompt(image) {
        navigator.clipboard.writeText(image.prompt);
    }

    showImageDetails(image) {
        // Show modal with image details
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2000;
        `;
        
        const content = document.createElement('div');
        content.style.cssText = `
            background: white;
            padding: 20px;
            border-radius: 8px;
            max-width: 80%;
            max-height: 80%;
            overflow: auto;
        `;
        
        content.innerHTML = `
            <h3>Image Details</h3>
            <p><strong>Prompt:</strong> ${image.prompt}</p>
            <p><strong>Size:</strong> ${image.width} x ${image.height}</p>
            <pre>${JSON.stringify(image.metadata, null, 2)}</pre>
            <button onclick="this.closest('.modal').remove()">Close</button>
        `;
        
        modal.className = 'modal';
        modal.appendChild(content);
        document.body.appendChild(modal);
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    showFullResolution(image) {
        // Show full resolution image in modal
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2000;
        `;
        
        const img = document.createElement('img');
        img.src = image.element.src;
        img.style.cssText = `
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
        `;
        
        modal.appendChild(img);
        document.body.appendChild(modal);
        
        modal.addEventListener('click', () => {
            modal.remove();
        });
    }

    fitToScreen() {
        if (this.images.size === 0) return;
        
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        this.images.forEach(img => {
            minX = Math.min(minX, img.x);
            minY = Math.min(minY, img.y);
            maxX = Math.max(maxX, img.x + img.width);
            maxY = Math.max(maxY, img.y + img.height);
        });
        
        const contentWidth = maxX - minX;
        const contentHeight = maxY - minY;
        const padding = 50;
        
        const scaleX = (this.canvas.width - padding * 2) / contentWidth;
        const scaleY = (this.canvas.height - padding * 2) / contentHeight;
        this.zoom = Math.min(scaleX, scaleY, 1);
        
        this.panX = (this.canvas.width - contentWidth * this.zoom) / 2 - minX * this.zoom;
        this.panY = (this.canvas.height - contentHeight * this.zoom) / 2 - minY * this.zoom;
        
        this.render();
    }

    saveCanvas() {
        const canvasState = {
            images: Array.from(this.images.entries()).map(([id, img]) => ({
                id,
                x: img.x,
                y: img.y,
                width: img.width,
                height: img.height,
                prompt: img.prompt,
                metadata: img.metadata,
                src: img.element.src
            })),
            zoom: this.zoom,
            panX: this.panX,
            panY: this.panY
        };
        
        localStorage.setItem('fooocus_canvas_state', JSON.stringify(canvasState));
    }

    loadCanvas() {
        const saved = localStorage.getItem('fooocus_canvas_state');
        if (!saved) return;
        
        try {
            const state = JSON.parse(saved);
            this.zoom = state.zoom || 1;
            this.panX = state.panX || 0;
            this.panY = state.panY || 0;
            
            state.images.forEach(imgData => {
                const img = new Image();
                img.onload = () => {
                    const canvasImage = {
                        id: imgData.id,
                        element: img,
                        x: imgData.x,
                        y: imgData.y,
                        width: imgData.width,
                        height: imgData.height,
                        prompt: imgData.prompt,
                        metadata: imgData.metadata
                    };
                    this.images.set(imgData.id, canvasImage);
                    this.nextImageId = Math.max(this.nextImageId, imgData.id + 1);
                    this.render();
                };
                img.src = imgData.src;
            });
        } catch (e) {
            console.error('Failed to load canvas state:', e);
        }
    }

    undo() {
        // Simple undo implementation - could be enhanced with proper history
        console.log('Undo functionality - to be implemented');
    }

    clear() {
        this.images.clear();
        this.selectedImages.clear();
        this.render();
    }
}

// Initialize canvas when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('fooocus-canvas')) {
        window.fooocusCanvas = new FooocusCanvas('fooocus-canvas', 'canvas-controls');
        window.fooocusCanvas.loadCanvas();
        
        // Auto-save every 30 seconds
        setInterval(() => {
            window.fooocusCanvas.saveCanvas();
        }, 30000);
    }
});