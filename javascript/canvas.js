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
        
        // Outpainting and framing tools
        this.framingMode = false;
        this.frameRect = null;
        this.outpaintMode = false;
        this.detectedEmptyAreas = [];
        this.currentTool = 'select'; // 'select', 'frame', 'outpaint'
        
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
        
        // Drag and drop events
        this.canvas.addEventListener('dragover', this.onDragOver.bind(this));
        this.canvas.addEventListener('drop', this.onDrop.bind(this));
        this.canvas.addEventListener('dragenter', this.onDragEnter.bind(this));
        this.canvas.addEventListener('dragleave', this.onDragLeave.bind(this));
        
        // Paste events
        document.addEventListener('paste', this.onPaste.bind(this));
        
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

        // Handle different tool modes
        if (this.currentTool === 'frame') {
            this.startFraming(canvasPos.x, canvasPos.y);
            return;
        }

        if (this.currentTool === 'outpaint') {
            this.detectOutpaintArea(canvasPos.x, canvasPos.y);
            return;
        }

        // Default selection behavior
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
        const canvasPos = this.screenToCanvas(e.clientX, e.clientY);

        // Handle framing mode
        if (this.framingMode && this.frameRect) {
            this.updateFrame(canvasPos.x, canvasPos.y);
            this.render();
            return;
        }

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
        // Handle framing mode completion
        if (this.framingMode && this.frameRect) {
            this.completeFrame();
            return;
        }

        this.isDragging = false;
        this.dragTarget = null;
        this.canvas.style.cursor = this.isPanning ? 'grab' : this.getToolCursor();
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

    onDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        
        // Add visual feedback
        this.canvas.style.backgroundColor = 'rgba(0, 123, 255, 0.1)';
        this.canvas.style.border = '2px dashed #007bff';
    }

    onDragEnter(e) {
        e.preventDefault();
        this.dragCounter = (this.dragCounter || 0) + 1;
    }

    onDragLeave(e) {
        e.preventDefault();
        this.dragCounter = (this.dragCounter || 1) - 1;
        
        if (this.dragCounter === 0) {
            // Remove visual feedback
            this.canvas.style.backgroundColor = '';
            this.canvas.style.border = '';
        }
    }

    onDrop(e) {
        e.preventDefault();
        this.dragCounter = 0;
        
        // Remove visual feedback
        this.canvas.style.backgroundColor = '';
        this.canvas.style.border = '';
        
        const files = Array.from(e.dataTransfer.files);
        const imageFiles = files.filter(file => file.type.startsWith('image/'));
        
        if (imageFiles.length > 0) {
            imageFiles.forEach(file => {
                this.addImageFromFile(file, e.clientX, e.clientY);
            });
        } else {
            // Check for image URLs or HTML content
            const html = e.dataTransfer.getData('text/html');
            const text = e.dataTransfer.getData('text/plain');
            
            if (html) {
                this.extractImagesFromHTML(html, e.clientX, e.clientY);
            } else if (text && this.isImageURL(text)) {
                this.addImageFromURL(text, e.clientX, e.clientY);
            }
        }
    }

    onPaste(e) {
        // Only handle paste when canvas is focused or no input is focused
        const activeElement = document.activeElement;
        const isInputFocused = activeElement && (
            activeElement.tagName === 'INPUT' || 
            activeElement.tagName === 'TEXTAREA' || 
            activeElement.contentEditable === 'true'
        );
        
        if (isInputFocused) return;
        
        e.preventDefault();
        
        const items = Array.from(e.clipboardData.items);
        const imageItems = items.filter(item => item.type.startsWith('image/'));
        
        if (imageItems.length > 0) {
            imageItems.forEach(item => {
                const file = item.getAsFile();
                if (file) {
                    // Paste at center of canvas
                    const centerX = this.canvas.width / 2;
                    const centerY = this.canvas.height / 2;
                    this.addImageFromFile(file, centerX, centerY);
                }
            });
        } else {
            // Check for text content that might be an image URL
            const text = e.clipboardData.getData('text/plain');
            const html = e.clipboardData.getData('text/html');
            
            if (html) {
                const centerX = this.canvas.width / 2;
                const centerY = this.canvas.height / 2;
                this.extractImagesFromHTML(html, centerX, centerY);
            } else if (text && this.isImageURL(text)) {
                const centerX = this.canvas.width / 2;
                const centerY = this.canvas.height / 2;
                this.addImageFromURL(text, centerX, centerY);
            }
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

    addImageFromFile(file, screenX, screenY) {
        const img = new Image();
        img.onload = () => {
            const canvasPos = this.screenToCanvas(screenX, screenY);
            
            // Calculate size maintaining aspect ratio
            const maxSize = Math.max(this.imageSize.width, this.imageSize.height);
            const aspectRatio = img.naturalWidth / img.naturalHeight;
            let width, height;
            
            if (aspectRatio > 1) {
                width = maxSize;
                height = maxSize / aspectRatio;
            } else {
                width = maxSize * aspectRatio;
                height = maxSize;
            }
            
            const canvasImage = {
                id: this.nextImageId++,
                element: img,
                x: canvasPos.x - width / 2,
                y: canvasPos.y - height / 2,
                width: width,
                height: height,
                prompt: `Imported: ${file.name}`,
                metadata: {
                    source: 'file',
                    filename: file.name,
                    size: file.size,
                    type: file.type,
                    imported: new Date().toISOString()
                },
                selected: false
            };
            
            this.images.set(canvasImage.id, canvasImage);
            this.render();
            
            // Show notification
            this.showNotification(`Added image: ${file.name}`);
            
            // Notify parent about new image
            this.onImageAdded?.(canvasImage);
        };
        
        img.onerror = () => {
            this.showNotification(`Failed to load image: ${file.name}`, 'error');
        };
        
        img.src = URL.createObjectURL(file);
    }

    addImageFromURL(url, screenX, screenY) {
        const img = new Image();
        img.crossOrigin = 'anonymous'; // Try to load with CORS
        
        img.onload = () => {
            const canvasPos = this.screenToCanvas(screenX, screenY);
            
            // Calculate size maintaining aspect ratio
            const maxSize = Math.max(this.imageSize.width, this.imageSize.height);
            const aspectRatio = img.naturalWidth / img.naturalHeight;
            let width, height;
            
            if (aspectRatio > 1) {
                width = maxSize;
                height = maxSize / aspectRatio;
            } else {
                width = maxSize * aspectRatio;
                height = maxSize;
            }
            
            const canvasImage = {
                id: this.nextImageId++,
                element: img,
                x: canvasPos.x - width / 2,
                y: canvasPos.y - height / 2,
                width: width,
                height: height,
                prompt: `Imported from URL`,
                metadata: {
                    source: 'url',
                    url: url,
                    imported: new Date().toISOString()
                },
                selected: false
            };
            
            this.images.set(canvasImage.id, canvasImage);
            this.render();
            
            // Show notification
            this.showNotification(`Added image from URL`);
            
            // Notify parent about new image
            this.onImageAdded?.(canvasImage);
        };
        
        img.onerror = () => {
            this.showNotification(`Failed to load image from URL`, 'error');
        };
        
        img.src = url;
    }

    extractImagesFromHTML(html, screenX, screenY) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const images = doc.querySelectorAll('img');
        
        images.forEach((imgElement, index) => {
            const src = imgElement.src;
            if (src && this.isImageURL(src)) {
                // Offset multiple images slightly
                const offsetX = index * 20;
                const offsetY = index * 20;
                this.addImageFromURL(src, screenX + offsetX, screenY + offsetY);
            }
        });
        
        if (images.length === 0) {
            this.showNotification('No images found in the content', 'warning');
        }
    }

    isImageURL(url) {
        if (!url || typeof url !== 'string') return false;
        
        // Check for common image extensions
        const imageExtensions = /\.(jpg|jpeg|png|gif|bmp|webp|svg)(\?.*)?$/i;
        if (imageExtensions.test(url)) return true;
        
        // Check for data URLs
        if (url.startsWith('data:image/')) return true;
        
        // Check for blob URLs
        if (url.startsWith('blob:')) return true;
        
        return false;
    }

    showNotification(message, type = 'info') {
        // Remove existing notification
        const existingNotification = document.querySelector('.canvas-notification');
        if (existingNotification) {
            existingNotification.remove();
        }
        
        const notification = document.createElement('div');
        notification.className = 'canvas-notification';
        notification.textContent = message;
        
        const colors = {
            info: '#007bff',
            success: '#28a745',
            warning: '#ffc107',
            error: '#dc3545'
        };
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${colors[type] || colors.info};
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 2000;
            font-size: 14px;
            max-width: 300px;
            word-wrap: break-word;
            animation: slideIn 0.3s ease-out;
        `;
        
        // Add animation styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }, 3000);
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
        
        // Draw active frame
        if (this.framingMode && this.frameRect) {
            this.drawFrame();
        }
        
        // Draw detected empty areas
        this.drawEmptyAreas();
        
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
        // Handle generation indicators
        if (img.isGenerating) {
            this.ctx.strokeStyle = '#007bff';
            this.ctx.setLineDash([10, 5]);
            this.ctx.lineWidth = 2 / this.zoom;
            this.ctx.strokeRect(img.x, img.y, img.width, img.height);
            
            this.ctx.fillStyle = 'rgba(0, 123, 255, 0.1)';
            this.ctx.fillRect(img.x, img.y, img.width, img.height);
            
            // Draw loading text
            this.ctx.fillStyle = '#007bff';
            this.ctx.font = `${16 / this.zoom}px Arial`;
            this.ctx.textAlign = 'center';
            this.ctx.fillText('⏳ Generating...', 
                img.x + img.width / 2, 
                img.y + img.height / 2);
            
            this.ctx.setLineDash([]);
            return;
        }
        
        // Draw the actual image
        if (img.element) {
            this.ctx.drawImage(img.element, img.x, img.y, img.width, img.height);
        }
        
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

    drawFrame() {
        if (!this.frameRect) return;
        
        const rect = this.normalizeRect(this.frameRect);
        
        // Draw frame rectangle
        this.ctx.strokeStyle = '#ff6b6b';
        this.ctx.setLineDash([5, 5]);
        this.ctx.lineWidth = 2 / this.zoom;
        this.ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
        
        // Draw frame background
        this.ctx.fillStyle = 'rgba(255, 107, 107, 0.1)';
        this.ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
        
        this.ctx.setLineDash([]);
    }

    drawEmptyAreas() {
        this.detectedEmptyAreas.forEach(area => {
            this.ctx.strokeStyle = '#28a745';
            this.ctx.setLineDash([3, 3]);
            this.ctx.lineWidth = 1 / this.zoom;
            this.ctx.strokeRect(area.x, area.y, area.width, area.height);
            
            this.ctx.fillStyle = 'rgba(40, 167, 69, 0.1)';
            this.ctx.fillRect(area.x, area.y, area.width, area.height);
        });
        this.ctx.setLineDash([]);
    }

    drawUI() {
        // Draw zoom level
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(10, 10, 100, 30);
        this.ctx.fillStyle = 'white';
        this.ctx.font = '14px Arial';
        this.ctx.fillText(`Zoom: ${Math.round(this.zoom * 100)}%`, 15, 30);
        
        // Draw current tool
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(10, 50, 120, 30);
        this.ctx.fillStyle = 'white';
        this.ctx.fillText(`Tool: ${this.currentTool}`, 15, 70);
        
        // Draw selection count
        if (this.selectedImages.size > 0) {
            this.ctx.fillStyle = 'rgba(0, 123, 255, 0.8)';
            this.ctx.fillRect(10, 90, 120, 30);
            this.ctx.fillStyle = 'white';
            this.ctx.fillText(`Selected: ${this.selectedImages.size}`, 15, 110);
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

    // Framing functionality
    setTool(tool) {
        this.currentTool = tool;
        this.canvas.style.cursor = this.getToolCursor();
        
        // Reset modes
        this.framingMode = false;
        this.frameRect = null;
        this.detectedEmptyAreas = [];
        
        this.render();
        this.updateToolButtons();
    }

    getToolCursor() {
        switch(this.currentTool) {
            case 'frame': return 'crosshair';
            case 'outpaint': return 'cell';
            default: return 'default';
        }
    }

    updateToolButtons() {
        // Update tool button states
        document.querySelectorAll('.canvas-tool-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const activeBtn = document.querySelector(`[data-tool="${this.currentTool}"]`);
        if (activeBtn) {
            activeBtn.classList.add('active');
        }
    }

    startFraming(x, y) {
        this.framingMode = true;
        this.frameRect = {
            startX: x,
            startY: y,
            endX: x,
            endY: y
        };
        this.render();
    }

    updateFrame(x, y) {
        if (this.frameRect) {
            this.frameRect.endX = x;
            this.frameRect.endY = y;
        }
    }

    completeFrame() {
        if (!this.frameRect) return;

        const rect = this.normalizeRect(this.frameRect);
        
        // Check if frame contains any images
        const imagesInFrame = this.getImagesInRect(rect);
        
        if (imagesInFrame.length > 0) {
            // Show frame options
            this.showFrameOptions(rect, imagesInFrame);
        } else {
            // Empty area - suggest outpainting
            this.suggestOutpaint(rect);
        }

        this.framingMode = false;
        this.frameRect = null;
        this.render();
    }

    normalizeRect(rect) {
        return {
            x: Math.min(rect.startX, rect.endX),
            y: Math.min(rect.startY, rect.endY),
            width: Math.abs(rect.endX - rect.startX),
            height: Math.abs(rect.endY - rect.startY)
        };
    }

    getImagesInRect(rect) {
        const imagesInFrame = [];
        this.images.forEach(img => {
            if (this.rectIntersects(rect, {
                x: img.x, y: img.y, 
                width: img.width, height: img.height
            })) {
                imagesInFrame.push(img);
            }
        });
        return imagesInFrame;
    }

    rectIntersects(rect1, rect2) {
        return !(rect1.x + rect1.width < rect2.x || 
                rect2.x + rect2.width < rect1.x || 
                rect1.y + rect1.height < rect2.y || 
                rect2.y + rect2.height < rect1.y);
    }

    showFrameOptions(rect, images) {
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
            max-width: 400px;
        `;
        
        content.innerHTML = `
            <h3>Frame Options</h3>
            <p>Found ${images.length} image(s) in frame</p>
            <div style="display: flex; gap: 10px; margin-top: 15px;">
                <button onclick="window.fooocusCanvas.cropToFrame(${JSON.stringify(rect)}, ${JSON.stringify(images.map(i => i.id))}); this.closest('.frame-modal').remove();">
                    Crop Images
                </button>
                <button onclick="window.fooocusCanvas.inpaintFrame(${JSON.stringify(rect)}, ${JSON.stringify(images.map(i => i.id))}); this.closest('.frame-modal').remove();">
                    Inpaint Area
                </button>
                <button onclick="window.fooocusCanvas.upscaleFrame(${JSON.stringify(rect)}, ${JSON.stringify(images.map(i => i.id))}); this.closest('.frame-modal').remove();">
                    Upscale Area
                </button>
                <button onclick="this.closest('.frame-modal').remove();">Cancel</button>
            </div>
        `;
        
        modal.className = 'frame-modal';
        modal.appendChild(content);
        document.body.appendChild(modal);
    }

    suggestOutpaint(rect) {
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
            max-width: 400px;
        `;
        
        content.innerHTML = `
            <h3>Empty Area Detected</h3>
            <p>This area appears to be empty. Would you like to:</p>
            <div style="display: flex; gap: 10px; margin-top: 15px;">
                <button onclick="window.fooocusCanvas.outpaintArea(${JSON.stringify(rect)}); this.closest('.outpaint-modal').remove();">
                    Generate Content
                </button>
                <button onclick="window.fooocusCanvas.extendNearbyImages(${JSON.stringify(rect)}); this.closest('.outpaint-modal').remove();">
                    Extend Nearby Images
                </button>
                <button onclick="this.closest('.outpaint-modal').remove();">Cancel</button>
            </div>
        `;
        
        modal.className = 'outpaint-modal';
        modal.appendChild(content);
        document.body.appendChild(modal);
    }

    // Outpainting functionality
    detectOutpaintArea(x, y) {
        // Find the nearest image to the click point
        let nearestImage = null;
        let minDistance = Infinity;
        
        this.images.forEach(img => {
            const centerX = img.x + img.width / 2;
            const centerY = img.y + img.height / 2;
            const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
            
            if (distance < minDistance) {
                minDistance = distance;
                nearestImage = img;
            }
        });
        
        if (nearestImage) {
            this.showOutpaintOptions(nearestImage, x, y);
        }
    }

    showOutpaintOptions(image, clickX, clickY) {
        // Determine which side of the image was clicked
        const imgCenterX = image.x + image.width / 2;
        const imgCenterY = image.y + image.height / 2;
        
        const directions = [];
        if (clickX < image.x) directions.push('left');
        if (clickX > image.x + image.width) directions.push('right');
        if (clickY < image.y) directions.push('top');
        if (clickY > image.y + image.height) directions.push('bottom');
        
        if (directions.length === 0) {
            // Click was inside image - show all directions
            directions.push('left', 'right', 'top', 'bottom');
        }
        
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
            max-width: 400px;
        `;
        
        const directionButtons = directions.map(dir => 
            `<button onclick="window.fooocusCanvas.performOutpaint(${image.id}, '${dir}'); this.closest('.outpaint-modal').remove();">
                Outpaint ${dir.charAt(0).toUpperCase() + dir.slice(1)}
            </button>`
        ).join('');
        
        content.innerHTML = `
            <h3>Outpaint Options</h3>
            <p>Extend image in which direction?</p>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
                ${directionButtons}
                <button onclick="this.closest('.outpaint-modal').remove();">Cancel</button>
            </div>
        `;
        
        modal.className = 'outpaint-modal';
        modal.appendChild(content);
        document.body.appendChild(modal);
    }

    performOutpaint(imageId, direction) {
        const image = this.images.get(imageId);
        if (!image) return;
        
        // Calculate outpaint area based on direction
        let outpaintRect;
        const expansion = 256; // pixels to expand
        
        switch(direction) {
            case 'left':
                outpaintRect = {
                    x: image.x - expansion,
                    y: image.y,
                    width: expansion + 50, // overlap
                    height: image.height
                };
                break;
            case 'right':
                outpaintRect = {
                    x: image.x + image.width - 50, // overlap
                    y: image.y,
                    width: expansion + 50,
                    height: image.height
                };
                break;
            case 'top':
                outpaintRect = {
                    x: image.x,
                    y: image.y - expansion,
                    width: image.width,
                    height: expansion + 50 // overlap
                };
                break;
            case 'bottom':
                outpaintRect = {
                    x: image.x,
                    y: image.y + image.height - 50, // overlap
                    width: image.width,
                    height: expansion + 50
                };
                break;
        }
        
        // Trigger outpaint generation
        if (window.triggerOutpaint) {
            window.triggerOutpaint(image, outpaintRect, direction);
        }
        
        // Show generation indicator
        this.showGenerationIndicator(outpaintRect);
    }

    showGenerationIndicator(rect) {
        // Add a visual indicator for the generation area
        const indicator = {
            id: 'generating-' + Date.now(),
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            isGenerating: true
        };
        
        this.images.set(indicator.id, indicator);
        this.render();
        
        // Remove indicator after timeout (will be replaced by actual image)
        setTimeout(() => {
            this.images.delete(indicator.id);
            this.render();
        }, 30000);
    }

    cropToFrame(rect, imageIds) {
        // Implement cropping functionality
        console.log('Cropping images to frame:', rect, imageIds);
    }

    inpaintFrame(rect, imageIds) {
        // Trigger inpainting for the framed area
        if (window.triggerInpaint) {
            window.triggerInpaint(rect, imageIds);
        }
    }

    upscaleFrame(rect, imageIds) {
        // Trigger upscaling for the framed area
        if (window.triggerUpscale) {
            window.triggerUpscale(rect, imageIds);
        }
    }

    outpaintArea(rect) {
        // Generate content for empty area
        if (window.triggerGeneration) {
            window.triggerGeneration(rect);
        }
        this.showGenerationIndicator(rect);
    }

    extendNearbyImages(rect) {
        // Find images near the rect and extend them
        const nearbyImages = [];
        this.images.forEach(img => {
            const distance = this.getDistanceToRect(img, rect);
            if (distance < 100) { // within 100 pixels
                nearbyImages.push(img);
            }
        });
        
        if (nearbyImages.length > 0) {
            // Trigger extension of nearby images
            if (window.triggerImageExtension) {
                window.triggerImageExtension(nearbyImages, rect);
            }
        }
    }

    getDistanceToRect(image, rect) {
        const imgCenterX = image.x + image.width / 2;
        const imgCenterY = image.y + image.height / 2;
        const rectCenterX = rect.x + rect.width / 2;
        const rectCenterY = rect.y + rect.height / 2;
        
        return Math.sqrt((imgCenterX - rectCenterX) ** 2 + (imgCenterY - rectCenterY) ** 2);
    }

    undo() {
        // Simple undo implementation - could be enhanced with proper history
        console.log('Undo functionality - to be implemented');
    }

    // Auto-detect empty areas for outpainting
    detectEmptyAreas() {
        this.detectedEmptyAreas = [];
        
        if (this.images.size === 0) return;
        
        // Find bounding box of all images
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        this.images.forEach(img => {
            if (img.isGenerating) return; // Skip generation indicators
            
            minX = Math.min(minX, img.x);
            minY = Math.min(minY, img.y);
            maxX = Math.max(maxX, img.x + img.width);
            maxY = Math.max(maxY, img.y + img.height);
        });
        
        // Expand the area to look for empty spaces
        const expansion = 100;
        const searchArea = {
            x: minX - expansion,
            y: minY - expansion,
            width: (maxX - minX) + (expansion * 2),
            height: (maxY - minY) + (expansion * 2)
        };
        
        // Grid-based empty area detection
        const gridSize = 64;
        const cols = Math.ceil(searchArea.width / gridSize);
        const rows = Math.ceil(searchArea.height / gridSize);
        
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const cellX = searchArea.x + col * gridSize;
                const cellY = searchArea.y + row * gridSize;
                const cellRect = {
                    x: cellX,
                    y: cellY,
                    width: gridSize,
                    height: gridSize
                };
                
                // Check if this cell overlaps with any image
                let isEmpty = true;
                this.images.forEach(img => {
                    if (img.isGenerating) return;
                    
                    if (this.rectIntersects(cellRect, {
                        x: img.x, y: img.y,
                        width: img.width, height: img.height
                    })) {
                        isEmpty = false;
                    }
                });
                
                if (isEmpty) {
                    // Check if this empty area is adjacent to any image
                    let isAdjacent = false;
                    this.images.forEach(img => {
                        if (img.isGenerating) return;
                        
                        const distance = this.getDistanceToRect(img, cellRect);
                        if (distance < gridSize * 1.5) {
                            isAdjacent = true;
                        }
                    });
                    
                    if (isAdjacent) {
                        this.detectedEmptyAreas.push(cellRect);
                    }
                }
            }
        }
        
        // Merge adjacent empty areas
        this.mergeEmptyAreas();
    }

    mergeEmptyAreas() {
        // Simple merging of adjacent empty areas
        const merged = [];
        const used = new Set();
        
        for (let i = 0; i < this.detectedEmptyAreas.length; i++) {
            if (used.has(i)) continue;
            
            let area = { ...this.detectedEmptyAreas[i] };
            used.add(i);
            
            // Try to merge with other areas
            let changed = true;
            while (changed) {
                changed = false;
                for (let j = 0; j < this.detectedEmptyAreas.length; j++) {
                    if (used.has(j)) continue;
                    
                    const other = this.detectedEmptyAreas[j];
                    
                    // Check if areas are adjacent
                    if (this.areAreasAdjacent(area, other)) {
                        // Merge areas
                        const newArea = this.mergeRects(area, other);
                        area = newArea;
                        used.add(j);
                        changed = true;
                    }
                }
            }
            
            merged.push(area);
        }
        
        this.detectedEmptyAreas = merged.filter(area => 
            area.width > 32 && area.height > 32 // Filter out tiny areas
        );
    }

    areAreasAdjacent(rect1, rect2) {
        // Check if rectangles are touching or overlapping
        const gap = 10; // Allow small gaps
        
        return !(rect1.x > rect2.x + rect2.width + gap || 
                rect2.x > rect1.x + rect1.width + gap || 
                rect1.y > rect2.y + rect2.height + gap || 
                rect2.y > rect1.y + rect1.height + gap);
    }

    mergeRects(rect1, rect2) {
        const minX = Math.min(rect1.x, rect2.x);
        const minY = Math.min(rect1.y, rect2.y);
        const maxX = Math.max(rect1.x + rect1.width, rect2.x + rect2.width);
        const maxY = Math.max(rect1.y + rect1.height, rect2.y + rect2.height);
        
        return {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        };
    }

    // Auto-detect empty areas when images change
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
            
            // Auto-detect empty areas after adding image
            setTimeout(() => {
                this.detectEmptyAreas();
                this.render();
            }, 100);
            
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

    clear() {
        this.images.clear();
        this.selectedImages.clear();
        this.detectedEmptyAreas = [];
        this.frameRect = null;
        this.render();
    }
}

// Initialize canvas when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('fooocus-canvas')) {
        window.fooocusCanvas = new FooocusCanvas('fooocus-canvas', 'canvas-controls');
        window.fooocusCanvas.loadCanvas();
        
        // Set up file input handler
        const fileInput = document.getElementById('canvas-file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                files.forEach((file, index) => {
                    if (file.type.startsWith('image/')) {
                        // Position files in a grid pattern
                        const centerX = window.fooocusCanvas.canvas.width / 2;
                        const centerY = window.fooocusCanvas.canvas.height / 2;
                        const offsetX = (index % 3 - 1) * 100;
                        const offsetY = Math.floor(index / 3) * 100;
                        
                        window.fooocusCanvas.addImageFromFile(file, centerX + offsetX, centerY + offsetY);
                    }
                });
                // Reset file input
                e.target.value = '';
            });
        }
        
        // Show paste hint when appropriate
        let pasteHintTimeout;
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'v') {
                const hint = document.getElementById('canvas-paste-hint');
                if (hint) {
                    hint.style.display = 'block';
                    clearTimeout(pasteHintTimeout);
                    pasteHintTimeout = setTimeout(() => {
                        hint.style.display = 'none';
                    }, 2000);
                }
            }
        });
        
        // Auto-save every 30 seconds
        setInterval(() => {
            window.fooocusCanvas.saveCanvas();
        }, 30000);
    }
});