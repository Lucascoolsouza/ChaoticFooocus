# Vibe Memory Integration for Fooocus

This document describes the integration of the Vibe Score Memory (VSM) system into Fooocus WebUI, allowing users to train an aesthetic memory that remembers liked and disliked images to steer future generations.

## Overview

The Vibe Memory system uses CLIP embeddings to remember the aesthetic preferences of users. When enabled, it:

1. **Remembers** liked and disliked images using CLIP embeddings
2. **Scores** new generations based on similarity to liked/disliked aesthetics  
3. **Filters** generated images to prefer those aligned with user preferences
4. **Persists** memory across sessions in a JSON file

## Features

- ✅ **Aesthetic Memory**: Remembers your visual preferences using CLIP embeddings
- ✅ **Real-time Filtering**: Automatically filters generated images based on vibe scores
- ✅ **Configurable Thresholds**: Adjust acceptance criteria and retry limits
- ✅ **Persistent Storage**: Memory saved to JSON file across sessions
- ✅ **UI Integration**: Easy-to-use controls in the Advanced settings panel
- ✅ **Graceful Fallbacks**: Works even when CLIP is not available

## Installation Requirements

### Required Dependencies
```bash
# Install CLIP for image embeddings
pip install git+https://github.com/openai/CLIP.git

# Or if you prefer the official package
pip install clip-by-openai
```

### Optional Dependencies
- PyTorch (usually already installed with Fooocus)
- PIL/Pillow (usually already installed)

## Usage

### 1. Enable Vibe Memory

1. Open Fooocus WebUI
2. Go to the **Advanced** tab
3. Expand the **Vibe Memory (VSM)** accordion
4. Check **Enable Vibe Memory**

### 2. Configure Settings

- **Acceptance Threshold** (-1.0 to 1.0): Minimum vibe score to accept images
  - Higher values = more selective (only images very similar to likes)
  - Lower values = more permissive (accepts more varied images)
  - Default: -0.1

- **Max Retries** (1-10): Maximum attempts to generate acceptable images
  - Higher values = more attempts to find good images (slower)
  - Lower values = faster generation but may accept lower-scoring images
  - Default: 3

### 3. Train Your Memory

#### Method 1: Using the UI Buttons (Planned)
- Generate images normally
- Click **👍 Like Current** for images you enjoy
- Click **👎 Dislike Current** for images you don't like

#### Method 2: Using File Paths (Current)
```python
from modules.vibe_memory_integration import add_like_from_image_path, add_dislike_from_image_path

# Add likes
add_like_from_image_path("path/to/liked_image.png")

# Add dislikes  
add_dislike_from_image_path("path/to/disliked_image.png")
```

### 4. Monitor Performance

- **Show Stats**: Click to see current memory statistics
- **Clear Memory**: Reset all learned preferences
- **Status Display**: Shows current configuration and memory state

## Technical Implementation

### File Structure

```
├── extras/VSM/
│   ├── vibe_score_memory.py      # Core VSM implementation
│   └── readme.md                 # VSM documentation
├── modules/
│   ├── vibe_memory_integration.py # Fooocus integration layer
│   ├── flags.py                  # Updated with vibe_memory flag
│   └── async_worker.py           # Updated with filtering logic
└── webui.py                      # Updated with UI components
```

### Integration Points

1. **UI Layer** (`webui.py`):
   - Vibe Memory accordion in Advanced settings
   - Like/Dislike buttons (planned)
   - Configuration sliders and status display

2. **Task Processing** (`async_worker.py`):
   - Vibe memory parameters added to AsyncTask
   - Image filtering applied after generation
   - Progress updates during filtering

3. **Core Logic** (`vibe_memory_integration.py`):
   - Interface between Fooocus and VSM
   - Memory management and persistence
   - Graceful error handling

4. **VSM Core** (`extras/VSM/vibe_score_memory.py`):
   - CLIP embedding generation
   - Cosine similarity scoring
   - JSON persistence

### Memory File Format

The vibe memory is stored in `outputs/vibe_memory.json`:

```json
{
  "liked": [
    [0.1, 0.2, -0.3, ...],  // CLIP embedding vectors
    [0.4, -0.1, 0.2, ...]
  ],
  "disliked": [
    [-0.2, 0.3, 0.1, ...],
    [0.1, -0.4, 0.2, ...]
  ]
}
```

## Configuration Options

### Environment Variables
- `VIBE_MEMORY_PATH`: Custom path for memory file (default: `outputs/vibe_memory.json`)
- `CLIP_MODEL_NAME`: CLIP model to use (default: `ViT-B/32`)

### Advanced Settings
- **Threshold Tuning**: Start with -0.1, adjust based on results
- **Retry Limits**: Balance between quality and speed
- **Memory Size**: No built-in limits, but large memories may slow scoring

## Troubleshooting

### Common Issues

1. **"CLIP not available" Warning**
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

2. **"PyTorch not available" Error**
   - Ensure PyTorch is properly installed
   - Check CUDA compatibility if using GPU

3. **Memory Not Persisting**
   - Check write permissions in outputs directory
   - Verify JSON file is not corrupted

4. **Slow Generation**
   - Reduce max retries
   - Lower threshold for less selective filtering
   - Consider smaller CLIP model (RN50 vs ViT-B/32)

### Debug Mode

Enable debug logging to see vibe memory operations:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Performance Considerations

- **Memory Usage**: Each CLIP embedding is ~2KB, so 1000 images ≈ 2MB
- **Generation Speed**: Each retry adds ~1-2 seconds depending on model size
- **CLIP Model Choice**: 
  - `RN50`: Faster, less accurate
  - `ViT-B/32`: Balanced (default)
  - `ViT-L/14`: Slower, more accurate

## Future Enhancements

- [ ] **Gallery Integration**: Like/dislike directly from image gallery
- [ ] **Batch Training**: Import multiple images at once
- [ ] **Memory Analytics**: Visualize learned preferences
- [ ] **Style Transfer**: Apply learned aesthetics to specific styles
- [ ] **Memory Sharing**: Export/import memory files
- [ ] **Advanced Filtering**: Multiple memory profiles
- [ ] **Real-time Preview**: Show vibe scores during generation

## API Reference

### Core Functions

```python
# Get vibe memory instance
from modules.vibe_memory_integration import get_vibe_memory
vibe = get_vibe_memory()

# Check if enabled for task
from modules.vibe_memory_integration import is_vibe_memory_enabled
enabled = is_vibe_memory_enabled(async_task)

# Add preferences
from modules.vibe_memory_integration import add_like_from_image_path, add_dislike_from_image_path
add_like_from_image_path("image.png")
add_dislike_from_image_path("image.png")

# Get statistics
from modules.vibe_memory_integration import get_memory_stats
stats = get_memory_stats()  # {"liked": 10, "disliked": 5, "available": True}

# Clear memory
from modules.vibe_memory_integration import clear_memory
clear_memory()
```

### VSM Core API

```python
from extras.VSM.vibe_score_memory import VibeMemory

# Initialize
vibe = VibeMemory(memory_path="memory.json")

# Add preferences
vibe.add_like(pil_image)
vibe.add_dislike(torch_tensor)

# Score images
score = vibe.score(clip_embedding)  # Returns float score
```

## Contributing

To contribute to the Vibe Memory integration:

1. **Test Changes**: Run `python test_vibe_simple.py`
2. **Follow Patterns**: Match existing Fooocus integration patterns
3. **Handle Errors**: Graceful fallbacks for missing dependencies
4. **Document**: Update this README for new features

## License

This integration follows the same license as Fooocus. The VSM core implementation may have its own license terms.