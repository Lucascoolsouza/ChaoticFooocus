# VSM (Vibe Score Memory) User Guide

## Overview

VSM (Vibe Score Memory) is an enhanced aesthetic preference learning system for Fooocus that remembers your liked and disliked images, then automatically steers future generations toward your preferred aesthetic style.

## Features

- **CLIP-based Learning**: Uses OpenAI's CLIP model to understand image aesthetics
- **Persistent Memory**: Stores preferences in JSON format for long-term learning
- **Category Organization**: Organize preferences by categories (portraits, landscapes, etc.)
- **Weighted Scoring**: More important preferences can have higher weights
- **Filtering & Guidance**: Automatically filter or guide generations based on preferences
- **Statistics & Analytics**: Track memory usage and effectiveness
- **Import/Export**: Share preference sets with others

## Installation

### Prerequisites

1. **Install CLIP** (Required for VSM to work):
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

2. **Verify PyTorch and PIL** are installed (usually included with Fooocus)

### Verification

Run the diagnostic script to check if VSM is properly set up:
```bash
python diagnose_vsm_simple.py
```

## How to Use VSM

### 1. Enable VSM in the UI

1. Open Fooocus
2. In the **Enhancement** section, check the **"Vibe Memory (VSM)"** checkbox
3. Select **"Vibe Memory"** from the Enhancement dropdown
4. Adjust settings:
   - **Threshold**: Score threshold for accepting images (-1.0 to 1.0)
   - **Max Retries**: How many times to retry if score is too low (1-10)

### 2. Build Your Preference Memory

#### Using the UI Buttons
- Generate images as usual
- Use the **👍 Like Current** button for images you love
- Use the **👎 Dislike Current** button for images you don't like
- VSM will learn from these preferences over time

#### Batch Import (Advanced)
You can import multiple images at once by modifying the memory file directly or using the API functions.

### 3. Let VSM Guide Your Generations

Once you have some preferences stored:
- Enable VSM filtering in the Enhancement dropdown
- Generate images normally
- VSM will automatically:
  - Score each generated image against your preferences
  - Retry generation if the score is below your threshold
  - Keep the best result within the retry limit

## Settings Explained

### Threshold (-1.0 to 1.0)
- **Positive values** (0.1 to 1.0): Only accept images similar to your likes
- **Zero** (0.0): Accept neutral images
- **Negative values** (-1.0 to -0.1): Avoid images similar to your dislikes
- **Recommended**: Start with -0.1 to 0.1

### Max Retries (1-10)
- How many times VSM will regenerate if the score is too low
- **Higher values**: Better filtering but slower generation
- **Lower values**: Faster but less selective
- **Recommended**: 3-5 retries

### Categories
VSM can organize preferences by category:
- **General**: Default category for all preferences
- **Portraits**: For character/face preferences
- **Landscapes**: For background/scenery preferences
- **Style**: For artistic style preferences
- **Custom**: Create your own categories

## Troubleshooting

### Getting 0.000 Scores

This is the most common issue. Causes and solutions:

1. **CLIP not installed**
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

2. **No memories stored**
   - Use 👍/👎 buttons to add preferences
   - Need at least 1 like or dislike to get non-zero scores

3. **All images too similar**
   - Add more diverse preferences
   - Try liking/disliking very different image styles

### VSM Not Working

1. **Check VSM is enabled**
   - ✅ "Vibe Memory (VSM)" checkbox checked
   - ✅ "Vibe Memory" selected in Enhancement dropdown

2. **Check dependencies**
   ```bash
   python diagnose_vsm_simple.py
   ```

3. **Restart application** after installing CLIP

### Memory File Issues

VSM stores preferences in `outputs/vibe_memory.json`. If corrupted:
1. Delete the file to start fresh
2. Or run the fix script: `python fix_vsm_zero_score.py`

## Advanced Usage

### Memory Management

- **View Statistics**: Check memory stats in the UI or logs
- **Clear Categories**: Remove specific preference categories
- **Export/Import**: Share preference sets
- **Optimize Memory**: Remove duplicates and update weights

### API Usage

For developers, VSM provides Python APIs:

```python
from modules.vibe_memory_integration import get_vibe_memory

# Get vibe memory instance
vibe = get_vibe_memory()

# Add preferences programmatically
vibe.add_like(image, category="portraits", weight=1.5)
vibe.add_dislike(image, category="backgrounds", weight=0.8)

# Score an image
score = vibe.score(embedding, category="portraits")

# Get detailed statistics
stats = vibe.get_statistics()
```

### Custom Categories

Organize preferences by use case:
- **Characters**: For consistent character generation
- **Environments**: For background preferences  
- **Lighting**: For lighting and mood preferences
- **Colors**: For color palette preferences

## Best Practices

### Building Good Preferences

1. **Start Small**: Add 5-10 diverse preferences initially
2. **Be Selective**: Only like/dislike images you feel strongly about
3. **Use Categories**: Organize preferences by type
4. **Regular Cleanup**: Remove outdated preferences periodically

### Optimal Settings

- **For Quality**: Higher threshold (0.2-0.5), more retries (5-8)
- **For Speed**: Lower threshold (-0.2-0.1), fewer retries (2-3)
- **For Exploration**: Neutral threshold (-0.1-0.1), moderate retries (3-5)

### Memory Maintenance

- **Monitor Size**: Keep total memories under 1000 for performance
- **Remove Duplicates**: Use the optimize function regularly
- **Update Weights**: Newer preferences can have higher weights
- **Backup**: Export your preferences regularly

## Technical Details

### How VSM Works

1. **Image Encoding**: Uses CLIP to convert images to 512-dimensional embeddings
2. **Similarity Scoring**: Compares new images to stored preferences using cosine similarity
3. **Weighted Scoring**: Combines likes (positive) and dislikes (negative) with weights
4. **Filtering**: Regenerates images that don't meet the threshold
5. **Learning**: Continuously improves as you add more preferences

### Performance Impact

- **Memory Usage**: ~1KB per stored preference
- **Generation Speed**: 10-50% slower depending on settings
- **Quality**: Significantly improved alignment with your preferences

### File Format

VSM stores data in JSON format with this structure:
```json
{
  "liked": [{"embedding": [...], "category": "...", "weight": 1.0}],
  "disliked": [{"embedding": [...], "category": "...", "weight": 1.0}],
  "metadata": {"version": "2.0", "total_likes": 10},
  "categories": {"portraits": {"liked": [0,1], "disliked": []}},
  "statistics": {"generation_count": 100, "average_score": 0.15}
}
```

## FAQ

**Q: How many preferences should I store?**
A: Start with 10-20, can scale to hundreds. More isn't always better.

**Q: Can I share my preferences with others?**
A: Yes, use the export/import functionality to share preference files.

**Q: Does VSM work with all models?**
A: Yes, VSM works at the image level, independent of the generation model.

**Q: Can I use VSM for specific styles?**
A: Absolutely! Create categories for different styles and use category-specific filtering.

**Q: How do I reset VSM?**
A: Delete `outputs/vibe_memory.json` or use the clear function in the API.

## Support

If you encounter issues:

1. Run the diagnostic: `python diagnose_vsm_simple.py`
2. Check the logs for error messages
3. Ensure CLIP is properly installed
4. Try the fix script: `python fix_vsm_zero_score.py`

For advanced users, check the comprehensive test suite: `python test_vsm_comprehensive.py`

---

*VSM enhances your Fooocus experience by learning your aesthetic preferences and automatically steering generations toward your preferred style. Happy generating!*