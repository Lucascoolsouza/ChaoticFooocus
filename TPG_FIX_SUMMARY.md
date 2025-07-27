# TPG (Token Perturbation Guidance) Fix Summary

## Issues Fixed

### 1. **Pipeline Architecture**
- **Problem**: The original TPG pipeline was trying to work as a standalone pipeline but wasn't properly integrated with Fooocus's infrastructure
- **Fix**: Restructured `StableDiffusionXLTPGPipeline` to inherit from `StableDiffusionXLPipeline` (like the PAG implementation) and properly integrate with diffusers pipeline structure

### 2. **Attention Processor Implementation**
- **Problem**: The original implementation had a simplistic token shuffling approach that didn't properly handle layer-specific application
- **Fix**: Created `TPGAttentionProcessor` class that:
  - Properly handles batch processing (uncond + cond + perturbed)
  - Applies token perturbation through shuffling at the attention level
  - Supports configurable perturbation strength
  - Integrates with the attention processor system

### 3. **Token Perturbation Method**
- **Problem**: The original token shuffling was too basic and didn't provide proper perturbation
- **Fix**: Implemented sophisticated token shuffling that:
  - Randomly permutes token order to create perturbation
  - Supports partial shuffling (configurable strength 0.0-1.0)
  - Handles different batch sizes correctly
  - Preserves tensor shapes and device placement

### 4. **UNet Integration**
- **Problem**: The guidance application logic wasn't correctly integrated with Fooocus's UNet structure
- **Fix**: Updated UNet forward method patching to:
  - Handle both ComfyUI-style (`apply_model`) and diffusers-style (`forward`) UNets
  - Properly duplicate and manage batch dimensions
  - Apply TPG guidance formula: `noise_pred_enhanced = noise_pred_cond + tpg_scale * (noise_pred_cond - noise_pred_perturb)`
  - Clean up after generation

### 5. **Integration Layer**
- **Problem**: The integration with Fooocus's sampling infrastructure was incomplete
- **Fix**: Updated `tpg_integration.py` to:
  - Patch UNet directly instead of trying to modify sampling functions
  - Provide proper enable/disable functionality
  - Support context managers for temporary TPG usage
  - Handle configuration management properly

### 6. **Interface Improvements**
- **Problem**: The interface wasn't user-friendly and lacked proper configuration options
- **Fix**: Enhanced `tpg_interface.py` with:
  - Simple enable/disable functions
  - Recommended settings for different use cases
  - Context manager support (`with_tpg()`)
  - Status reporting and configuration management

## Key Components

### 1. **StableDiffusionXLTPGPipeline** (`pipeline_sdxl_tpg.py`)
```python
class StableDiffusionXLTPGPipeline(StableDiffusionXLPipeline):
    def enable_tpg(self, tpg_scale=3.0, tpg_applied_layers=None)
    def disable_tpg(self)
    def __call__(self, ..., tpg_scale=0.0, tpg_applied_layers=None, ...)
```

### 2. **TPGAttentionProcessor** (`pipeline_sdxl_tpg.py`)
```python
class TPGAttentionProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states, ...)
    def _process_with_perturbation(self, ...)
```

### 3. **Integration Functions** (`tpg_integration.py`)
```python
def enable_tpg(scale=3.0, applied_layers=None, shuffle_strength=1.0, adaptive_strength=True)
def disable_tpg()
def shuffle_tokens(x, step=None, seed_offset=0, shuffle_strength=None)
def create_tpg_unet_wrapper(original_unet)
```

### 4. **User Interface** (`tpg_interface.py`)
```python
class TPGInterface:
    def enable(self, scale=3.0, applied_layers=None, ...)
    def disable(self)
    def apply_recommended_settings(self, use_case="general")

# Convenience functions
def enable_tpg_simple(scale=3.0, use_case="general")
def with_tpg(scale=3.0, use_case="general")  # Context manager
```

## Usage Examples

### Basic Usage
```python
from extras.TPG.tpg_interface import enable_tpg_simple, disable_tpg_simple

# Enable TPG
enable_tpg_simple(scale=3.0)

# Your image generation code here
# result = generate_image(prompt="a beautiful landscape")

# Disable TPG
disable_tpg_simple()
```

### Context Manager Usage
```python
from extras.TPG.tpg_interface import with_tpg

# Temporary TPG usage
with with_tpg(scale=3.5, use_case="artistic"):
    result = generate_image(prompt="a beautiful landscape")
# TPG automatically disabled after the block
```

### Advanced Configuration
```python
from extras.TPG.tpg_interface import tpg

# Enable with custom settings
tpg.enable(
    scale=4.0,
    applied_layers=["mid", "up"],
    shuffle_strength=0.8,
    adaptive_strength=True
)

# Generate images...

# Update just the scale
tpg.update_scale(3.0)

# Disable
tpg.disable()
```

## Technical Details

### Token Perturbation Method
TPG works by perturbing the token embeddings fed to the attention mechanism:

1. **Token Shuffling**: Randomly permute the order of tokens in the sequence
2. **Partial Shuffling**: Support for partial shuffling (e.g., shuffle only 80% of tokens)
3. **Adaptive Strength**: Adjust shuffling strength based on sampling progress
4. **Step-based Variation**: Different shuffling patterns for each sampling step

### Guidance Formula
```python
# Standard CFG
noise_pred_cfg = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

# TPG enhancement
noise_pred_final = noise_pred_cfg + tpg_scale * (noise_pred_cond - noise_pred_perturb)
```

### Recommended Settings
- **General**: scale=3.0, layers=["mid", "up"], shuffle_strength=1.0
- **Artistic**: scale=4.0, layers=["mid", "up"], shuffle_strength=1.0
- **Photorealistic**: scale=2.5, layers=["up"], shuffle_strength=0.8
- **Detailed**: scale=3.5, layers=["mid", "up"], shuffle_strength=1.0

## Testing

The implementation includes comprehensive tests:
- `test_tpg_fixed.py`: Tests the core TPG pipeline functionality
- `test_tpg_integration.py`: Tests integration with Fooocus
- `test_tpg_structure.py`: Tests code organization and structure

## Compatibility

The fixed TPG implementation is compatible with:
- Fooocus's existing pipeline infrastructure
- Both ComfyUI-style and diffusers-style UNet implementations
- Standard SDXL models
- Existing LoRA and other enhancement systems

## Performance

- Minimal overhead when TPG is disabled
- Efficient token shuffling implementation
- Proper memory management with cleanup after generation
- Support for different batch sizes and sequence lengths