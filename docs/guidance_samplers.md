# Advanced Guidance Samplers for ChaoticFooocus

This document explains the new guidance sampling system that integrates TPG, NAG, and PAG with the existing Fooocus infrastructure.

## Overview

The guidance samplers provide advanced control over the diffusion process by modifying how the UNet processes conditioning information. Instead of using complex pipeline implementations, these samplers work by temporarily modifying the UNet's forward method during sampling.

## Available Guidance Methods

### 1. TPG (Token Perturbation Guidance)

- **Purpose**: Enhances image quality by perturbing token embeddings
- **How it works**: Creates shuffled versions of text embeddings and uses the difference to guide generation
- **Parameters**:
  - `tpg_scale`: Guidance strength (default: 3.0)
  - `tpg_applied_layers_index`: Which layers to apply TPG to (optional)

### 2. NAG (Negative Attention Guidance)

- **Purpose**: Restores effective negative prompting in few-step models where CFG fails
- **How it works**: Creates degraded positive conditioning to enhance negative prompt influence
- **Key Benefits**:
  - Enables direct suppression of visual, semantic, and stylistic attributes
  - Works with few-step models like Flux-Schnell where traditional CFG is weak
  - Supports prompt-based debiasing and enhanced controllability
  - Allows removal of specific elements (glasses, realistic style, blur, etc.)
- **Parameters**:
  - `nag_scale`: Guidance strength (default: 1.5, >1.0 to enable)
  - Values: 1.0=disabled, 1.5-2.0=moderate, 2.0-3.0=strong, 3.0+=very strong

### 3. PAG (Perturbed Attention Guidance)

- **Purpose**: Enhances generation by using perturbed attention as reference
- **How it works**: Adds controlled noise to embeddings and uses the difference for guidance
- **Parameters**:
  - `pag_scale`: Guidance strength (default: 0.0, >0.0 to enable)
  - `pag_applied_layers`: Which attention layers to apply PAG to (default: ["mid", "up"])

## Integration with Fooocus

### Sampler Architecture

Each guidance method is implemented as a sampler class that:

1. Stores the original UNet forward method
2. Replaces it with a modified version during sampling
3. Restores the original method after sampling

### Activation/Deactivation

```python
# Example usage
tpg_sampler.tpg_scale = 3.0
tpg_sampler.activate(unet)
# ... sampling happens here ...
tpg_sampler.deactivate()
```

### Multiple Guidance Methods

The system supports using multiple guidance methods simultaneously:

- TPG + NAG
- TPG + PAG  
- NAG + PAG
- TPG + NAG + PAG

## Technical Implementation

### UNet Forward Method Modification

The samplers work by:

1. Intercepting the UNet forward call
2. Duplicating the conditioning for guidance (uncond, cond, guidance_cond)
3. Processing all three through the UNet
4. Applying guidance math to combine the results
5. Returning the enhanced prediction

### Device Management

The samplers automatically handle device placement and ensure all tensors are on the correct device.

### Error Handling

Each sampler includes robust error handling and will fall back to standard behavior if guidance fails.

## Custom Samplers

The system also includes many creative custom samplers from the original sampling.py:

### Artistic Samplers

- **dreamy**: Creates dreamy, soft images with temporal blending
- **comic**: High contrast, sharp edges for comic book style
- **fractal**: Self-similar patterns at different scales
- **pixelart**: Quantized, blocky pixel art style
- **triangular**: Enhances triangular artifacts and patterns

### Experimental Samplers

- **euler_chaotic**: Adds chaotic perturbations to Euler sampling
- **euler_triangle_wave**: Oscillating noise patterns
- **euler_dreamy**: Smooth transitions with motion blur effects
- **euler_dreamy_pp**: Progressive dreamy effects with extrapolation

## Scheduler Integration

The system works with all the custom schedulers:

- **quantum**: Quantum tunneling-inspired discrete levels
- **organic**: Fibonacci-based natural growth patterns  
- **spiral**: Spiral trajectory noise scheduling
- **chaotic**: Logistic map chaos for unique patterns
- **claylike**: Heavy smoothing for sculpted effects
- And many more creative schedulers

## Usage Examples

### Basic TPG Usage

```python
# Enable TPG with default settings
tpg_enabled = True
tpg_scale = 3.0
```

### Combined Guidance

```python
# Use TPG + NAG together
tpg_enabled = True
tpg_scale = 2.5
nag_scale = 1.5
```

### With Custom Samplers

```python
# Use PAG with dreamy sampler and quantum scheduler
pag_enabled = True
pag_scale = 2.0
sampler_name = "dreamy"
scheduler_name = "quantum"
```

## Performance Considerations

- Guidance methods increase memory usage (2-3x batch size)
- Processing time increases proportionally
- Multiple guidance methods can be combined but will use more resources
- The sampler approach is more efficient than the original pipeline approach

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce guidance scales or use fewer guidance methods
2. **No Effect**: Ensure guidance scales are set correctly (>0 for PAG, >1 for NAG)
3. **Crashes**: Check that UNet architecture is compatible

### Debug Information

The system provides detailed logging:

- Activation/deactivation messages
- Guidance scale information
- Error messages with fallback behavior

## Future Enhancements

Potential improvements:

- Adaptive guidance scaling based on timestep
- Layer-specific guidance control
- Integration with ControlNet and other conditioning
- Performance optimizations for multiple guidance methods
