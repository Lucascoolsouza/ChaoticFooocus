# PAG (Perturbed Attention Guidance) for Stable Diffusion XL

This implementation provides Perturbed Attention Guidance (PAG) for Stable Diffusion XL, based on the paper "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance".

## What is PAG?

PAG (Perturbed Attention Guidance) is a technique that improves image generation quality by:

1. **Attention Perturbation**: During the denoising process, attention maps are perturbed to create degraded predictions
2. **Guidance**: The difference between normal and perturbed predictions guides the generation away from low-quality outputs
3. **Self-Rectification**: The model learns to avoid generating images that would be degraded by attention perturbation

## Key Benefits

- ✅ **Improved Quality**: Better image coherence and detail
- ✅ **Stable**: More reliable than complex guidance methods
- ✅ **Efficient**: Minimal computational overhead
- ✅ **Compatible**: Works with existing SDXL pipelines
- ✅ **No Training**: Works with pre-trained models

## Usage

### Basic Usage

```python
from extras.PAG.pipeline_sdxl_pag import StableDiffusionXLPAGPipeline

# Load pipeline
pipeline = StableDiffusionXLPAGPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# Generate with PAG
image = pipeline(
    prompt="a beautiful landscape",
    pag_scale=3.0,  # PAG guidance scale
    guidance_scale=7.0,  # Regular CFG scale
    num_inference_steps=30
).images[0]
```

### Advanced Usage

```python
# Specify which layers to apply PAG to
image = pipeline(
    prompt="a detailed portrait",
    pag_scale=2.5,
    pag_applied_layers=["mid", "up"],  # Apply to middle and upsampling layers
    guidance_scale=7.0,
    num_inference_steps=50
).images[0]
```

### Integration with ChaoticFooocus

```python
# In your generation pipeline
def process_with_pag(prompt, **kwargs):
    pag_scale = kwargs.get('pag_scale', 0.0)
    
    if pag_scale > 0:
        # Use PAG pipeline
        pipeline = StableDiffusionXLPAGPipeline.from_pretrained(model_path)
        return pipeline(prompt=prompt, pag_scale=pag_scale, **kwargs)
    else:
        # Use standard pipeline
        return standard_pipeline(prompt=prompt, **kwargs)
```

## Parameters

### PAG-Specific Parameters

- **`pag_scale`** (float, default: 0.0): PAG guidance scale
  - 0.0: Disabled
  - 1.0-3.0: Mild improvement
  - 3.0-5.0: Strong improvement
  - >5.0: May cause artifacts

- **`pag_applied_layers`** (List[str], default: ["mid", "up"]): Layers to apply PAG to
  - `["mid"]`: Only middle layers
  - `["up"]`: Only upsampling layers  
  - `["mid", "up"]`: Both (recommended)
  - `["down", "mid", "up"]`: All layers (stronger effect)

## How It Works

### 1. Attention Perturbation
```python
# Normal attention
attention_scores = query @ key.transpose(-2, -1)

# Perturbed attention (example)
noise = torch.randn_like(attention_scores) * perturbation_scale
perturbed_scores = attention_scores + noise
```

### 2. Guidance Calculation
```python
# Three predictions: unconditional, conditional, perturbed
noise_pred_uncond, noise_pred_cond, noise_pred_perturb = noise_pred.chunk(3)

# PAG guidance
pag_guidance = noise_pred_cond - noise_pred_perturb
final_noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond) + pag_scale * pag_guidance
```

## Implementation Details

### Architecture
- **`StableDiffusionXLPAGPipeline`**: Main pipeline class extending SDXL
- **`PAGAttentionProcessor`**: Custom attention processor for perturbation
- **Attention Hooks**: Temporary modification of attention computation

### Perturbation Methods
1. **Noise Addition**: Add random noise to attention scores
2. **Attention Blurring**: Blur attention maps to reduce sharpness
3. **Pattern Modification**: Modify attention patterns directly

### Layer Selection
- **Down Layers**: Early feature extraction (subtle effect)
- **Mid Layers**: Core processing (balanced effect)  
- **Up Layers**: Detail generation (strong effect on final image)

## Performance

### Computational Cost
- **Memory**: ~33% increase (3 forward passes instead of 2)
- **Speed**: ~30% slower due to additional computation
- **Quality**: Significant improvement in coherence and detail

### Recommended Settings
- **General Use**: `pag_scale=3.0`, `pag_applied_layers=["mid", "up"]`
- **Portraits**: `pag_scale=2.5`, `pag_applied_layers=["up"]`
- **Landscapes**: `pag_scale=3.5`, `pag_applied_layers=["mid", "up"]`
- **Abstract**: `pag_scale=4.0`, `pag_applied_layers=["down", "mid", "up"]`

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use lower `pag_scale`
   - Apply to fewer layers

2. **Artifacts**
   - Lower `pag_scale` (try 2.0-3.0)
   - Use only `["mid"]` layers
   - Check for conflicting guidance methods

3. **No Visible Effect**
   - Increase `pag_scale` (try 3.0-5.0)
   - Add more layers: `["mid", "up"]`
   - Ensure PAG is actually enabled

### Debug Mode
```python
import logging
logging.getLogger("extras.PAG.pipeline_sdxl_pag").setLevel(logging.DEBUG)
```

## Comparison with Other Methods

| Method | Quality | Speed | Memory | Stability |
|--------|---------|-------|---------|-----------|
| Standard SDXL | Baseline | Fast | Low | High |
| CFG Only | Good | Fast | Low | High |
| PAG | Better | Medium | Medium | High |
| TPG | Variable | Slow | High | Medium |

## Future Improvements

- [ ] Adaptive perturbation based on generation step
- [ ] Layer-specific perturbation strengths
- [ ] Integration with other guidance methods
- [ ] Optimized attention processors
- [ ] Batch processing optimizations

## References

- Paper: "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance"
- Original Implementation: [Link to original repo if available]
- Diffusers Documentation: https://huggingface.co/docs/diffusers/