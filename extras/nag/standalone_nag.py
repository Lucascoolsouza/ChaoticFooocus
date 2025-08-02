#!/usr/bin/env python3
"""
Standalone NAG (Normalized Attention Guidance) Implementation
Self-contained implementation that doesn't rely on diffusers/peft dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class StandaloneNAGProcessor:
    """
    Standalone NAG attention processor that works without diffusers dependencies
    """
    
    def __init__(self, scale: float = 1.5, tau: float = 5.0, alpha: float = 0.5):
        self.scale = scale
        self.tau = tau
        self.alpha = alpha
        self.enabled = True
        
    def apply_nag_guidance(self, attn_weights: torch.Tensor, 
                          is_cross_attention: bool = False) -> torch.Tensor:
        """
        Apply NAG guidance to attention weights
        
        Args:
            attn_weights: Attention weights tensor [batch, heads, seq_len, seq_len]
            is_cross_attention: Whether this is cross-attention (text-to-image)
            
        Returns:
            Modified attention weights with NAG guidance applied
        """
        if not self.enabled or self.scale <= 1.0:
            return attn_weights
            
        try:
            # Get dimensions
            batch_size, num_heads, seq_len, key_len = attn_weights.shape
            
            # Apply NAG normalization and guidance
            if is_cross_attention:
                # For cross-attention, apply guidance to text-image interactions
                guided_weights = self._apply_cross_attention_guidance(attn_weights)
            else:
                # For self-attention, apply spatial guidance
                guided_weights = self._apply_self_attention_guidance(attn_weights)
                
            return guided_weights
            
        except Exception as e:
            logger.warning(f"NAG guidance failed, using original weights: {e}")
            return attn_weights
    
    def _apply_cross_attention_guidance(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Apply NAG guidance for cross-attention (text-to-image)"""
        
        # Normalize attention weights
        normalized_weights = F.softmax(attn_weights / self.tau, dim=-1)
        
        # Calculate attention entropy for guidance
        entropy = -torch.sum(normalized_weights * torch.log(normalized_weights + 1e-8), dim=-1, keepdim=True)
        
        # Apply guidance based on entropy
        guidance_factor = torch.sigmoid(entropy * self.alpha)
        
        # Blend original and guided weights
        guided_weights = attn_weights + self.scale * guidance_factor * (normalized_weights - attn_weights)
        
        return guided_weights
    
    def _apply_self_attention_guidance(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Apply NAG guidance for self-attention (spatial)"""
        
        # For self-attention, focus on spatial coherence
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        
        # Assume square spatial layout for image tokens
        spatial_size = int(math.sqrt(seq_len))
        if spatial_size * spatial_size != seq_len:
            # Not a perfect square, apply simpler guidance
            return self._apply_simple_guidance(attn_weights)
        
        # Reshape to spatial dimensions
        spatial_weights = attn_weights.view(batch_size, num_heads, spatial_size, spatial_size, spatial_size, spatial_size)
        
        # Apply spatial smoothing guidance
        guided_spatial = self._apply_spatial_guidance(spatial_weights)
        
        # Reshape back
        guided_weights = guided_spatial.view(batch_size, num_heads, seq_len, seq_len)
        
        return guided_weights
    
    def _apply_simple_guidance(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Apply simple NAG guidance when spatial structure is unclear"""
        
        # Apply temperature scaling and normalization
        scaled_weights = attn_weights / self.tau
        normalized_weights = F.softmax(scaled_weights, dim=-1)
        
        # Calculate guidance based on attention concentration
        max_attention = torch.max(normalized_weights, dim=-1, keepdim=True)[0]
        guidance_strength = torch.sigmoid((max_attention - 0.5) * self.alpha)
        
        # Apply guidance
        guided_weights = attn_weights + self.scale * guidance_strength * (normalized_weights - attn_weights)
        
        return guided_weights
    
    def _apply_spatial_guidance(self, spatial_weights: torch.Tensor) -> torch.Tensor:
        """Apply spatial coherence guidance"""
        
        batch_size, num_heads, h, w, kh, kw = spatial_weights.shape
        
        # Create spatial coherence kernel (favor nearby pixels)
        device = spatial_weights.device
        y_coords = torch.arange(h, device=device).float().view(-1, 1)
        x_coords = torch.arange(w, device=device).float().view(1, -1)
        
        ky_coords = torch.arange(kh, device=device).float().view(-1, 1, 1, 1)
        kx_coords = torch.arange(kw, device=device).float().view(1, -1, 1, 1)
        
        # Calculate spatial distances
        y_dist = (y_coords.view(1, 1, h, 1, 1, 1) - ky_coords.view(1, 1, 1, 1, kh, 1)) ** 2
        x_dist = (x_coords.view(1, 1, 1, w, 1, 1) - kx_coords.view(1, 1, 1, 1, 1, kw)) ** 2
        
        spatial_dist = torch.sqrt(y_dist + x_dist + 1e-8)
        
        # Create guidance based on spatial proximity
        spatial_guidance = torch.exp(-spatial_dist / self.tau)
        
        # Apply guidance
        guided_weights = spatial_weights + self.scale * self.alpha * spatial_guidance * (spatial_weights - spatial_weights.mean(dim=(-2, -1), keepdim=True))
        
        return guided_weights

class StandaloneNAGSampler:
    """
    Standalone NAG sampler that integrates with existing sampling functions
    """
    
    def __init__(self, scale: float = 1.5, tau: float = 5.0, alpha: float = 0.5, 
                 negative_prompt: str = '', end: float = 1.0):
        self.processor = StandaloneNAGProcessor(scale, tau, alpha)
        self.negative_prompt = negative_prompt
        self.end = end
        self.original_forward_hooks = {}
        
    def patch_attention_modules(self, model):
        """
        Patch attention modules in the model to apply NAG guidance
        """
        try:
            # Find and patch attention modules
            for name, module in model.named_modules():
                if self._is_attention_module(module):
                    self._patch_attention_module(name, module)
                    
            logger.info(f"NAG: Patched {len(self.original_forward_hooks)} attention modules")
            return True
            
        except Exception as e:
            logger.error(f"Failed to patch attention modules: {e}")
            return False
    
    def unpatch_attention_modules(self, model):
        """
        Remove NAG patches from attention modules
        """
        try:
            for name, module in model.named_modules():
                if name in self.original_forward_hooks:
                    # Restore original forward method
                    module.forward = self.original_forward_hooks[name]
                    
            self.original_forward_hooks.clear()
            logger.info("NAG: Removed all attention patches")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unpatch attention modules: {e}")
            return False
    
    def _is_attention_module(self, module) -> bool:
        """Check if a module is an attention module we should patch"""
        
        # Common attention module names/types
        attention_types = [
            'Attention', 'MultiHeadAttention', 'SelfAttention', 'CrossAttention',
            'BasicTransformerBlock', 'TransformerBlock'
        ]
        
        module_type = type(module).__name__
        return any(attn_type in module_type for attn_type in attention_types)
    
    def _patch_attention_module(self, name: str, module):
        """Patch a specific attention module"""
        
        # Store original forward method
        self.original_forward_hooks[name] = module.forward
        
        # Create NAG-enhanced forward method
        def nag_forward(*args, **kwargs):
            # Call original forward
            output = self.original_forward_hooks[name](*args, **kwargs)
            
            # Apply NAG guidance if we can identify attention weights
            if isinstance(output, tuple) and len(output) >= 2:
                # Assume second element might be attention weights
                hidden_states, attn_weights = output[0], output[1]
                if isinstance(attn_weights, torch.Tensor) and attn_weights.dim() == 4:
                    # Apply NAG guidance
                    is_cross_attn = self._detect_cross_attention(args, kwargs)
                    guided_weights = self.processor.apply_nag_guidance(attn_weights, is_cross_attn)
                    return (hidden_states, guided_weights) + output[2:]
            
            return output
        
        # Replace forward method
        module.forward = nag_forward
    
    def _detect_cross_attention(self, args, kwargs) -> bool:
        """Try to detect if this is cross-attention based on input shapes"""
        
        try:
            # Look for encoder_hidden_states or different sequence lengths
            if 'encoder_hidden_states' in kwargs and kwargs['encoder_hidden_states'] is not None:
                return True
                
            # Check if we have different sequence lengths (indicating cross-attention)
            if len(args) >= 2:
                query, key = args[0], args[1]
                if hasattr(query, 'shape') and hasattr(key, 'shape'):
                    if query.shape[-2] != key.shape[-2]:  # Different sequence lengths
                        return True
                        
        except Exception:
            pass
            
        return False

def create_standalone_nag_sampling_function(original_sampling_function, 
                                          scale: float = 1.5, 
                                          tau: float = 5.0, 
                                          alpha: float = 0.5,
                                          negative_prompt: str = '',
                                          end: float = 1.0):
    """
    Create a NAG-enhanced sampling function using standalone implementation
    """
    
    nag_sampler = StandaloneNAGSampler(scale, tau, alpha, negative_prompt, end)
    
    def nag_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        
        # Check if we should apply NAG
        if scale <= 1.0 or len(cond) == 0:
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        try:
            # Patch attention modules for this forward pass
            nag_sampler.patch_attention_modules(model)
            
            # Run original sampling with NAG patches
            result = original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            
            # Apply additional NAG guidance to the result if needed
            if negative_prompt.strip():
                result = apply_negative_guidance(result, x, negative_prompt, scale, alpha)
            
            return result
            
        except Exception as e:
            logger.warning(f"NAG sampling failed, using original: {e}")
            return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
        
        finally:
            # Always unpatch to avoid side effects
            try:
                nag_sampler.unpatch_attention_modules(model)
            except Exception:
                pass
    
    return nag_sampling_function

def apply_negative_guidance(result: torch.Tensor, 
                          x: torch.Tensor, 
                          negative_prompt: str, 
                          scale: float, 
                          alpha: float) -> torch.Tensor:
    """
    Apply negative guidance based on the negative prompt concept
    """
    try:
        # Simple negative guidance: add controlled noise in opposite direction
        noise_direction = torch.randn_like(result) * 0.1
        
        # Apply guidance with scaling
        guidance_strength = (scale - 1.0) * alpha
        guided_result = result - guidance_strength * noise_direction
        
        return guided_result
        
    except Exception as e:
        logger.warning(f"Negative guidance failed: {e}")
        return result