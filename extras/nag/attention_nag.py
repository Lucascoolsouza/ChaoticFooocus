from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention


class NAGAttnProcessor2_0:
    def __init__(self, nag_scale: float = 1.0, nag_tau: float = 2.5, nag_alpha:float = 0.5):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        apply_guidance = self.nag_scale >= 1 and encoder_hidden_states is not None
        if apply_guidance:
            origin_batch_size = batch_size - len(hidden_states)
            assert batch_size / origin_batch_size in [2, 3, 4]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if apply_guidance:
            if batch_size == 2 * origin_batch_size:
                query = query.tile(2, 1, 1)
            else:
                query = torch.cat((query, query[origin_batch_size:2 * origin_batch_size]), dim=0)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if apply_guidance:
            hidden_states_negative = hidden_states[-origin_batch_size:]
            if batch_size == 2 * origin_batch_size:
                hidden_states_positive = hidden_states[:origin_batch_size]
            else:
                hidden_states_positive = hidden_states[origin_batch_size:2 * origin_batch_size]
            # Apply extremely conservative NAG guidance formula
            if self.nag_scale <= 1.0:
                # At scale 1.0 or below, apply ultra-minimal guidance
                guidance_strength = 0.001  # 0.1% effect
                hidden_states_guidance = hidden_states_positive + (hidden_states_positive - hidden_states_negative) * guidance_strength
            else:
                # For higher scales, use very conservative NAG formula
                conservative_scale = 1.0 + (self.nag_scale - 1.0) * 0.05  # 95% reduction
                hidden_states_guidance = hidden_states_positive * conservative_scale - hidden_states_negative * (conservative_scale - 1)
            
            # Apply normalization with safety checks (more conservative)
            norm_positive = torch.norm(hidden_states_positive, p=1, dim=-1, keepdim=True).expand(*hidden_states_positive.shape)
            norm_guidance = torch.norm(hidden_states_guidance, p=1, dim=-1, keepdim=True).expand(*hidden_states_guidance.shape)

            # Prevent division by zero with larger epsilon
            norm_positive = torch.clamp(norm_positive, min=1e-6)
            norm_guidance = torch.clamp(norm_guidance, min=1e-6)

            scale = norm_guidance / norm_positive
            # Much more conservative tau clamping
            conservative_tau = min(self.nag_tau, 1.5)  # Cap tau at 1.5 to prevent extreme effects
            scale = torch.clamp(scale, min=0.5, max=conservative_tau)  # Narrower range
            hidden_states_guidance = hidden_states_guidance * torch.minimum(scale, scale.new_ones(1) * conservative_tau) / scale

            # Apply ultra-conservative alpha blending
            conservative_alpha = self.nag_alpha * 0.01  # 99% reduction
            hidden_states_guidance = hidden_states_guidance * conservative_alpha + hidden_states_positive * (1 - conservative_alpha)

            if batch_size == 2 * origin_batch_size:
                hidden_states = hidden_states_guidance
            elif batch_size == 3 * origin_batch_size:
                hidden_states = torch.cat((hidden_states[:origin_batch_size], hidden_states_guidance), dim=0)
            elif batch_size == 4 * origin_batch_size:
                hidden_states = torch.cat((hidden_states[:origin_batch_size], hidden_states_guidance, hidden_states[2 * origin_batch_size:3 * origin_batch_size]), dim=0)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states