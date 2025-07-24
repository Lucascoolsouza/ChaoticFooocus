"""
Detail-Daemon sampler wrappers for Focus
Based on the ComfyUI extension by muerrilla
https://github.com/muerrilla/sd-webui-detail-daemon
"""
from __future__ import annotations
import io
import math
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # non-GUI
import matplotlib.pyplot as plt




# --------------------------------------------------
# 1. Schedule generation
# --------------------------------------------------
def make_detail_daemon_schedule(steps: int,
                                start: float,
                                end: float,
                                bias: float,
                                amount: float,
                                exponent: float,
                                start_offset: float,
                                end_offset: float,
                                fade: float,
                                smooth: bool) -> np.ndarray:
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    # ascending part
    if mid_idx >= start_idx:
        vals = np.linspace(0, 1, mid_idx - start_idx + 1)
        if smooth:
            vals = 0.5 * (1 - np.cos(vals * np.pi))
        vals **= exponent
        vals *= amount - start_offset
        vals += start_offset
        multipliers[start_idx:mid_idx + 1] = vals

    # descending part
    if end_idx >= mid_idx:
        vals = np.linspace(1, 0, end_idx - mid_idx + 1)
        if smooth:
            vals = 0.5 * (1 - np.cos(vals * np.pi))
        vals **= exponent
        vals *= amount - end_offset
        vals += end_offset
        multipliers[mid_idx:end_idx + 1] = vals

    multipliers[:start_idx] = start_offset
    if end_idx + 1 < steps:
        multipliers[end_idx + 1:] = end_offset
    multipliers *= 1 - fade
    return multipliers


# --------------------------------------------------
# 2. Core sampler wrapper
# --------------------------------------------------
def detail_daemon_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    dds_wrapped_sampler: object,
    dds_make_schedule: callable,
    dds_cfg_scale_override: float,
    **kwargs: dict,
):
    """
    Wrapped sampler that modulates the sigma schedule according to Detail-Daemon.
    """
    steps = len(sigmas) - 1
    schedule_np = dds_make_schedule(steps)
    schedule = torch.from_numpy(schedule_np).float().to(sigmas.device)

    cfg = dds_cfg_scale_override if dds_cfg_scale_override > 0 else 1.0 # Assuming default cfg_scale is 1.0 if not overridden

    def model_fn(x_in, sigma_in, **extra_args):
        sigma_cpu = sigma_in.detach().cpu()
        deltas = (sigmas[:-1] - sigma_cpu).abs()
        idx = int(deltas.argmin())

        sched_len = len(schedule)
        if idx >= sched_len:
            return model(x_in, sigma_in, **extra_args)

        # linear interpolation between neighbours
        if idx + 1 < sched_len and deltas[idx] != 0:
            nlow, nhigh = sigmas[idx], sigmas[idx + 1]
            ratio = ((sigma_cpu - nlow) / (nhigh - nlow)).clamp(0, 1)
            dd = torch.lerp(schedule[idx], schedule[idx + 1], ratio).item()
        else:
            dd = schedule[idx].item()

        sigma_adj = sigma_in * max(1e-6, 1.0 - dd * 0.1 * cfg)
        return model(x_in, sigma_adj, **extra_args)

    return dds_wrapped_sampler.sampler_function(model_fn, x, sigmas, **kwargs)


# --------------------------------------------------
# 3. Utility: MultiplySigmas
# --------------------------------------------------
def multiply_sigmas(p, model, x, sigmas, sampler_orig, factor=1.0, start=0.0, end=1.0, **kw):
    sigmas = sigmas.clone()
    total = len(sigmas)
    s_idx = int(start * total)
    e_idx = int(end * total)
    sigmas[s_idx:e_idx] *= factor
    return sampler_orig(p, model, x, sigmas, **kw)


# --------------------------------------------------
# 4. Utility: LyingSigma (fake sigma)
# --------------------------------------------------
def lying_sigma_sampler(p, model, x, sigmas, sampler_orig,
                        dishonesty=-0.05,
                        start_pct=0.1,
                        end_pct=0.9,
                        **kw):
    model_wrap = model
    ms = p.sd_model.model_sampling
    start_sig = ms.percent_to_sigma(start_pct)
    end_sig = ms.percent_to_sigma(end_pct)

    def model_fn(x_in, sigma_in, **extra):
        sig_val = sigma_in.max().detach().cpu().item()
        if end_sig <= sig_val <= start_sig:
            sigma_in = sigma_in * (1.0 + dishonesty)
        return model_wrap(x_in, sigma_in, **extra)

    return sampler_orig(p, model_fn, x, sigmas, **kw)


# --------------------------------------------------
# 5. Graph output helper
# --------------------------------------------------
def plot_detail_daemon_schedule(schedule: np.ndarray) -> Image.Image:
    plt.figure(figsize=(6, 4))
    plt.plot(schedule, label="Multiplier")
    plt.title("Detail-Daemon Schedule")
    plt.xlabel("Step")
    plt.ylabel("Multiplier")
    plt.grid(True)
    plt.ylim(-1, 1)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


# --------------------------------------------------
# 6. Register samplers in Focus
# --------------------------------------------------
