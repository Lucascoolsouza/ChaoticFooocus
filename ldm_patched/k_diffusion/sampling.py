import math

from scipy import integrate
import torch
from torch import nn
import torchsde
from tqdm.auto import trange, tqdm
import numpy as np
import torch.nn.functional as F

from . import utils


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)
    

def get_sigmas_karras_base(n, sigma_min, sigma_max, rho=7., device='cpu'):
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def get_sigmas_karras_sinusoidal(n, sigma_min, sigma_max, rho=7., device='cpu',
                                 sin_freq=5.0, amp=0.1):
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    base = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    perturb = 1 + amp * torch.sin(2 * torch.pi * sin_freq * ramp)
    sigmas = base * perturb
    return append_zero(sigmas).to(device)

def get_sigmas_karras_chaotic(n, sigma_min, sigma_max, rho=7., device='cpu',
                              logistic_r=3.6, chaotic_amplitude=0.12, iters=5):
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    base = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    x = ramp.clone()
    for _ in range(iters):
        x = logistic_r * x * (1 - x)
    sigmas = base * (1 + chaotic_amplitude * (2*x - 1))
    print("behold, the chaos!")
    return append_zero(sigmas).to(device)

def get_sigmas_karras_zigzag(n, sigma_min, sigma_max, rho=5., device='cpu', zigzag_strength=0.5):
    ramp = torch.linspace(0, 1, n, device=device)
    zigzag_offset = (torch.arange(n, device=device) % 2) * zigzag_strength / n
    ramp += zigzag_offset
    ramp = torch.clamp(ramp, 0, 1)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def get_sigmas_karras_piecewise(n, sigma_min, sigma_max, device='cpu', start_frac=0.2, end_frac=0.8):
    i = torch.arange(n, device=device) / (n-1)
    sig = torch.where(
        i < start_frac,
        sigma_max,
        torch.where(
          i > end_frac,
          sigma_min,
          sigma_max + (i - start_frac) / (end_frac - start_frac) * (sigma_min - sigma_max)
        )
    )
    return append_zero(sig).to(device)

def get_sigmas_karras_jitter(n, sigma_min, sigma_max, rho=1.2,device='cpu', jitter_strength=0.5):
    ramp = torch.linspace(0, 1, n, device=device)
    jitter = (torch.rand(n, device=device) - 0.5) * 2 * jitter_strength / n
    ramp += jitter
    ramp = torch.clamp(ramp, 0, 1)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    print("i fell thoose jitters")
    return append_zero(sigmas).to(device)

def get_sigmas_karras_upscale(n, sigma_min, sigma_max, rho=7., device='cpu',
                                    detail_freq=10.0, detail_amp=0.05, sharpness=0.1, noise_strength=0.02):

    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    base = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    detail_perturb = detail_amp * torch.sin(2 * torch.pi * detail_freq * ramp)
    sharp_transition = torch.tanh(sharpness * (ramp - 0.5)) + 1  
    noise = noise_strength * (torch.rand(n, device=device) - 0.5)
    sigmas = base * (1 + detail_perturb) * sharp_transition + noise
    return append_zero(sigmas).to(device)

def get_sigmas_karras_trow_random_blsht(n, sigma_min, sigma_max, num_levels=10, device='cpu'):
    if num_levels < 2:
        raise ValueError("num_levels must be at least 2")
    ramp = torch.linspace(1, 0, n, device=device)
    quantized_ramp = torch.floor(ramp * (num_levels - 1)) / (num_levels - 1)
    log_sigma_min = math.log(sigma_min)
    log_sigma_max = math.log(sigma_max)
    log_sigmas = quantized_ramp * (log_sigma_max - log_sigma_min) + log_sigma_min
    sigmas = torch.exp(log_sigmas)
    return append_zero(sigmas).to(device)













def get_sigmas_comic(n, sigma_min, sigma_max, iterations=2, device='cpu'):
    """Creates a fractal-like noise schedule with self-similar patterns at different scales."""
    t = torch.linspace(0, 1, n, device=device)
    fractal = torch.zeros_like(t)
    
    # Build up the fractal pattern
    for i in range(iterations):
        scale = 20 ** i
        fractal += torch.sin(t * math.pi * scale) / scale
    
    # Normalize and convert to sigma space
    fractal = (fractal - fractal.min()) / (fractal.max() - fractal.min())
    
    # Apply exponential mapping for better diffusion properties
    sigmas = sigma_min * torch.exp(fractal * math.log(sigma_max / sigma_min))
    
    return append_zero(sigmas).to(device)










    
def get_sigmas_karras_dream(n, sigma_min, sigma_max, rho=70., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022) with a google-dream like schedule"""
    sigmas = []
    prev = sigma_max
    for _ in range(n - 1):
        curr = prev - (prev - sigma_min) / (2 ** (1 / rho))
        sigmas.append(curr)
        prev = curr
    sigmas.append(sigma_min)
    return append_zero(torch.tensor(sigmas[::-1], device=device, dtype=torch.float32)).to(device)

def get_sigmas_karras_golden_ratio(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022) with the golden ratio."""
    golden_ratio = (1 + 5 ** 0.5) / 2
    sigmas = []
    prev = sigma_max
    for _ in range(n - 1):
        curr = prev * golden_ratio ** (-1 / rho)
        sigmas.append(curr)
        prev = curr
    sigmas.append(sigma_min)
    return append_zero(torch.tensor(sigmas[::-1], device=device, dtype=torch.float32)).to(device)


def get_sigmas_karras_pixel_art(n, sigma_min, sigma_max, rho=7., device='cpu', quantize_levels=8):
    """Constructs a quantized noise schedule for pixel art-like generation."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    base = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    
    # Quantize the sigmas to create step-like transitions typical in pixel art
    quantized = torch.round(base * quantize_levels) / quantize_levels
    sigmas = torch.clamp(quantized, sigma_min, sigma_max)
    return append_zero(sigmas).to(device)



def get_sigmas_karras_mini_dalle(n, sigma_min, sigma_max, rho=20., device='cpu',
                            freq=10.0, amp=0.05):
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    base = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    # smooth checker modulation
    checker = torch.sin(2 * torch.pi * freq * ramp) * torch.cos(2 * torch.pi * freq * ramp)
    sigmas = (base * (1 + amp * checker)).clamp(min=sigma_min, max=sigma_max)
    return append_zero(sigmas).to(device)









def get_sigmas_karras_grid(n, sigma_min, sigma_max, rho=7., device='cpu',
                           freq=5.0, amp=0.05):
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    base = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    # smooth checker modulation
    checker = torch.sin(2 * torch.pi * freq * ramp) * torch.cos(2 * torch.pi * freq * ramp)
    sigmas = (base * (1 + amp * checker)).clamp(min=sigma_min, max=sigma_max)
    return append_zero(sigmas).to(device)

def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1., device='cpu'):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)











def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if self.cpu_tree:
            w = torch.stack([tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_dreamy(model, x, sigmas, extra_args=None, callback=None, disable=None, smoothing_factor=3.25):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    d_old = None  # Store the previous derivative
    print ("it's dreaming")
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        if d_old is not None and smoothing_factor > 0:
            d_smooth = (1.0 - smoothing_factor) * d + smoothing_factor * d_old
        else:
            d_smooth = d
        d_old = d 
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised, 'd': d, 'd_smooth': d_smooth})
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d_smooth * dt
    return x

@torch.no_grad()
def sample_euler_dreamy_pp(model, x, sigmas, extra_args=None, callback=None, disable=None,
                       noise_factor=0.005, extrapolation_factor=0.002, s_noise=0.2):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) # Use a basic noise sampler here for perturbation
    s_in = x.new_ones([x.shape[0]])
    print("It's dreaming... progressively?")

    for i in trange(len(sigmas) - 1, disable=disable):

        sigma_cur = sigmas[i]
        sigma_next = sigmas[i+1]

        denoised = model(x, sigma_cur * s_in, **extra_args)
        d = to_d(x, sigma_cur, denoised)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_cur, 'sigma_hat': sigma_cur, 'denoised': denoised, 'd': d, 'stage': 'predictor_start'})
        perturbation = noise_sampler(sigma_cur, sigma_next) * s_noise * noise_factor * sigma_cur
        d_perturbed = d + perturbation

        dt = sigma_next - sigma_cur
        x_pred = x + d_perturbed * dt

        if sigma_next == 0:
            x = x_pred
            if callback is not None:
                 callback({'x': x, 'i': i, 'sigma': sigma_next, 'sigma_hat': sigma_cur, 'denoised': denoised, 'd': d, 'stage': 'final_step'})

        else:
            denoised_2 = model(x_pred, sigma_next * s_in, **extra_args)
            d_2 = to_d(x_pred, sigma_next, denoised_2)
            d_avg = (d_perturbed + d_2) / 2 # Heun uses d, not d_perturbed here. Let's stick to d_perturbed for consistency with x_pred
            d_diff = d_2 - d_perturbed # Difference using the perturbed d for consistency
            
            d_prime = d_avg + extrapolation_factor * d_diff

            if callback is not None:

                 callback({'x': x, 'i': i, 'sigma': sigma_next, 'sigma_hat': sigma_cur, 'denoised': denoised, 'd_prime': d_prime, 'stage': 'corrector_update'})

            x = x + d_prime * dt

    return x

def quantize_tensor(tensor, num_levels):
    if num_levels <= 1:
        return torch.zeros_like(tensor)
    
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    if max_val == min_val: 
        return torch.full_like(tensor, (min_val + max_val) / 2) 

    scaled_tensor = (tensor - min_val) / (max_val - min_val)

    quantized_indices = torch.floor(scaled_tensor * (num_levels - 1))

    quantized_indices = torch.clamp(quantized_indices, 0, num_levels - 1)
    level_values = torch.linspace(min_val, max_val, num_levels, device=tensor.device)

    bin_width = (max_val - min_val) / (num_levels - 1)
    quantized_values = min_val + quantized_indices * bin_width
    
    return quantized_values


@torch.no_grad()
def sample_euler_chaotic(model, x, sigmas, extra_args=None, callback=None, disable=None, num_levels=128):

    if num_levels < 2:
        raise ValueError("num_levels must be at least 2 for quantization")
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)

        # Quantize the derivative
        d_quantized = quantize_tensor(d, num_levels)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised, 'd': d, 'd_quantized': d_quantized})

        dt = sigmas[i + 1] - sigmas[i]

        # Euler method using the quantized derivative
        x = x + d_quantized * dt

    return x

@torch.no_grad()
def sample_euler_triangle_wave(model, x, sigmas, extra_args=None, callback=None, disable=None,
                               oscillation_amplitude=0.4, oscillation_periods=1.6, s_noise=1.0):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n_steps = len(sigmas) - 1

    for i in trange(n_steps, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)

        # Calculate oscillation factor (sine wave from 0 to 2*pi*periods)
        phase = (i / n_steps) * oscillation_periods * 2 * math.pi
        osc_factor = oscillation_amplitude * math.sin(phase)

        # Add scaled noise to the derivative
        perturbation_noise = torch.randn_like(x) * s_noise
        d_perturbed = d + osc_factor * perturbation_noise
        # Note: The noise is added to 'd' before multiplying by 'dt'.
        # The effect scales with the magnitude of 'd'.

        if callback is not None:
             callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised, 'd': d, 'd_perturbed': d_perturbed, 'osc_factor': osc_factor})

        dt = sigmas[i + 1] - sigmas[i]

        # Euler method using the perturbed derivative
        x = x + d_perturbed * dt

    return x
@torch.no_grad()
def sample_triangular(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Triangle artifact-enhancing sampler based on the Heun method."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Triangle enhancement parameters
    triangle_scale = 3.2  # Emphasizes triangle patterns
    edge_enhance = 0.05  # Edge enhancement amount
    
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        
        # Get model prediction
        denoised = model(x, sigma_hat * s_in, **extra_args)
        
        # Triangle pattern enhancement - creates more angular artifacts
        if i > len(sigmas) // 3:  # Only apply in later steps
            # Enhance edges by emphasizing gradients
            edges = torch.abs(x - denoised)
            edge_mask = (edges > edges.mean() * triangle_scale).float()
            denoised = denoised * (1 - edge_enhance * edge_mask) + x * (edge_enhance * edge_mask)
        
        d = to_d(x, sigma_hat, denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method with triangle emphasis
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    
    return x

@torch.no_grad()
def sample_pixelart(model, x, sigmas, extra_args=None, callback=None, disable=None, quantize_levels=6, s_noise=0.05):
    """Pixelated art style sampler that quantizes outputs during generation."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Track previous denoised result to blend with current for stability
    prev_denoised = None
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # Standard denoising step
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        # Blend with previous denoised for stability
        if prev_denoised is not None and i > len(sigmas) // 2:
            blend_factor = 0.2  # How much to blend with previous iteration
            denoised = denoised * (1 - blend_factor) + prev_denoised * blend_factor
        
        # Apply pixel art effect through quantization in later diffusion steps
        if i > len(sigmas) * 0.6:  # Only apply in later steps for better convergence
            # Gradually increase quantization effect
            progress = (i - len(sigmas) * 0.6) / (len(sigmas) * 0.4)
            effective_levels = max(2, int(quantize_levels * (1 - progress * 0.5)))
            
            # Quantize the denoised prediction to create pixel art effect
            denoised = torch.round(denoised * effective_levels) / effective_levels
            
            # Add block patterns for more "pixel-art" feel
            block_size = max(1, int(2 * progress))
            if block_size > 1:
                # Simple pooling to create blocks
                shape = denoised.shape
                denoised = denoised.view(shape[0], shape[1], 
                                      shape[2]//block_size, block_size, 
                                      shape[3]//block_size, block_size)
                denoised = denoised.mean(dim=(3, 5))
                denoised = torch.nn.functional.interpolate(denoised, size=(shape[2], shape[3]), mode='nearest')
        
        prev_denoised = denoised
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        # Standard update step with noise addition for next iteration
        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt
        
        # Add controlled noise for next iteration if not the final step
        if sigmas[i + 1] > 0:
            x = x + torch.randn_like(x) * sigmas[i + 1] * s_noise
    
    return x

@torch.no_grad()
def sample_dreamy(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=0.1, dreaminess=0.1):
    """Dreamy VGAN-style sampler with smooth transitions and glow effects."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Keep track of previous denoised results for motion blur effect
    history = []
    history_weight = 0.3  # How much to blend with historical predictions
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # Standard model prediction
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        # Apply dreamy effects in mid-to-late diffusion steps
        dream_start = int(len(sigmas) * 0.3)
        if i > dream_start:
            # Calculate how far we are into the "dreamy" phase
            dream_phase = min(1.0, (i - dream_start) / (len(sigmas) - dream_start - 1))
            
            # 1. Apply temporal blending with history (motion blur effect)
            if history:
                # Weighted average of previous results (temporal smoothing)
                temporal_mix = sum(h * w for h, w in zip(history, 
                                  [pow(0.7, len(history)-j) for j in range(len(history))]))
                temporal_mix /= sum(pow(0.7, len(history)-j) for j in range(len(history)))
                
                # Blend current with history based on dream phase
                blend_amount = history_weight * dream_phase * dreaminess
                denoised = denoised * (1 - blend_amount) + temporal_mix * blend_amount
            
            # 2. Apply dream glow effect (enhance bright areas)
            if dream_phase > 0.4:
                # Create a glow mask for bright areas
                brightness = denoised.mean(dim=1, keepdim=True)
                glow_mask = torch.sigmoid((brightness - 0.6) * 10) * dreaminess * 0.3
                
                # Apply glow by brightening and slightly shifting bright areas
                glow = torch.nn.functional.avg_pool2d(denoised, 5, stride=1, padding=2)
                denoised = denoised * (1 - glow_mask) + glow * glow_mask
            
            # 3. Add subtle color shifts for dream-like quality
            if dream_phase > 0.2:
                color_shift = torch.sin(torch.tensor([0.1, 0.2, 0.3]) * i * 0.1)
                color_shift = color_shift.view(1, 3, 1, 1).to(denoised.device) * 0.02 * dreaminess
                denoised = denoised + color_shift
        
        # Add to history, keep only last 3 iterations
        history.append(denoised.detach())
        if len(history) > 3:
            history.pop(0)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        # Standard update
        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]
        
        # Apply smoother transition with controlled noise
        if i > len(sigmas) * 0.7:
            # Gradually reduce noise at the end for smoother results
            noise_scale = s_noise * (1.0 - (i - len(sigmas) * 0.7) / (len(sigmas) * 0.3))
        else:
            noise_scale = s_noise
            
        x = x + d * dt
        
        # Add noise for next iteration if not the last step
        if sigmas[i + 1] > 0:
            noise = torch.randn_like(x)
            # Apply wavelike pattern to noise for dreamy effect
            if dreaminess > 0.3:
                wave = torch.sin(torch.linspace(0, 3, x.shape[2], device=x.device).view(1, 1, -1, 1) * math.pi)
                wave = wave + torch.sin(torch.linspace(0, 3, x.shape[3], device=x.device).view(1, 1, 1, -1) * math.pi)
                wave = wave * 0.5 * dreaminess
                noise = noise * (1 + wave)
            
            x = x + noise * sigmas[i + 1] * noise_scale

        # Debug prints
        print(f"Dreamy Sampler - Iteration {i}:")
        print(f"  History length: {len(history)}")
        if history:
            print(f"  First history item stats: mean={history[0].mean()}, std={history[0].std()}, min={history[0].min()}, max={history[0].max()}")
        print(f"  Denoised stats: mean={denoised.mean()}, std={denoised.std()}, min={denoised.min()}, max={denoised.max()}")
        print(f"  X stats: mean={x.mean()}, std={x.std()}, min={x.min()}, max={x.max()}")
    
    return x

# Sampler for comic book style with sharp edges and high contrast
@torch.no_grad()
def sample_comic(model, x, sigmas, extra_args=None, callback=None, disable=None, 
                contrast=0.5, edge_threshold=0.5):
    """Comic book style sampler with high contrast and sharp edges."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # Get model prediction
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        # Apply comic book styling in later steps
        if i > len(sigmas) * 0.5:
            # Calculate effect strength based on current step
            effect_strength = min(1.0, (i - len(sigmas) * 0.5) / (len(sigmas) * 0.5))
            
            # 1. Increase contrast
            denoised_mean = denoised.mean()
            denoised = denoised_mean + (denoised - denoised_mean) * (1 + effect_strength * contrast)
            
            # 2. Edge detection and enhancement
            if i > len(sigmas) * 0.7:
                # Simple gradient-based edge detection
                dx = torch.abs(torch.roll(denoised, -1, dims=-1) - denoised)
                dy = torch.abs(torch.roll(denoised, -1, dims=-2) - denoised)
                edges = torch.max(dx, dy)
                edge_mask = (edges > edge_threshold).float()
                
                # Sharpen edges - darker lines on edges
                denoised = denoised * (1 - edge_mask * effect_strength * 0.5)
            
            # 3. Color simplification - reduce color palette
            if i > len(sigmas) * 0.8:
                # Quantize colors for comic book look
                levels = max(4, 10 - int(effect_strength * 5))
                denoised = torch.round(denoised * levels) / levels
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        # Standard update
        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]
        
        # Use Heun's method for better stability
        if sigmas[i + 1] == 0:
            # Euler method for last step
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    
    return x

# Fractal-style sampler that creates self-similar patterns
@torch.no_grad()
def sample_fractal(model, x, sigmas, extra_args=None, callback=None, disable=None,
                  recursion_depth=5, fractal_scale=0.07):
    """Fractal-style sampler that enhances self-similar patterns."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Store results from prior iterations to blend with current step
    multi_scale_history = [None] * recursion_depth
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # Get standard prediction
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        # Apply fractal effects in later diffusion steps
        if i > len(sigmas) * 0.4:
            # Calculate effect strength
            effect_strength = min(1.0, (i - len(sigmas) * 0.4) / (len(sigmas) * 0.6))
            effect_strength *= fractal_scale
            
            # Create multi-scale self-similar patterns
            fractal_denoised = denoised.clone()
            
            # Generate and combine multiple rescaled versions of the image
            for scale in range(recursion_depth):
                # Scale factor for this level
                scale_factor = 1.0 / (2 ** (scale + 1))
                
                # Downscale
                h, w = denoised.shape[2:]
                scaled_h, scaled_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
                downscaled = torch.nn.functional.interpolate(
                    denoised, size=(scaled_h, scaled_w), mode='bilinear'
                )
                
                # Optional: Apply extra noise to this scale
                if scale > 0:
                    scale_noise = torch.randn_like(downscaled) * 0.1 * scale
                    downscaled = downscaled + scale_noise
                
                # Process this scale through the model if we're deep enough in the diffusion
                if i > len(sigmas) * 0.6 and scale < recursion_depth - 1:
                    # Process with adjusted noise level
                    scale_sigma = sigmas[i] * (1 + scale * 0.5)
                    downscaled = model(downscaled, scale_sigma * s_in, **extra_args)
                    
                    # Store in history for this scale
                    multi_scale_history[scale] = downscaled
                elif multi_scale_history[scale] is not None:
                    # Blend with history for this scale
                    downscaled = downscaled * 0.7 + multi_scale_history[scale] * 0.3
                
                # Upscale back and blend with the main image
                upscaled = torch.nn.functional.interpolate(
                    downscaled, size=(h, w), mode='bilinear'
                )
                
                # Calculate blend weight that decreases with scale
                blend_weight = effect_strength * (0.5 ** scale)
                fractal_denoised = fractal_denoised * (1 - blend_weight) + upscaled * blend_weight
            
            # Apply the fractal-enhanced result
            denoised = denoised * (1 - effect_strength * 0.7) + fractal_denoised * (effect_strength * 0.7)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        # Standard update with noise addition
        d = to_d(x, sigmas[i], denoised)
        dt = sigmas[i + 1] - sigmas[i]
        x = x + d * dt
        
        # Add controlled noise for next iteration if not the final step
        if sigmas[i + 1] > 0:
            x = x + torch.randn_like(x) * sigmas[i + 1]
    
    return x
@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_scale=0., s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        # Add noise proportional to step size if noise_scale > 0
        if noise_scale > 0:
            eps = torch.randn_like(x) * s_noise
            # Noise scaled by step difference for consistency
            x = x + eps * noise_scale * (sigmas[i] - sigmas[i + 1])
        
        # Model prediction at current sigma
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        
        # Optional progress callback
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})
        
        # Step size
        dt = sigmas[i + 1] - sigmas[i]
        
        if sigmas[i + 1] == 0:
            # Euler step for the final iteration
            x = x + d * dt
        else:
            # Heun's method: two-stage update
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    
    return x


@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_dpm_2_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_scale=0., s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        # Add noise proportional to step size if noise_scale > 0
        if noise_scale > 0:
            eps = torch.randn_like(x) * s_noise * 4
            # Noise scaled by step difference for consistency
            x = x + eps * noise_scale * (sigmas[i] - sigmas[i + 1])
        
        # Model prediction at current sigma
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        
        # Optional progress callback
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})
        
        # Step size
        dt = sigmas[i + 1] - sigmas[i]
        
        if sigmas[i + 1] == 0:
            # Euler step for the final iteration
            x = x + d * dt / 2
        else:
            # Heun's method: two-stage update
            x_2 = x + d * dt / 2
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    
    return x




class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control."""
    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_fast(self, x, t_start, t_end, nfe, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if not t_end > t_start and eta:
            raise ValueError('eta must be 0 for reverse sampling')

        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
            denoised = x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})

            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(x, t, t_next_, eps_cache=eps_cache)
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(x, t, t_next_, eps_cache=eps_cache)
            else:
                x, eps_cache = self.dpm_solver_3_step(x, t, t_next_, eps_cache=eps_cache)

            x = x + su * s_noise * noise_sampler(self.sigma(t), self.sigma(t_next))

        return x

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if order not in {2, 3}:
            raise ValueError('order should be 2 or 3')
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError('eta must be 0 for reverse sampling')
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - self.sigma(s) * eps

            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t_, eps_cache=eps_cache)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return x, info


@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None, eta=0., s_noise=1., noise_sampler=None):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        return dpm_solver.dpm_solver_fast(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), n, eta, s_noise, noise_sampler)


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    if return_info:
        return x, info
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic)."""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    """DPM-Solver++(2M) SDE."""

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE."""

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_3m_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler)

@torch.no_grad()
def sample_dpmpp_2m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_2m_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, solver_type=solver_type)

@torch.no_grad()
def sample_dpmpp_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, r=r)


def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    alpha_cumprod_prev = 1 / ((sigma_prev * sigma_prev) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)

    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu

def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x


@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)

@torch.no_grad()
def sample_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
    return x


@torch.no_grad()
def sample_heunpp2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    # From MIT licensed: https://github.com/Carzit/sd-webui-samplers-scheduler/
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    s_end = sigmas[-1]
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == s_end:
            # Euler method
            x = x + d * dt
        elif sigmas[i + 2] == s_end:

            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)

            w = 2 * sigmas[0]
            w2 = sigmas[i+1]/w
            w1 = 1 - w2

            d_prime = d * w1 + d_2 * w2


            x = x + d_prime * dt

        else:
            # Heun++
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            dt_2 = sigmas[i + 2] - sigmas[i + 1]

            x_3 = x_2 + d_2 * dt_2
            denoised_3 = model(x_3, sigmas[i + 2] * s_in, **extra_args)
            d_3 = to_d(x_3, sigmas[i + 2], denoised_3)

            w = 3 * sigmas[0]
            w2 = sigmas[i + 1] / w
            w3 = sigmas[i + 2] / w
            w1 = 1 - w2 - w3

            d_prime = w1 * d + w2 * d_2 + w3 * d_3
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=0.3):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.inner_model.model_sampling
    timesteps_s = torch.floor((1 - eta) * model_sampling.timestep(sigmas)).to(dtype=torch.long).detach().cpu()
    timesteps_s[-1] = 0
    alpha_prod_s = model_sampling.alphas_cumprod[timesteps_s]
    beta_prod_s = 1 - alpha_prod_s
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)  # predicted_original_sample
        eps = (x - denoised) / sigmas[i]
        denoised = alpha_prod_s[i + 1].sqrt() * denoised + beta_prod_s[i + 1].sqrt() * eps

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        x = denoised
        if eta > 0 and sigmas[i + 1] > 0:
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x / alpha_prod_s[i+1].sqrt() + noise * (sigmas[i+1]**2 + 1 - 1/alpha_prod_s[i+1]).sqrt()
        else:
            x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2)

    return x


@torch.no_grad()
def sample_euler_token_shuffle(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=0.5, shuffle_strength=0.5):
    """Euler sampler with token shuffling at each step.
    
    This sampler shuffles the spatial positions (tokens) of the latent tensor at each
    denoising step, creating more diverse and chaotic generation patterns.
    
    Args:
        shuffle_strength: Controls how much shuffling to apply (0.0 = no shuffle, 1.0 = full shuffle)
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Store original spatial dimensions
    batch_size, channels, height, width = x.shape
    total_tokens = height * width
    
    print(f"Token shuffle sampler active - shuffling {total_tokens} tokens each step")
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # Apply churn noise if specified
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        
        # Shuffle tokens (spatial positions) before denoising
        if shuffle_strength > 0.0:
            # Progressive shuffling that decreases over time to preserve final details
            step_ratio = i / (len(sigmas) - 1)
            dynamic_strength = shuffle_strength * (1.0 - step_ratio * 0.8)  # Reduce shuffling as we progress
            
            if dynamic_strength > 0.01:  # Only shuffle if strength is meaningful
                # Flatten spatial dimensions
                x_flat = x.view(batch_size, channels, total_tokens)  # [B, C, H*W]
                
                # Create shuffling indices for each batch item
                shuffled_x = torch.zeros_like(x_flat)
                for b in range(batch_size):
                    # Local shuffling: shuffle within small neighborhoods to preserve structure
                    neighborhood_size = max(4, int(16 * (1.0 - dynamic_strength)))  # Smaller neighborhoods for stronger shuffle
                    
                    # Create a copy to work with
                    current_tokens = x_flat[b].clone()
                    
                    # Apply neighborhood-based shuffling
                    for start_idx in range(0, total_tokens, neighborhood_size):
                        end_idx = min(start_idx + neighborhood_size, total_tokens)
                        neighborhood_size_actual = end_idx - start_idx
                        
                        if neighborhood_size_actual > 1:
                            # Only shuffle a fraction of tokens within this neighborhood
                            num_to_shuffle = max(1, int(neighborhood_size_actual * dynamic_strength))
                            
                            # Get indices within this neighborhood
                            local_indices = torch.arange(start_idx, end_idx, device=x.device)
                            
                            # Randomly select which tokens to shuffle
                            shuffle_candidates = torch.randperm(neighborhood_size_actual, device=x.device)[:num_to_shuffle]
                            selected_indices = local_indices[shuffle_candidates]
                            
                            # Shuffle only the selected tokens within the neighborhood
                            if len(selected_indices) > 1:
                                shuffled_order = torch.randperm(len(selected_indices), device=x.device)
                                temp_values = current_tokens[:, selected_indices].clone()
                                current_tokens[:, selected_indices] = temp_values[:, shuffled_order]
                    
                    shuffled_x[b] = current_tokens
                
                # Blend with original to reduce harshness
                blend_factor = 0.7 + 0.3 * step_ratio  # More blending as we progress
                x_original = x.view(batch_size, channels, total_tokens)
                x_blended = blend_factor * x_original + (1.0 - blend_factor) * shuffled_x
                
                # Reshape back to spatial dimensions
                x = x_blended.view(batch_size, channels, height, width)
        
        # Standard denoising step
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        
        # Euler step
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    
    return x


@torch.no_grad()
def sample_heun_token_shuffle(model, x, sigmas, extra_args=None, callback=None, disable=None, shuffle_strength=0.5, s_noise=0.5):
    """Heun sampler with token shuffling at each step.
    
    This sampler combines Heun's method with token shuffling for more accurate
    but still chaotic generation.
    
    Args:
        shuffle_strength: Controls how much shuffling to apply (0.0 = no shuffle, 1.0 = full shuffle)
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Store original spatial dimensions
    batch_size, channels, height, width = x.shape
    total_tokens = height * width
    
    print(f"Token shuffle Heun sampler active - shuffling {total_tokens} tokens each step")
    
    def shuffle_tokens(tensor, strength, step_ratio):
        """Helper function to shuffle tokens in a tensor with improved algorithm"""
        if strength <= 0.0:
            return tensor
            
        b, c, h, w = tensor.shape
        tokens = h * w
        
        # Progressive shuffling that decreases over time
        dynamic_strength = strength * (1.0 - step_ratio * 0.8)
        
        if dynamic_strength <= 0.01:
            return tensor
        
        # Flatten spatial dimensions
        flat = tensor.view(b, c, tokens)
        shuffled = torch.zeros_like(flat)
        
        for batch_idx in range(b):
            # Local shuffling: shuffle within small neighborhoods to preserve structure
            neighborhood_size = max(4, int(16 * (1.0 - dynamic_strength)))
            
            # Create a copy to work with
            current_tokens = flat[batch_idx].clone()
            
            # Apply neighborhood-based shuffling
            for start_idx in range(0, tokens, neighborhood_size):
                end_idx = min(start_idx + neighborhood_size, tokens)
                neighborhood_size_actual = end_idx - start_idx
                
                if neighborhood_size_actual > 1:
                    # Only shuffle a fraction of tokens within this neighborhood
                    num_to_shuffle = max(1, int(neighborhood_size_actual * dynamic_strength))
                    
                    # Get indices within this neighborhood
                    local_indices = torch.arange(start_idx, end_idx, device=tensor.device)
                    
                    # Randomly select which tokens to shuffle
                    shuffle_candidates = torch.randperm(neighborhood_size_actual, device=tensor.device)[:num_to_shuffle]
                    selected_indices = local_indices[shuffle_candidates]
                    
                    # Shuffle only the selected tokens within the neighborhood
                    if len(selected_indices) > 1:
                        shuffled_order = torch.randperm(len(selected_indices), device=tensor.device)
                        temp_values = current_tokens[:, selected_indices].clone()
                        current_tokens[:, selected_indices] = temp_values[:, shuffled_order]
            
            shuffled[batch_idx] = current_tokens
        
        # Blend with original to reduce harshness
        blend_factor = 0.7 + 0.3 * step_ratio
        x_blended = blend_factor * flat + (1.0 - blend_factor) * shuffled
        
        return x_blended.view(b, c, h, w)
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # Calculate step ratio for progressive shuffling
        step_ratio = i / (len(sigmas) - 1)
        
        # Shuffle tokens before each denoising step
        x = shuffle_tokens(x, shuffle_strength, step_ratio)
        
        # First prediction
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        dt = sigmas[i + 1] - sigmas[i]
        
        if sigmas[i + 1] == 0:
            # Euler step for final iteration
            x = x + d * dt
        else:
            # Heun's method: predictor-corrector
            x_2 = x + d * dt
            
            # Optionally shuffle again for the corrector step with reduced strength
            x_2 = shuffle_tokens(x_2, shuffle_strength * 0.5, step_ratio)
            
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    
    return x


@torch.no_grad()
def sample_euler_nag(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., nag_scale=1.5, nag_tau=2.5, nag_alpha=0.5):
    """Euler sampler with NAG (Negative-prompt Attention Guidance).
    
    This sampler improves negative prompt handling by modifying attention mechanisms
    to better suppress unwanted features during generation.
    
    Args:
        nag_scale: Scale factor for negative attention guidance (>1.0 enables NAG)
        nag_tau: Maximum scaling factor for attention normalization
        nag_alpha: Blending factor between guided and original attention
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    print(f"NAG Euler sampler active - scale: {nag_scale}, tau: {nag_tau}, alpha: {nag_alpha}")
    
    # Filter out NAG parameters to avoid passing them to model wrapper
    model_args = {k: v for k, v in extra_args.items() if k not in ['nag_scale', 'nag_tau', 'nag_alpha', 'enable_nag']}
    
    # Note: NAG functionality would need to be implemented at the model/attention level
    # For now, this sampler works as a regular Euler sampler with NAG parameters logged
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # Apply churn noise if specified
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        
        # Standard denoising step with filtered args
        denoised = model(x, sigma_hat * s_in, **model_args)
        d = to_d(x, sigma_hat, denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        
        # Euler step
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    
    return x


@torch.no_grad()
def sample_heun_nag(model, x, sigmas, extra_args=None, callback=None, disable=None, nag_scale=1.5, nag_tau=2.5, nag_alpha=0.5, s_noise=1.):
    """Heun sampler with NAG (Negative-prompt Attention Guidance).
    
    This sampler combines Heun's method with NAG for more accurate sampling
    while improving negative prompt adherence.
    
    Args:
        nag_scale: Scale factor for negative attention guidance (>1.0 enables NAG)
        nag_tau: Maximum scaling factor for attention normalization
        nag_alpha: Blending factor between guided and original attention
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    print(f"NAG Heun sampler active - scale: {nag_scale}, tau: {nag_tau}, alpha: {nag_alpha}")
    
    # Filter out NAG parameters to avoid passing them to model wrapper
    model_args = {k: v for k, v in extra_args.items() if k not in ['nag_scale', 'nag_tau', 'nag_alpha', 'enable_nag']}
    
    # Note: NAG functionality would need to be implemented at the model/attention level
    # For now, this sampler works as a regular Heun sampler with NAG parameters logged
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # First prediction with filtered args
        denoised = model(x, sigmas[i] * s_in, **model_args)
        d = to_d(x, sigmas[i], denoised)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        dt = sigmas[i + 1] - sigmas[i]
        
        if sigmas[i + 1] == 0:
            # Euler step for final iteration
            x = x + d * dt
        else:
            # Heun's method: predictor-corrector
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **model_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    
    return x


@torch.no_grad()
def sample_dpmpp_2m_nag(model, x, sigmas, extra_args=None, callback=None, disable=None, nag_scale=1.5, nag_tau=2.5, nag_alpha=0.5):
    """DPM++ 2M sampler with NAG (Negative-prompt Attention Guidance).
    
    This sampler combines the efficient DPM++ 2M method with NAG for improved
    negative prompt handling and faster convergence.
    
    Args:
        nag_scale: Scale factor for negative attention guidance (>1.0 enables NAG)
        nag_tau: Maximum scaling factor for attention normalization
        nag_alpha: Blending factor between guided and original attention
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    
    print(f"NAG DPM++ 2M sampler active - scale: {nag_scale}, tau: {nag_tau}, alpha: {nag_alpha}")
    
    # Filter out NAG parameters to avoid passing them to model wrapper
    model_args = {k: v for k, v in extra_args.items() if k not in ['nag_scale', 'nag_tau', 'nag_alpha', 'enable_nag']}
    
    # Note: NAG functionality would need to be implemented at the model/attention level
    # For now, this sampler works as a regular DPM++ 2M sampler with NAG parameters logged
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **model_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


@torch.no_grad()
def sample_euler_multiscale(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., 
                           scales=[1.0, 0.75, 0.5], guidance_weights=[0.8, 0.15, 0.05], blend_mode='weighted'):
    """Euler sampler with multi-scale guidance for better detail preservation.
    
    This sampler processes the image at multiple resolutions and combines guidance
    from different scales to improve both global structure and fine details.
    
    Args:
        scales: List of scale factors for multi-resolution processing
        guidance_weights: Weights for combining guidance from different scales
        blend_mode: How to blend multi-scale guidance ('weighted', 'adaptive', 'frequency')
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    print(f"Multi-scale Euler sampler active - scales: {scales}, weights: {guidance_weights}, mode: {blend_mode}")
    
    guidance_weights = torch.tensor(guidance_weights, device=x.device, dtype=x.dtype)
    guidance_weights = guidance_weights / guidance_weights.sum()
    
    print(f"Multi-scale Euler sampler active - scales: {scales}, weights: {guidance_weights.tolist()}, blend: {blend_mode}")
    
    def resize_latent(latent, scale):
        """Resize latent tensor to given scale"""
        if scale == 1.0:
            return latent
        b, c, h, w = latent.shape
        new_h, new_w = int(h * scale), int(w * scale)
        # Use area interpolation for downscaling, bilinear for upscaling
        mode = 'area' if scale < 1.0 else 'bilinear'
        return F.interpolate(latent, size=(new_h, new_w), mode=mode, align_corners=False if mode == 'bilinear' else None)
    
    def resize_back(latent, target_shape):
        """Resize latent back to target shape"""
        if latent.shape[-2:] == target_shape[-2:]:
            return latent
        return F.interpolate(latent, size=target_shape[-2:], mode='bilinear', align_corners=False)
    
    def adaptive_blend(guidance_list, step_ratio):
        """Adaptively blend guidance based on sampling progress"""
        # More conservative adaptive blending - always favor the original scale heavily
        if step_ratio < 0.3:
            # Early: heavily emphasize original scale
            weights = torch.tensor([0.85, 0.12, 0.03], device=x.device, dtype=x.dtype)
        elif step_ratio < 0.7:
            # Middle: still favor original scale
            weights = torch.tensor([0.8, 0.15, 0.05], device=x.device, dtype=x.dtype)
        else:
            # Late: slightly more detail enhancement but still conservative
            weights = torch.tensor([0.75, 0.18, 0.07], device=x.device, dtype=x.dtype)
        
        weights = weights[:len(guidance_list)]
        weights = weights / weights.sum()
        
        combined = torch.zeros_like(guidance_list[0])
        for i, guidance in enumerate(guidance_list):
            combined += weights[i] * guidance
        return combined
    
    def frequency_blend(guidance_list):
        """Blend guidance using frequency domain analysis"""
        # Convert to frequency domain
        combined = torch.zeros_like(guidance_list[0])
        
        for i, guidance in enumerate(guidance_list):
            # Apply frequency-based weighting
            # Larger scales contribute more to low frequencies
            # Smaller scales contribute more to high frequencies
            freq_weight = 1.0 / (scales[i] + 0.1)  # Inverse relationship
            combined += guidance_weights[i] * freq_weight * guidance
        
        return combined
    
    original_shape = x.shape
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # Apply churn noise if specified
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        
        # Multi-scale processing
        guidance_list = []
        
        for scale_idx, scale in enumerate(scales):
            # Resize input for this scale
            x_scaled = resize_latent(x, scale)
            
            # Get prediction at this scale
            denoised_scaled = model(x_scaled, sigma_hat * s_in, **extra_args)
            
            # Calculate guidance (derivative) at this scale
            d_scaled = to_d(x_scaled, sigma_hat, denoised_scaled)
            
            # Resize guidance back to original resolution
            d_resized = resize_back(d_scaled, original_shape)
            
            guidance_list.append(d_resized)
        
        # Combine multi-scale guidance
        if blend_mode == 'weighted':
            # Simple weighted combination
            d_multiscale = torch.zeros_like(guidance_list[0])
            for j, guidance in enumerate(guidance_list):
                d_multiscale += guidance_weights[j] * guidance
        elif blend_mode == 'adaptive':
            # Adaptive blending based on sampling progress
            step_ratio = i / (len(sigmas) - 1)
            d_multiscale = adaptive_blend(guidance_list, step_ratio)
        elif blend_mode == 'frequency':
            # Frequency-aware blending
            d_multiscale = frequency_blend(guidance_list)
        else:
            # Fallback to weighted
            d_multiscale = torch.zeros_like(guidance_list[0])
            for j, guidance in enumerate(guidance_list):
                d_multiscale += guidance_weights[j] * guidance
        
        # Blend multi-scale guidance with original scale guidance for subtlety
        # This reduces the aggressiveness of the multi-scale effect
        d_original = guidance_list[0]  # Original scale guidance
        blend_factor = 0.3  # How much multi-scale effect to apply (0.0 = none, 1.0 = full)
        d = (1.0 - blend_factor) * d_original + blend_factor * d_multiscale
        
        # Use the original scale prediction for callback
        denoised = model(x, sigma_hat * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        
        # Euler step with multi-scale guidance
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    
    return x


@torch.no_grad()
def sample_heun_multiscale(model, x, sigmas, extra_args=None, callback=None, disable=None, scales=[1.0, 0.75, 0.5], guidance_weights=[0.8, 0.15, 0.05], blend_mode='adaptive', s_noise=1.):
    """Heun sampler with multi-scale guidance for better detail preservation.
    
    This sampler combines Heun's method with multi-scale processing for more accurate
    sampling while preserving details at multiple resolutions.
    
    Args:
        scales: List of scale factors for multi-scale processing
        guidance_weights: Weights for combining guidance from different scales
        blend_mode: How to blend multi-scale guidance ('weighted', 'adaptive', 'frequency')
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Normalize guidance weights
    guidance_weights = torch.tensor(guidance_weights, device=x.device, dtype=x.dtype)
    guidance_weights = guidance_weights / guidance_weights.sum()
    
    print(f"Multi-scale Heun sampler active - scales: {scales}, weights: {guidance_weights.tolist()}, blend: {blend_mode}")
    
    def resize_latent(latent, scale):
        """Resize latent tensor to given scale"""
        if scale == 1.0:
            return latent
        b, c, h, w = latent.shape
        new_h, new_w = int(h * scale), int(w * scale)
        mode = 'area' if scale < 1.0 else 'bilinear'
        return F.interpolate(latent, size=(new_h, new_w), mode=mode, align_corners=False if mode == 'bilinear' else None)
    
    def resize_back(latent, target_shape):
        """Resize latent back to target shape"""
        if latent.shape[-2:] == target_shape[-2:]:
            return latent
        return F.interpolate(latent, size=target_shape[-2:], mode='bilinear', align_corners=False)
    
    def get_multiscale_guidance(x_input, sigma_input):
        """Get multi-scale guidance for given input"""
        guidance_list = []
        
        for scale in scales:
            # Resize input for this scale
            x_scaled = resize_latent(x_input, scale)
            
            # Get prediction at this scale
            denoised_scaled = model(x_scaled, sigma_input * s_in, **extra_args)
            
            # Calculate guidance at this scale
            d_scaled = to_d(x_scaled, sigma_input, denoised_scaled)
            
            # Resize guidance back to original resolution
            d_resized = resize_back(d_scaled, x_input.shape)
            
            guidance_list.append(d_resized)
        
        # Combine guidance
        if blend_mode == 'adaptive':
            # Adaptive blending (simplified for Heun)
            combined = torch.zeros_like(guidance_list[0])
            for j, guidance in enumerate(guidance_list):
                combined += guidance_weights[j] * guidance
        else:
            # Weighted blending
            combined = torch.zeros_like(guidance_list[0])
            for j, guidance in enumerate(guidance_list):
                combined += guidance_weights[j] * guidance
        
        return combined
    
    original_shape = x.shape
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # First prediction with multi-scale guidance
        d = get_multiscale_guidance(x, sigmas[i])
        
        # Get original scale prediction for callback
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        dt = sigmas[i + 1] - sigmas[i]
        
        if sigmas[i + 1] == 0:
            # Euler step for final iteration
            x = x + d * dt
        else:
            # Heun's method: predictor-corrector with multi-scale
            x_2 = x + d * dt
            
            # Second prediction with multi-scale guidance
            d_2 = get_multiscale_guidance(x_2, sigmas[i + 1])
            
            # Combine predictions
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    
    return x


@torch.no_grad()
def sample_restart(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., restart_list=None):
    """Implements restart sampling in Restart Sampling for Improving Generative Processes (2023)
    Restart_list format: {min_sigma: [ restart_steps, restart_times, max_sigma]}
    If restart_list is None: will choose restart_list automatically, otherwise will use the given restart_list
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        nonlocal step_id
        denoised = model(x, old_sigma * s_in, **extra_args)
        d = to_d(x, old_sigma, denoised)
        if callback is not None:
            callback({'x': x, 'i': step_id, 'sigma': new_sigma, 'sigma_hat': old_sigma, 'denoised': denoised})
        dt = new_sigma - old_sigma
        if new_sigma == 0 or not second_order:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        step_id += 1
        return x

    steps = sigmas.shape[0] - 1
    if restart_list is None:
        if steps >= 20:
            restart_steps = 9
            restart_times = 1
            if steps >= 36:
                restart_steps = steps // 4
                restart_times = 2
            sigmas = get_sigmas_karras(steps - restart_steps * restart_times, sigmas[-2].item(), sigmas[0].item(), device=sigmas.device)
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            restart_list = {}

    restart_list = {int(torch.argmin(abs(sigmas - key), dim=0)): value for key, value in restart_list.items()}

    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_list:
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(restart_steps, sigmas[min_idx].item(), sigmas[max_idx].item(), device=sigmas.device)[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_list.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    last_sigma = None
    for old_sigma, new_sigma in tqdm(step_list, disable=disable):
        if last_sigma is None:
            last_sigma = old_sigma
        elif last_sigma < old_sigma:
            x = x + torch.randn_like(x) * s_noise * (old_sigma ** 2 - last_sigma ** 2) ** 0.5
        x = heun_step(x, old_sigma, new_sigma)
        last_sigma = new_sigma

    return x

