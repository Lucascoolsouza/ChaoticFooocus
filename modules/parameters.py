from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
import numpy as np
from PIL import Image

class PerformancePreset(str, Enum):
    QUALITY = 'Quality'
    SPEED = 'Speed'
    EXTREME_SPEED = 'Extreme Speed'
    LIGHTNING = 'Lightning'
    HYPER_SD = 'Hyper-SD'
    
    @classmethod
    def from_string(cls, value: str) -> 'PerformancePreset':
        """Convert string to PerformancePreset enum"""
        try:
            return cls(value)
        except ValueError:
            # Try case-insensitive match
            value_lower = value.lower()
            for member in cls:
                if member.value.lower() == value_lower:
                    return member
            # Default to QUALITY if no match found
            return cls.QUALITY

@dataclass
class ControlNetTask:
    """Represents a ControlNet task with image and processing parameters"""
    image: Optional[Union[np.ndarray, Image.Image]] = None
    stop: float = 0.0
    weight: float = 1.0
    type: str = "ImagePrompt"
    
    @classmethod
    def from_args(cls, image: Any, stop: float, weight: float, type_str: str) -> 'ControlNetTask':
        """Create from legacy argument format"""
        return cls(
            image=image if image is not None else None,
            stop=float(stop) if stop is not None else 0.0,
            weight=float(weight) if weight is not None else 1.0,
            type=str(type_str) if type_str is not None else "ImagePrompt"
        )

@dataclass
class DrunkUNetParams:
    """Parameters for the Drunk UNet feature"""
    enabled: bool = False
    attn_noise_strength: float = 0.0
    layer_dropout_prob: float = 0.0
    prompt_noise_strength: float = 0.0
    cognitive_echo_strength: float = 0.0
    dynamic_guidance_preset: str = 'None'
    dynamic_guidance_base: float = 7.0
    dynamic_guidance_amplitude: float = 2.0
    
    @classmethod
    def from_args(cls, *args) -> 'DrunkUNetParams':
        """Create from legacy argument format"""
        if not args or len(args) < 8:
            return cls()
            
        return cls(
            enabled=bool(args[0]) if args[0] is not None else False,
            attn_noise_strength=float(args[1]) if args[1] is not None else 0.0,
            layer_dropout_prob=float(args[2]) if args[2] is not None else 0.0,
            prompt_noise_strength=float(args[3]) if args[3] is not None else 0.0,
            cognitive_echo_strength=float(args[4]) if args[4] is not None else 0.0,
            dynamic_guidance_preset=str(args[5]) if args[5] is not None else 'None',
            dynamic_guidance_base=float(args[6]) if args[6] is not None else 7.0,
            dynamic_guidance_amplitude=float(args[7]) if args[7] is not None else 2.0
        )

@dataclass
class DiscoDiffusionParams:
    """Parameters for the Disco Diffusion feature"""
    enabled: bool = False
    scale: float = 0.8
    preset: str = 'custom'
    transforms: List[str] = field(default_factory=lambda: ['translate', 'rotate', 'zoom'])
    seed: Optional[int] = None
    clip_model: str = 'RN50'
    guidance_steps: int = 10
    cutn: int = 16
    tv_scale: float = 150.0
    range_scale: float = 50.0
    
    @classmethod
    def from_args(cls, *args) -> 'DiscoDiffusionParams':
        """Create from legacy argument format"""
        if not args or len(args) < 12:
            return cls()
            
        return cls(
            enabled=bool(args[0]) if args[0] is not None else False,
            scale=float(args[1]) if args[1] is not None else 0.8,
            preset=str(args[2]) if args[2] is not None else 'custom',
            transforms=list(args[3]) if args[3] is not None else ['translate', 'rotate', 'zoom'],
            seed=int(args[4]) if args[4] is not None else None,
            clip_model=str(args[5]) if args[5] is not None else 'RN50',
            guidance_steps=int(args[6]) if args[6] is not None else 10,
            cutn=int(args[7]) if args[7] is not None else 16,
            tv_scale=float(args[8]) if args[8] is not None else 150.0,
            range_scale=float(args[9]) if args[9] is not None else 50.0
        )

@dataclass
class GenerationParameters:
    """Container for all generation parameters with type hints and defaults"""
    # Core generation parameters
    prompt: str = ""
    negative_prompt: str = ""
    style_selections: List[str] = field(default_factory=list)
    performance_preset: PerformancePreset = PerformancePreset.QUALITY
    aspect_ratio: str = "1:1"
    image_number: int = 1
    output_format: str = "png"
    seed: int = -1
    read_wildcards_in_order: bool = False
    
    # Model parameters
    base_model: str = ""
    refiner_model: str = ""
    refiner_switch: float = 0.5
    loras: List[Tuple[bool, str, float]] = field(default_factory=list)
    
    # Image generation parameters
    guidance_scale: float = 7.0
    sharpness: float = 2.0
    base_multiplier: float = 1.0
    artistic_strength: float = 0.0
    
    # Image input/output
    input_image: Optional[Any] = None
    current_tab: str = "uov_tab"
    uov_method: str = ""
    uov_input_image: Optional[Any] = None
    latent_upscale_method: str = ""
    latent_upscale_scheduler: str = ""
    latent_upscale_size: str = ""
    outpaint_selections: List[str] = field(default_factory=list)
    inpaint_input_image: Optional[Any] = None
    inpaint_additional_prompt: str = ""
    inpaint_mask_image_upload: Optional[Any] = None
    
    # UI toggles
    disable_preview: bool = False
    disable_intermediate_results: bool = False
    disable_seed_increment: bool = False
    black_out_nsfw: bool = False
    
    # Advanced parameters
    adm_scaler_positive: float = 1.5
    adm_scaler_negative: float = 0.8
    adm_scaler_end: float = 0.3
    adaptive_cfg: float = 7.0
    clip_skip: int = 1
    sampler_name: str = "euler"
    scheduler_name: str = "normal"
    vae_name: str = ""
    
    # Override parameters
    overwrite_step: int = -1
    overwrite_switch: float = -1
    overwrite_width: int = -1
    overwrite_height: int = -1
    overwrite_vary_strength: float = -1
    overwrite_upscale_strength: float = -1
    
    # ControlNet parameters
    controlnet_tasks: List[ControlNetTask] = field(default_factory=list)
    controlnet_softness: float = 0.25
    
    # Feature parameters
    drunk_unet: DrunkUNetParams = field(default_factory=DrunkUNetParams)
    disco_diffusion: DiscoDiffusionParams = field(default_factory=DiscoDiffusionParams)
    
    # UI state
    generate_image_grid: bool = True
    force_grid_checkbox: bool = False
    
    # Metadata
    metadata_scheme: str = "fooocus"
    save_metadata_to_images: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary for compatibility with existing code"""
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, (DrunkUNetParams, DiscoDiffusionParams)):
                result.update({f"{k}_{field}": field_value for field, field_value in v.__dict__.items() 
                             if not field.startswith('_')})
            elif isinstance(v, list):
                result[k] = v.copy()
            else:
                result[k] = v
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationParameters':
        """Create from a dictionary"""
        params = cls()
        for k, v in data.items():
            if hasattr(params, k):
                setattr(params, k, v)
            elif k.startswith('drunk_unet_'):
                field = k[11:]
                if hasattr(params.drunk_unet, field):
                    setattr(params.drunk_unet, field, v)
            elif k.startswith('disco_diffusion_'):
                field = k[16:]
                if hasattr(params.disco_diffusion, field):
                    setattr(params.disco_diffusion, field, v)
        return params


def convert_legacy_args_to_params(args: List[Any]) -> GenerationParameters:
    """Convert legacy argument list to structured parameters"""
    from modules.config import default_max_lora_number
    from modules.flags import Performance
    
    if not args:
        return GenerationParameters()
        
    # Make a copy and reverse for popping
    args = list(args)
    args.reverse()
    
    params = GenerationParameters()
    
    try:
        # Basic UI parameters
        params.generate_image_grid = bool(args.pop())
        params.force_grid_checkbox = bool(args.pop())
        
        # Core generation parameters
        params.prompt = str(args.pop() or "")
        params.negative_prompt = str(args.pop() or "")
        params.style_selections = list(args.pop() or [])
        params.aspect_ratio = str(args.pop() or "1:1")
        params.image_number = int(args.pop() or 1)
        params.output_format = str(args.pop() or "png")
        params.seed = int(args.pop() or -1)
        params.read_wildcards_in_order = bool(args.pop())
        
        # Model parameters
        params.sharpness = float(args.pop() or 2.0)
        params.guidance_scale = float(args.pop() or 7.0)
        params.base_model = str(args.pop() or "")
        params.refiner_model = str(args.pop() or "")
        params.refiner_switch = float(args.pop() or 0.5)
        
        # Parse LoRAs
        loras = []
        for _ in range(default_max_lora_number):
            enabled = bool(args.pop())
            name = str(args.pop() or "")
            weight = float(args.pop() or 0.0)
            if name:
                loras.append((enabled, name, weight))
        params.loras = loras
        
        # Image input parameters
        params.input_image = args.pop()  # input_image_checkbox
        params.current_tab = str(args.pop() or "uov_tab")
        params.uov_method = str(args.pop() or "")
        params.uov_input_image = args.pop()
        params.latent_upscale_method = str(args.pop() or "")
        params.latent_upscale_scheduler = str(args.pop() or "")
        params.latent_upscale_size = str(args.pop() or "")
        params.outpaint_selections = list(args.pop() or [])
        params.inpaint_input_image = args.pop()
        params.inpaint_additional_prompt = str(args.pop() or "")
        params.inpaint_mask_image_upload = args.pop()
        
        # UI toggles
        params.disable_preview = bool(args.pop())
        params.disable_intermediate_results = bool(args.pop())
        params.disable_seed_increment = bool(args.pop())
        params.black_out_nsfw = bool(args.pop())
        
        # Advanced parameters
        params.adm_scaler_positive = float(args.pop() or 1.5)
        params.adm_scaler_negative = float(args.pop() or 0.8)
        params.adm_scaler_end = float(args.pop() or 0.3)
        params.adaptive_cfg = float(args.pop() or 7.0)
        params.clip_skip = int(args.pop() or 1)
        params.sampler_name = str(args.pop() or "euler")
        params.scheduler_name = str(args.pop() or "normal")
        params.vae_name = str(args.pop() or "")
        
        # Override parameters
        params.overwrite_step = int(args.pop() or -1)
        params.overwrite_switch = float(args.pop() or -1)
        params.overwrite_width = int(args.pop() or -1)
        params.overwrite_height = int(args.pop() or -1)
        params.overwrite_vary_strength = float(args.pop() or -1)
        params.overwrite_upscale_strength = float(args.pop() or -1)
        
        # ControlNet parameters
        controlnet_tasks = []
        for _ in range(4):  # default_controlnet_image_count
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                controlnet_tasks.append(ControlNetTask.from_args(cn_img, cn_stop, cn_weight, cn_type))
        params.controlnet_tasks = controlnet_tasks
        
        params.controlnet_softness = float(args.pop() or 0.25)
        
        # Drunk UNet parameters
        drunk_params = []
        for _ in range(8):
            drunk_params.append(args.pop() if args else None)
        params.drunk_unet = DrunkUNetParams.from_args(*drunk_params)
        
        # Disco Diffusion parameters
        disco_params = []
        for _ in range(12):
            disco_params.append(args.pop() if args else None)
        params.disco_diffusion = DiscoDiffusionParams.from_args(*disco_params)
        
        # Performance selection (should be the last parameter)
        if args:
            perf_value = args.pop()
            if isinstance(perf_value, (int, float)):
                # Handle legacy numeric performance values
                perf_map = {
                    60: PerformancePreset.QUALITY,
                    30: PerformancePreset.SPEED,
                    8: PerformancePreset.EXTREME_SPEED,
                    4: PerformancePreset.LIGHTNING
                }
                params.performance_preset = perf_map.get(perf_value, PerformancePreset.QUALITY)
            else:
                params.performance_preset = PerformancePreset.from_string(str(perf_value))
        
    except Exception as e:
        print(f"[WARNING] Error converting legacy args: {e}")
    
    return params
