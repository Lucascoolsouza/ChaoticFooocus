from enum import IntEnum, Enum

disabled = 'Disabled'
enabled = 'Enabled'
subtle_variation = 'Vary (Subtle)'
strong_variation = 'Vary (Strong)'
upscale_15 = 'Upscale (1.5x)'
upscale_2 = 'Upscale (2x)'
upscale_fast = 'Upscale (Fast 2x)'
ultrasharp = 'Ultrasharp'
realistic_rescaler = 'Realistic Rescaler'
latent_upscale = 'Latent Upscale'
pixelsharpen = 'PixelSharpen'
tghqface8x = 'TGHQFace8x'

remove_background = 'Remove Background'
seamless_tiling = 'Seamless Tiling'

uov_list = [disabled, subtle_variation, strong_variation, upscale_15, upscale_2, upscale_fast, ultrasharp, realistic_rescaler, latent_upscale, pixelsharpen, tghqface8x, remove_background, seamless_tiling]

enhancement_uov_before = "Before First Enhancement"
enhancement_uov_after = "After Last Enhancement"
enhancement_uov_processing_order = [enhancement_uov_before, enhancement_uov_after]

enhancement_uov_prompt_type_original = 'Original Prompts'
enhancement_uov_prompt_type_last_filled = 'Last Filled Enhancement Prompts'
enhancement_uov_prompt_types = [enhancement_uov_prompt_type_original, enhancement_uov_prompt_type_last_filled]

CIVITAI_NO_KARRAS = ["euler", "euler_ancestral", "heun", "dpm_fast", "dpm_adaptive", "ddim", "uni_pc"]

# fooocus: a1111 (Civitai)
KSAMPLER = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "heunpp2": "Heun++2",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "DPM++ 3M SDE",
    "dpmpp_3m_sde_gpu": "DPM++ 3M SDE",
    "ddpm": "DDPM",
    "lcm": "LCM",
    "tcd": "TCD",
    "restart": "Restart",
    "euler_token_shuffle": "Euler Token Shuffle",
    "heun_token_shuffle": "Heun Token Shuffle",
    "euler_nag": "Euler NAG",
    "heun_nag": "Heun NAG",
    "dpmpp_2m_nag": "DPM++ 2M NAG",
    "euler_multiscale": "Euler Multi-Scale",
    "heun_multiscale": "Heun Multi-Scale",
    "dpmpp_sde_gpu_token_shuffle": "DPM++ SDE GPU Token Shuffle",
    "euler_pixel_art": "Euler Pixel Art",
    "heun_pixel_art": "Heun Pixel Art",
    "euler_disco": "Euler Disco",
    "heun_disco": "Heun Disco",
    "psycho_euler": "Psycho Euler",
    "euler_wave_noise": "Euler Wave Noise",
    "euler_forgetful": "Euler Forgetful",
    "euler_drunk_guidance": "Euler Drunk Guidance",
    "euler_chaos_steps": "Euler Chaos Steps"
}

SAMPLER_EXTRA = {
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": ""
}

SAMPLERS = KSAMPLER | SAMPLER_EXTRA

KSAMPLER_NAMES = list(KSAMPLER.keys())

# ! = overcooked, !! = error, !!! = black output, ? = cool unrelated aesthetic
SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple",
                    "ddim_uniform", "lcm", "turbo", "align_your_steps", "tcd",
                    "edm_playground_v2.5"]

SAMPLER_NAMES = KSAMPLER_NAMES + list(SAMPLER_EXTRA.keys())

sampler_list = SAMPLER_NAMES
scheduler_list = SCHEDULER_NAMES

clip_skip_max = 12

default_vae = 'Default (model)'

refiner_swap_method = 'joint'

default_input_image_tab = 'uov_tab'
input_image_tab_ids = ['uov_tab', 'ip_tab', 'inpaint_tab', 'describe_tab', 'enhance_tab', 'metadata_tab']

cn_ip = "ImagePrompt"
cn_ip_face = "FaceSwap"
cn_canny = "PyraCanny"
cn_cpds = "CPDS"

ip_list = [cn_ip, cn_canny, cn_cpds, cn_ip_face]
default_ip = cn_ip

default_parameters = {
    cn_ip: (0.5, 0.6), cn_ip_face: (0.9, 0.75), cn_canny: (0.5, 1.0), cn_cpds: (0.5, 1.0)
}  # stop, weight

output_formats = ['png', 'jpeg', 'webp']

inpaint_mask_models = ['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 'isnet-anime', 'sam']

# Background removal options for enhance
bg_removal_models = ['u2net', 'u2netp', 'u2net_human_seg', 'silueta', 'isnet-general-use']

# Disco Diffusion options (Real Disco Diffusion algorithm)
disco_presets = ['custom', 'psychedelic', 'fractal', 'kaleidoscope', 'dreamy', 'scientific']
disco_transforms = ['translate', 'rotate', 'zoom']  # Real geometric transforms
disco_animation_modes = ['none', 'zoom', 'rotate', 'translate']
disco_symmetry_modes = ['none', 'horizontal', 'vertical', 'radial']
disco_noise_schedules = ['linear', 'cosine', 'exponential']
# CLIP model options for Disco Diffusion
disco_clip_models = [
    'RN50',           # ResNet-50 (fast, good quality)
    'RN101',          # ResNet-101 (slower, better quality)
    'RN50x4',         # ResNet-50 4x (high quality)
    'RN50x16',        # ResNet-50 16x (very high quality)
    'RN50x64',        # ResNet-50 64x (maximum quality, very slow)
    'ViT-B/32',       # Vision Transformer Base 32px patches
    'ViT-B/16',       # Vision Transformer Base 16px patches (better)
    'ViT-L/14',       # Vision Transformer Large 14px patches (high quality)
    'ViT-L/14@336px'  # Vision Transformer Large 336px input (maximum quality)
]
inpaint_mask_cloth_category = ['full', 'upper', 'lower']
inpaint_mask_sam_model = ['vit_b', 'vit_l', 'vit_h']

inpaint_engine_versions = ['None', 'v1', 'v2.5', 'v2.6']
inpaint_option_default = 'Inpaint or Outpaint (default)'
inpaint_option_detail = 'Improve Detail (face, hand, eyes, etc.)'
inpaint_option_modify = 'Modify Content (add objects, change background, etc.)'
inpaint_options = [inpaint_option_default, inpaint_option_detail, inpaint_option_modify]

describe_type_photo = 'Photograph'
describe_type_anime = 'Art/Anime'
describe_type_joy = 'Joy'
describe_types = [describe_type_photo, describe_type_anime, describe_type_joy]

sdxl_aspect_ratios = [
    '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
    '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
    '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
    '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
    '1664*576', '1728*576'
]


class MetadataScheme(Enum):
    FOOOCUS = 'fooocus'
    A1111 = 'a1111'


metadata_scheme = [
    (f'{MetadataScheme.FOOOCUS.value} (json)', MetadataScheme.FOOOCUS.value),
    (f'{MetadataScheme.A1111.value} (plain text)', MetadataScheme.A1111.value),
]


class OutputFormat(Enum):
    PNG = 'png'
    JPEG = 'jpeg'
    WEBP = 'webp'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: c.value, cls))


class PerformanceLoRA(Enum):
    QUALITY = None
    SPEED = None
    EXTREME_SPEED = 'sdxl_lcm_lora.safetensors'
    LIGHTNING = 'sdxl_lightning_4step_lora.safetensors'
    HYPER_SD = 'sdxl_hyper_sd_4step_lora.safetensors'


class Steps(IntEnum):
    QUALITY = 60
    SPEED = 30
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4

    @classmethod
    def keys(cls) -> list:
        return list(map(lambda c: c, Steps.__members__))


class StepsUOV(IntEnum):
    QUALITY = 36
    SPEED = 18
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4


class Performance(Enum):
    QUALITY = 'Quality'
    SPEED = 'Speed'
    EXTREME_SPEED = 'Extreme Speed'
    LIGHTNING = 'Lightning'
    HYPER_SD = 'Hyper-SD'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: (c.name, c.value), cls))

    @classmethod
    def values(cls) -> list:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def by_steps(cls, steps: int | str):
        return cls[Steps(int(steps)).name]

    @classmethod
    def has_restricted_features(cls, x) -> bool:
        if isinstance(x, Performance):
            x = x.value
        return x in [cls.EXTREME_SPEED.value, cls.LIGHTNING.value, cls.HYPER_SD.value]

    def steps(self) -> int | None:
        return Steps[self.name].value if self.name in Steps.__members__ else None

    def steps_uov(self) -> int | None:
        return StepsUOV[self.name].value if self.name in StepsUOV.__members__ else None

    def lora_filename(self) -> str | None:
        return PerformanceLoRA[self.name].value if self.name in PerformanceLoRA.__members__ else None
