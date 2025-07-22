# Joy Caption - High-quality image captioning model
# Based on: https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from modules.config import path_clip_vision
from modules.model_loader import load_file_from_url
import os

global_model = None
global_processor = None
global_tokenizer = None

def default_captioner(image_rgb, caption_type="descriptive", caption_length="any", extra_options=None):
    """
    Generate captions using Joy Caption model
    
    Args:
        image_rgb: RGB image array
        caption_type: Type of caption ("descriptive", "training_prompt", "rng-tags")
        caption_length: Length preference ("any", "very_short", "short", "medium_length", "long", "very_long")
        extra_options: Additional options like custom_prompt
    
    Returns:
        Generated caption string
    """
    global global_model, global_processor, global_tokenizer
    
    if extra_options is None:
        extra_options = []
    
    # Model configuration - try different model variants
    model_variants = [
        "meta-llama/Llama-3.2-11B-Vision-Instruct",  # Official model
        "unsloth/llama-3.2-11b-vision-instruct",     # Unsloth variant
        "meta-llama/Llama-3.2-11B-Vision"            # Base model
    ]
    
    try:
        # Initialize model and processor if not already loaded
        if global_model is None or global_processor is None:
            print("Loading Joy Caption model...")
            
            # Try different model variants until one works
            model_loaded = False
            for model_name in model_variants:
                try:
                    print(f"Trying model: {model_name}")
                    global_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                    global_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    global_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    print(f"Successfully loaded model: {model_name}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}: {str(e)}")
                    continue
            
            if not model_loaded:
                # Fallback to a more reliable model
                print("Trying fallback model: microsoft/Florence-2-large")
                try:
                    global_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
                    global_model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/Florence-2-large",
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    global_tokenizer = global_processor.tokenizer
                    print("Successfully loaded fallback model: microsoft/Florence-2-large")
                    model_loaded = True
                except Exception as fallback_error:
                    print(f"Fallback model also failed: {str(fallback_error)}")
                    raise Exception("Failed to load any available vision-language model")
            print("Joy Caption model loaded successfully")
        
        # Convert numpy array to PIL Image
        if isinstance(image_rgb, np.ndarray):
            image = Image.fromarray(image_rgb)
        else:
            image = image_rgb
            
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Build prompt based on caption type and length
        prompt = build_joy_prompt(caption_type, caption_length, extra_options)
        
        # Prepare inputs
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        input_text = global_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = global_processor(
            image, 
            input_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(global_model.device)
        
        # Generate caption
        with torch.no_grad():
            generate_ids = global_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                suppress_tokens=None
            )
        
        # Decode the generated text
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = global_processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
        
    except Exception as e:
        print(f"Error in Joy Caption: {str(e)}")
        return f"Error generating caption: {str(e)}"

def build_joy_prompt(caption_type, caption_length, extra_options):
    """Build the appropriate prompt for Joy Caption based on parameters"""
    
    # Base prompt parts
    length_modifiers = {
        "any": "",
        "very_short": "Write a very short caption.",
        "short": "Write a short caption.",
        "medium_length": "Write a medium-length caption.",
        "long": "Write a long caption.",
        "very_long": "Write a very long caption."
    }
    
    type_prompts = {
        "descriptive": "Write a descriptive caption for this image in a formal tone.",
        "training_prompt": "Write a stable diffusion prompt for this image.",
        "rng-tags": "Write a list of booru-style tags for this image."
    }
    
    # Start with the base prompt
    prompt = type_prompts.get(caption_type, type_prompts["descriptive"])
    
    # Add length modifier if specified
    if caption_length != "any":
        length_mod = length_modifiers.get(caption_length, "")
        if length_mod:
            prompt = f"{length_mod} {prompt}"
    
    # Add any custom options
    if "custom_prompt" in extra_options:
        prompt = extra_options["custom_prompt"]
    
    return prompt

def simple_captioner(image_rgb, prompt="Describe this image in detail."):
    """
    Simplified interface for basic captioning
    
    Args:
        image_rgb: RGB image array
        prompt: Custom prompt for captioning
        
    Returns:
        Generated caption string
    """
    return default_captioner(
        image_rgb, 
        caption_type="descriptive", 
        caption_length="medium_length",
        extra_options={"custom_prompt": prompt}
    )

def training_prompt_captioner(image_rgb):
    """
    Generate training prompts suitable for Stable Diffusion
    
    Args:
        image_rgb: RGB image array
        
    Returns:
        Generated training prompt string
    """
    return default_captioner(
        image_rgb,
        caption_type="training_prompt",
        caption_length="medium_length"
    )

def tag_captioner(image_rgb):
    """
    Generate booru-style tags for the image
    
    Args:
        image_rgb: RGB image array
        
    Returns:
        Generated tags string
    """
    return default_captioner(
        image_rgb,
        caption_type="rng-tags",
        caption_length="any"
    )

# Cleanup function to free memory
def cleanup_joy_caption():
    """Free up memory by clearing the global model"""
    global global_model, global_processor, global_tokenizer
    
    if global_model is not None:
        del global_model
        global_model = None
    
    if global_processor is not None:
        del global_processor
        global_processor = None
        
    if global_tokenizer is not None:
        del global_tokenizer
        global_tokenizer = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Joy Caption model cleared from memory")