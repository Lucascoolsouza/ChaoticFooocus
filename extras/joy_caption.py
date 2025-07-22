# Joy Caption - High-quality image captioning model
# Based on: https://huggingface.co/spaces/fancyfeast/llama-joycaption-beta-one-hf-llava

import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration
from modules.config import path_clip_vision
from modules.model_loader import load_file_from_url
import os

global_model = None
global_processor = None

MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"

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
    global global_model, global_processor
    
    if extra_options is None:
        extra_options = {}
    
    try:
        # Initialize model and processor if not already loaded
        if global_model is None or global_processor is None:
            print(f"Loading Joy Caption model: {MODEL_NAME}")
            global_processor = AutoProcessor.from_pretrained(MODEL_NAME)
            global_model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
            global_model.eval()
            print("Joy Caption model loaded successfully!")
        
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
        
        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Format the conversation
        convo_string = global_processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        # Process the inputs
        inputs = global_processor(text=[convo_string], images=[image], return_tensors="pt").to(global_model.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # Generate the captions
        with torch.no_grad():
            generate_ids = global_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_k=None,
                top_p=0.9,
            )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = global_processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return caption.strip()
        
    except Exception as e:
        print(f"Error in Joy Caption: {str(e)}")
        return f"Error generating caption: {str(e)}"

def build_joy_prompt(caption_type, caption_length, extra_options):
    """
    Build the appropriate prompt for Joy Caption based on parameters
    """
    
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
    global global_model, global_processor
    
    if global_model is not None:
        del global_model
        global_model = None
    
    if global_processor is not None:
        del global_processor
        global_processor = None
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Joy Caption model cleared from memory")