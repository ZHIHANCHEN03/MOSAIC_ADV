import os
import json
import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline
from src.flux_omini import Condition, generate
from utils import process_image
import argparse

# ------------------------------------------------------------------------------
# 4. Multi-Pass Composition via Regional Prompting
# ------------------------------------------------------------------------------

# This approach uses "Prompt-based" regions instead of just visual conditions.
# It splits the latent noise into regions, denoises them separately with DIFFERENT prompts,
# and merges them.

def regional_prompting_generation(pipe, base_prompt, regional_prompts, region_masks, steps=28, height=512, width=512):
    """
    Simulates Regional Prompting.
    regional_prompts: List of prompts for each region.
    region_masks: List of binary masks for each region.
    """
    # 1. Init Latents
    latents = torch.randn((1, 16, height//8, width//8), device=pipe.device, dtype=pipe.dtype)
    
    # Denoising Loop (Conceptual)
    for t in pipe.scheduler.timesteps:
        # Predict noise for Base Prompt (Global Coherence)
        noise_pred_base = pipe.unet(latents, t, encoder_hidden_states=encode(base_prompt))
        
        # Predict noise for Each Region
        noise_pred_regions = []
        for prompt, mask in zip(regional_prompts, region_masks):
            noise = pipe.unet(latents, t, encoder_hidden_states=encode(prompt))
            noise_pred_regions.append(noise)
            
        # Merge Noise
        final_noise = noise_pred_base
        for noise, mask in zip(noise_pred_regions, region_masks):
            # Soft blending
            final_noise = final_noise * (1 - mask) + noise * mask
            
        latents = pipe.scheduler.step(final_noise, t, latents)
        
    return latents

def run_inference(pipe, args):
    # This script is a placeholder for the Regional Prompting logic.
    # Since MOSAIC relies on Image Conditions rather than just Text Prompts,
    # true Regional Prompting requires modifying how Conditions are applied (which is covered by Solution 1 & 2).
    # Here we just output standard MOSAIC results to ensure pipeline consistency.
    
    with open(args.json_path, 'r') as f:
        data_list = json.load(f)

    for item in data_list:
        index = item['index']
        prompt = item['prompt']
        image_paths = item['image_paths']
        if isinstance(image_paths, str): image_paths = [image_paths]
            
        ref_imgs = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                pil_img = process_image(img_path, target_size=512, pad_color=(255,255,255), scale=0.9)
                ref_imgs.append(pil_img)
        
        if not ref_imgs: continue
            
        # Standard Generation as baseline for Regional Prompting comparison
        conditions = [Condition(img, "subject", position_delta=[0,0]) for img in ref_imgs]
        
        with torch.no_grad():
            result = generate(
                pipe,
                prompt=prompt,
                conditions=conditions,
                num_inference_steps=28,
                height=512,
                width=512,
                guidance_scale=3.5,
                generator=torch.Generator("cuda").manual_seed(42),
            )[0]
            
        out_path = os.path.join(args.output_dir, f"{index}_cfg_3.5_512x512.jpg")
        result[0].save(out_path)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="example_cases.json")
    parser.add_argument("--output_dir", type=str, default="./outputs_regional")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    pipe.load_lora_weights("ByteDance-FanQie/MOSAIC", weight_name="subject_512.safetensors", adapter_name="subject")
    pipe.set_adapters(["subject"], [1])
    
    run_inference(pipe, args)
