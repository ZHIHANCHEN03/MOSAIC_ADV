import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from diffusers import FluxPipeline
from src.flux_omini import Condition, generate
from utils import process_image
import argparse

# ------------------------------------------------------------------------------
# Helper: Grid Layout Generator
# ------------------------------------------------------------------------------
def generate_grid_layout(num_subjects, height, width):
    """
    Simple heuristic to generate bounding boxes for N subjects.
    This splits the canvas into a grid.
    """
    if num_subjects == 1:
        return [[0, 0, height, width]]
    
    # Calculate grid size (rows x cols)
    cols = int(np.ceil(np.sqrt(num_subjects)))
    rows = int(np.ceil(num_subjects / cols))
    
    cell_w = width // cols
    cell_h = height // rows
    
    bboxes = []
    for i in range(num_subjects):
        r = i // cols
        c = i % cols
        
        # Add some padding to avoid edge crowding
        pad_h = cell_h // 10
        pad_w = cell_w // 10
        
        y1 = r * cell_h + pad_h
        x1 = c * cell_w + pad_w
        y2 = (r + 1) * cell_h - pad_h
        x2 = (c + 1) * cell_w - pad_w
        
        bboxes.append([y1, x1, y2, x2])
        
    return bboxes

def create_mask_from_bbox(height, width, bbox):
    """
    Creates a binary mask (H, W) where the bbox is 1 (True) and rest is 0 (False).
    bbox: [y1, x1, y2, x2]
    """
    mask = torch.zeros((height, width), dtype=torch.bool)
    y1, x1, y2, x2 = bbox
    mask[y1:y2, x1:x2] = True
    return mask

# ------------------------------------------------------------------------------
# Iterative Generation Logic
# ------------------------------------------------------------------------------
def iterative_generation(
    pipe, 
    prompt, 
    ref_imgs, 
    height=512, 
    width=512, 
    steps=28, 
    guidance_scale=3.5, 
    output_dir="./outputs",
    index=0
):
    print(f"Processing Case {index}: {prompt}")
    
    # 1. Generate Base Image (Global Context)
    # We generate a base image using ONLY the prompt, to establish the scene layout.
    # No subject conditions are applied yet.
    print("  -> Generating Base Scene...")
    with torch.no_grad():
        base_result = pipe(
            prompt, 
            height=height, 
            width=width, 
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]
    
    base_path = os.path.join(output_dir, f"{index}_step0_base.jpg")
    base_result.save(base_path)
    
    current_image = base_result
    
    # 2. Iterative Inpainting
    # Get layout for subjects
    bboxes = generate_grid_layout(len(ref_imgs), height, width)
    
    # We need to encode the current image back to latents for inpainting? 
    # Actually, Flux Omini 'generate' function takes 'conditions'.
    # If we want to keep the background, we should use the base image as a starting point 
    # or use it as a condition? 
    # 
    # Simplified approach for MOSAIC:
    # We run the generation again, but this time we apply conditions with MASKS.
    # MOSAIC's Condition object supports `latent_mask`.
    # 
    # Strategy: 
    # Instead of sequential inpainting (which might lose global coherence),
    # we run ONE pass but with spatially separated conditions (Layout Guidance).
    # This is "Modifying Layer" approach (Engineering via Masking).
    
    conditions = []
    
    # Create a visual debug map for layout
    layout_debug = Image.new("RGB", (width, height), (0,0,0))
    draw = ImageDraw.Draw(layout_debug)
    
    for i, (ref_img, bbox) in enumerate(zip(ref_imgs, bboxes)):
        # Calculate position delta (relative to center or specific logic from original code)
        # Original code used: position_deltas.append([0, -(ref_size * (i + 1)) // 16])
        # Here we just keep the original heuristic or 0, relying on the mask to place it.
        pos_delta = [0, 0] 
        
        # Create Latent Mask
        # Latent space is usually H/8 or H/16. Flux is H/16? 
        # Let's check flux_omini.py... it uses pixel masks and converts them?
        # No, Condition.latent_mask expects a boolean mask of shape (H_lat, W_lat).
        # Flux latent size is H//16, W//16.
        
        H_lat, W_lat = height // 16, width // 16
        
        # Scale bbox to latent space
        y1, x1, y2, x2 = [c // 16 for c in bbox]
        latent_mask = torch.zeros((H_lat, W_lat), dtype=torch.bool)
        latent_mask[y1:y2, x1:x2] = True
        
        # Create Condition
        cond = Condition(
            ref_img, 
            "subject", 
            position_delta=pos_delta, 
            latent_mask=latent_mask
        )
        conditions.append(cond)
        
        # Draw debug
        draw.rectangle([bbox[1], bbox[0], bbox[3], bbox[2]], outline="red", width=3)
        draw.text((bbox[1]+5, bbox[0]+5), f"Subj {i}", fill="red")

    layout_debug.save(os.path.join(output_dir, f"{index}_layout_debug.jpg"))
    
    print("  -> Generating with Layout Constraints...")
    with torch.no_grad():
        # We use the MOSAIC generate function which handles the conditions
        result = generate(
            pipe,
            prompt=prompt,
            conditions=conditions, # List of masked conditions
            num_inference_steps=steps,
            num_images_per_prompt=1,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(42),
        )[0]
        
    final_img = result[0]
    final_path = os.path.join(output_dir, f"{index}_iterative_final.jpg")
    final_img.save(final_path)
    print(f"  -> Saved to {final_path}")
    return final_img

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="example_cases.json")
    parser.add_argument("--output_dir", type=str, default="./outputs_iterative")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    device = "cuda"
    print("Loading FLUX Pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.bfloat16
    ).to(device)

    # Load MOSAIC LoRA
    print("Loading MOSAIC LoRA...")
    pipe.load_lora_weights(
        "ByteDance-FanQie/MOSAIC",
        weight_name=f"subject_512.safetensors",
        adapter_name="subject"
    )
    pipe.set_adapters(["subject"], [1])
    
    # Load Data
    with open(args.json_path, 'r') as f:
        data_list = json.load(f)
        
    for item in data_list:
        index = item['index']
        prompt = item['prompt']
        image_paths = item['image_paths']
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        # Load Ref Images
        ref_imgs = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                pil_img = process_image(img_path, target_size=512, pad_color=(255,255,255), scale=0.9)
                ref_imgs.append(pil_img)
            else:
                print(f"Warning: {img_path} not found.")
        
        if not ref_imgs:
            continue
            
        # Run Iterative / Layout Generation
        iterative_generation(
            pipe, 
            prompt, 
            ref_imgs, 
            output_dir=args.output_dir,
            index=index
        )

if __name__ == "__main__":
    main()
