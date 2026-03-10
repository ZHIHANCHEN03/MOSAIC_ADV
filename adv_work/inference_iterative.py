import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import FluxPipeline
from src.flux_omini import Condition, generate
from utils import process_image
import argparse

# ------------------------------------------------------------------------------
# 3. Iterative "Divide-and-Conquer" Generation (Improved)
# ------------------------------------------------------------------------------

def generate_grid_layout(num_subjects, height, width):
    if num_subjects == 1:
        return [[0, 0, height, width]]
    cols = int(np.ceil(np.sqrt(num_subjects)))
    rows = int(np.ceil(num_subjects / cols))
    cell_w = width // cols
    cell_h = height // rows
    bboxes = []
    for i in range(num_subjects):
        r = i // cols
        c = i % cols
        pad_h, pad_w = cell_h // 10, cell_w // 10
        y1 = r * cell_h + pad_h; x1 = c * cell_w + pad_w
        y2 = (r + 1) * cell_h - pad_h; x2 = (c + 1) * cell_w - pad_w
        bboxes.append([y1, x1, y2, x2])
    return bboxes

def iterative_generation(pipe, prompt, ref_imgs, height=512, width=512, steps=28, guidance_scale=3.5, output_dir="./outputs"):
    # 1. Base Image
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
    
    # 2. Iterative Inpainting
    bboxes = generate_grid_layout(len(ref_imgs), height, width)
    
    # In this refined version, we use MOSAIC's latent_mask to apply conditions selectively.
    # Unlike sequential inpainting (pass 1 -> pass 2), we can try to do it in ONE pass
    # but with multiple conditions masked to different areas.
    # This is essentially "Layout-Guided Generation" but implemented via the `conditions` parameter.
    
    conditions = []
    H_lat, W_lat = height // 16, width // 16
    
    for i, (ref_img, bbox) in enumerate(zip(ref_imgs, bboxes)):
        y1, x1, y2, x2 = [c // 16 for c in bbox]
        latent_mask = torch.zeros((H_lat, W_lat), dtype=torch.bool)
        latent_mask[y1:y2, x1:x2] = True
        
        cond = Condition(ref_img, "subject", position_delta=[0,0], latent_mask=latent_mask)
        conditions.append(cond)
        
    print("  -> Generating with Layout Constraints...")
    with torch.no_grad():
        result = generate(
            pipe,
            prompt=prompt,
            conditions=conditions,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(42),
        )[0]
        
    return result[0]

def run_inference(pipe, args):
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
            
        final_img = iterative_generation(pipe, prompt, ref_imgs, output_dir=args.output_dir)
        
        out_path = os.path.join(args.output_dir, f"{index}_cfg_3.5_512x512.jpg")
        final_img.save(out_path)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="example_cases.json")
    parser.add_argument("--output_dir", type=str, default="./outputs_iterative")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    pipe.load_lora_weights("ByteDance-FanQie/MOSAIC", weight_name="subject_512.safetensors", adapter_name="subject")
    pipe.set_adapters(["subject"], [1])
    
    run_inference(pipe, args)
