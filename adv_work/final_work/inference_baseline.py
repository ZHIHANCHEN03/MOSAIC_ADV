import os
import sys
import json
import torch
import argparse
from torchvision import transforms
from PIL import Image
from diffusers import FluxPipeline

# Ensure we can import from root
sys.path.append(os.getcwd())

from src.flux_omini import Condition, generate
from utils import process_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="example_cases.json")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    print(f"Loading Flux Pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.bfloat16
    ).to(device)

    # Load MOSAIC LoRA
    pipe.load_lora_weights(
        "ByteDance-FanQie/MOSAIC",
        weight_name=f"subject_512.safetensors",
        adapter_name="subject"
    )
    pipe.set_adapters(["subject"], [1])

    max_num_refs = 6
    ref_size = 512
    height = 512
    width = 512
    guidance_scale = 3.5

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"Processing {len(data_list)} cases from {args.json_path}")

    for item in data_list:
        index = item.get('index', 0)
        prompt = item.get('prompt', "")
        print(f"[{index}] Prompt: {prompt}")
        
        image_paths = item.get('image_paths', [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        ref_imgs = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                pil_img = process_image(img_path, target_size=ref_size, pad_color=(255,255,255), scale=0.9)
            else:
                pil_img = Image.new("RGB", (ref_size, ref_size), (0,0,0))
                print(f"Warning: {img_path} not found, using black image")
            ref_imgs.append(pil_img)

        # Basic linear positioning for baseline (just stacking conditions)
        position_deltas = []
        for i in range(len(ref_imgs)):
            position_deltas.append([0, -(ref_size * (i + 1)) // 16])
            
        conditions = [Condition(appearance, "subject", position_deltas[i]) for i, appearance in enumerate(ref_imgs)]
        
        with torch.no_grad():
            result = generate(
                pipe,
                prompt=prompt,
                conditions=conditions,
                num_inference_steps=28,
                num_images_per_prompt=1,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                generator=torch.Generator("cuda").manual_seed(42),
            )[0]
            
        if len(result) == 0:
            print(f"Warning: empty result for {index}")
            continue
            
        result_img = result[0]
        result_img_path = os.path.join(args.output_dir, f"{index}_cfg_{guidance_scale}_{height}x{width}.jpg")
        result_img.save(result_img_path)
        print(f"Saved result to {result_img_path}")

if __name__ == "__main__":
    main()
