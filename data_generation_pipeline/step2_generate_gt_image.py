import os
import json
import random
import argparse
import torch
import traceback
from tqdm import tqdm


def main(args):

    pipe = "YOUR_IMAGE_GENERATION_PIPELINE" # eg. Flux, Qwen-Image, Seed4 ... 

    prompt_dir = args.prompt_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    total_len = len(os.listdir(prompt_dir))
    for i, prompt_file in enumerate(sorted(os.listdir(prompt_dir))):
        try:
            idx = prompt_file.split('.')[0]
            print(f"processing {i}/{total_len}")
            prompt_file = os.path.join(prompt_dir, prompt_file)
            save_path = os.path.join(save_dir, f'{idx}.png')
            if os.path.exists(save_path):
                continue
            with open(prompt_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                prompt = data["prompt"]
                camera_info = data.get("camera_info", "")
                prompt = camera_info + prompt
            
            image = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
            ).images[0]
            image.save(save_path)
        
        except Exception as e:
            traceback.print_exc()
            print(f"\nError processing {os.path.basename(prompt_file)}: {str(e)}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ground truth images from prompts.")
    parser.add_argument("--prompt_dir", type=str, default="data/prompts", help="Directory containing prompt files.")
    parser.add_argument("--save_dir", type=str, default="data/tgt_images", help="Directory to save the generated images.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    
    args = parser.parse_args()
    main(args)