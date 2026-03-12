import os
import json
import argparse
import torch
from diffusers import FluxPipeline
from PIL import Image

from src.flux_omini import Condition, generate
from utils import process_image


def run_inference(pipe, args):
    with open(args.json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    for item in data_list:
        index = item["index"]
        prompt = item["prompt"]
        image_paths = item["image_paths"]
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        ref_imgs = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                pil_img = process_image(
                    img_path, target_size=args.ref_size, pad_color=(255, 255, 255), scale=0.9
                )
            else:
                pil_img = Image.new("RGB", (args.ref_size, args.ref_size), (0, 0, 0))
                print(f"{img_path} not exists, all black")
            ref_imgs.append(pil_img)

        if not ref_imgs:
            continue

        # Keep baseline simple and consistent with original inference.py behavior.
        position_deltas = [[0, 0] for _ in range(len(ref_imgs))]
        conditions = [
            Condition(appearance, "subject", position_deltas[i])
            for i, appearance in enumerate(ref_imgs)
        ]

        print(f"[Baseline] Generating Case {index} with {len(conditions)} subjects...")
        with torch.no_grad():
            result = generate(
                pipe,
                prompt=prompt,
                conditions=conditions,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=1,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(pipe.device).manual_seed(args.seed),
            )[0]

        if len(result) == 0:
            print(f"warning: empty result for {index}")
            continue

        out_path = os.path.join(
            args.output_dir, f"{index}_cfg_{args.guidance_scale}_{args.height}x{args.width}.jpg"
        )
        result[0].save(out_path)
        print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="scaling_experiment.json")
    parser.add_argument("--output_dir", type=str, default="./outputs_baseline")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ref_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to(device)
    pipe.load_lora_weights(
        "ByteDance-FanQie/MOSAIC", weight_name="subject_512.safetensors", adapter_name="subject"
    )
    pipe.set_adapters(["subject"], [1])

    run_inference(pipe, args)


if __name__ == "__main__":
    main()

