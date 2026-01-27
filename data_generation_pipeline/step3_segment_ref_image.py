import numpy as np
from PIL import Image, ImageDraw
import os
import json
import glob
from tqdm import tqdm
from PIL import Image
import argparse
import traceback
# https://github.com/luca-medeiros/lang-segment-anything
from lang_sam import LangSAM


def open_vocabulary_detection_and_segmentation(
    langsam_model,
    image_path,
    text_prompts,
    output_dir,
):
    image = Image.open(image_path).convert("RGB")
    # LangSAM
    results = langsam_model.predict(
        [image] * len(text_prompts),
        text_prompts,
        box_threshold=0.30,
        text_threshold=0.25,
    )
    
    for idx, prompt in enumerate(text_prompts):
        if  isinstance(results[idx]['masks'], list) and len(results[idx]['masks']) == 0:
            print(f"Warning: No mask for prompt '{prompt}' in image '{image_path}'.")
            continue
        # mask_np = (results[idx]['masks'].any(axis=0)).astype('uint8') * 255
        mask_np = (results[idx]['masks'].any(axis=0)).astype('uint8') * 255
        mask = Image.fromarray(mask_np, mode='L')

        # mask_area = (mask_np > 0).sum()
        # image_area = image.width * image.height
        # if mask_area / image_area < 0.01:
        #     print(f"Warning: Mask area {mask_area} is less than 1% of image area {image_area}, skipping ref image generation.")
        #     continue

        mask_path = os.path.join(output_dir, f"{idx}_mask.png")
        mask.save(mask_path)

        image_np = np.array(image)
        mask_bool = mask_np > 0
        
        white_bg = np.full_like(image_np, 255)
        masked_np = np.where(mask_bool[..., np.newaxis], image_np, white_bg) # 白底
        # masked_np = image_np

        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)
        if not np.any(rows) or not np.any(cols):
            print(f"Warning: Empty mask for {image_path} with prompt '{prompt}', skipping ref image generation.")
            continue

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        masked_image = Image.fromarray(masked_np)
        cropped_image = masked_image.crop((xmin, ymin, xmax + 1, ymax + 1))
        
        # Maintain the aspect ratio, scale to 512, and center it on a white background.
        original_width, original_height = cropped_image.size
        ratio = 512.0 / max(original_width, original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        resized_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)

        padded_image = Image.new("RGB", (512, 512), (255, 255, 255))
        paste_x = (512 - new_width) // 2
        paste_y = (512 - new_height) // 2
        padded_image.paste(resized_image, (paste_x, paste_y))
        
        ref_image_path = os.path.join(output_dir, f"{idx}_entity.png")
        padded_image.save(ref_image_path)

        print(f"Saved mask and entity for prompt '{prompt}' in '{output_dir}'")


def main(args):
    
    langsam_model = LangSAM(sam_type="sam2.1_hiera_large")

    os.makedirs(args.output_dir, exist_ok=True)
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.tgt_images_dir, ext)))

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(args.prompts_dir, f"{image_id}.json")

            if not os.path.exists(json_path):
                print(f"Warning: Prompt file not found for {image_path}, skipping.")
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    prompts_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {json_path}, skipping.")
                    continue
            
            prompts = []
            if prompts_data.get("reference1", ""):
                prompts.append(prompts_data.get("reference1", "").get("entity", ""))
            if prompts_data.get("reference2", ""):
                prompts.append(prompts_data.get("reference2", "").get("entity", ""))
            if prompts_data.get("reference3", ""):
                prompts.append(prompts_data.get("reference3", "").get("entity", ""))
            # if prompts_data.get("reference4", ""):
            #     prompts.append(prompts_data.get("reference4", "").get("entity", ""))

            prompts = list(filter(None, prompts))

            if not prompts:
                print(f"Warning: No valid prompts found in {json_path}, skipping.")
                continue

            image_output_dir = os.path.join(args.output_dir, image_id)
            os.makedirs(image_output_dir, exist_ok=True)

            open_vocabulary_detection_and_segmentation(
                langsam_model=langsam_model,
                image_path=image_path,
                text_prompts=prompts,
                output_dir=image_output_dir,
            )
        
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing image {image_path}: {str(e)}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment reference images based on prompts (LangSAM version).")
    parser.add_argument("--tgt_images_dir", type=str, default="data/tgt_images", help="Directory of target images to be segmented.")
    parser.add_argument("--prompts_dir", type=str, default="data/prompts", help="Directory of prompt files.")
    parser.add_argument("--output_dir", type=str, default="data/ref_images", help="Directory to save segmented reference images.")
    args = parser.parse_args()
    main(args)