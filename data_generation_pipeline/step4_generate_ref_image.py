import os
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from tqdm import tqdm
import glob
import argparse
import random
import json
import traceback


def main(args):
    # init model
    pretrained_model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    pipe = FluxKontextPipeline.from_pretrained(pretrained_model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    
    total_len = len(os.listdir(args.prompts_dir))
    for i, json_name in enumerate(sorted(os.listdir(args.prompts_dir))):
        try:
            json_path = os.path.join(args.prompts_dir, json_name)
            num_id = json_path.split("/")[-1].split(".")[0]
            print(f"processing {i}/{total_len}")
            if not os.path.exists(json_path):
                print(f"Warning: Prompt file not found for {json_path}, skipping.")
                continue
            
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    prompts_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {json_path}, skipping.")
                    continue
            
            entities = []
            if prompts_data.get("reference1", ""):
                entities.append(prompts_data.get("reference1", "").get("entity", ""))
            if prompts_data.get("reference2", ""):
                entities.append(prompts_data.get("reference2", "").get("entity", ""))
            if prompts_data.get("reference3", ""):
                entities.append(prompts_data.get("reference3", "").get("entity", ""))
            # if have more ...
            
            entities = list(filter(None, entities))

            types = []
            if prompts_data.get("reference1", ""):
                types.append(prompts_data.get("reference1", "").get("type", ""))
            if prompts_data.get("reference2", ""):
                types.append(prompts_data.get("reference2", "").get("type", ""))
            if prompts_data.get("reference3", ""):
                types.append(prompts_data.get("reference3", "").get("type", ""))
            # if have more...

            types = list(filter(None, types))

            if not entities:
                print(f"Warning: No valid prompts found in {json_path}, skipping.")
                continue
            
            image_dir = os.path.join(args.images_dir, num_id)
            entity_image_paths = glob.glob(os.path.join(image_dir, '**', '*_entity.png'), recursive=True)
            
            idx = 0
            for entity, entity_type, entity_image_path in zip(entities, types, entity_image_paths):
                output_path = entity_image_path.replace("_entity.png", "_ref.png") # if "background" not in entity_type else os.path.join(image_dir, 'background_fill.png')
                
                # 加载图像
                input_image = load_image(entity_image_path)
                entity = entity.replace("a ", "").replace("an ", "")

                if entity_type == "object":
                    prompt1 = f"Generate a frontal view of the {entity}."
                    prompt2 = f"Generate a side view of the {entity}."
                    if random.random() < 0.7:
                        prompt = prompt1
                    else:
                        prompt = prompt2
                elif entity_type == "people":
                    prompt1 = f"Generate a full-body frontal view of the {entity} standing upright with a neutral expression."
                    prompt2 = f"Generate a upper-body frontal view of the {entity} standing upright with a neutral expression."
                    prob = random.random()
                    if prob < 0.7:
                        prompt = prompt1
                    else:
                        prompt = prompt2
                else:
                    raise ValueError(f"Unknown entity type: {entity_type}")
                image = pipe(
                    image=input_image,
                    prompt=prompt,
                    guidance_scale=3.5
                ).images[0]
                
                image.save(output_path)

                idx += 1
                
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {json_path}: {str(e)}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reference images from entity images.")
    parser.add_argument("--images_dir", type=str, default="data/ref_images")
    parser.add_argument("--prompts_dir", type=str, default="data/prompts")
    args = parser.parse_args()

    main(args)