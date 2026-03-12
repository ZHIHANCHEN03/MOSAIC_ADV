import json
import os
import random
import argparse

def generate_scaling_cases(output_path="adv_work/final_work/scaling_experiment.json", subject_counts=None, cases_per_count=10):
    if subject_counts is None:
        subject_counts = [8, 10, 12]
    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Scan for available assets in root assets/
    asset_dir = "assets"
    
    valid_images = []
    if os.path.exists(asset_dir):
        for f in os.listdir(asset_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                valid_images.append(os.path.join("assets", f))
    
    if not valid_images:
        print("Warning: No assets found in root/assets. Creating dummy paths.")
        # Fallback dummy paths if no real assets
        valid_images = [f"assets/dummy_{i}.jpg" for i in range(20)]
        
    # Ensure we have enough unique images
    while len(valid_images) < 12:
        valid_images += valid_images
        
    cases = []
    templates = [
        "A collection of {n} objects arranged on a wooden table.",
        "{n} people standing side by side for a group photo.",
        "{n} people sitting closely together and hugging.",
        "{n} objects arranged in a circle on the floor.",
        "{n} people walking together in a line.",
        "{n} people holding hands in a park."
    ]
    index = 0
    random.seed(42)
    for count in subject_counts:
        for _ in range(cases_per_count):
            selected_imgs = random.sample(valid_images, count)
            template = random.choice(templates)
            prompt = template.format(n=count)
            cases.append({
                "index": index,
                "prompt": prompt,
                "image_paths": selected_imgs,
                "num_subjects": count
            })
            index += 1
        
    with open(output_path, 'w') as f:
        json.dump(cases, f, indent=4)
        
    print(f"Generated {len(cases)} scaling test cases in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="adv_work/final_work/scaling_experiment.json")
    parser.add_argument("--subject_counts", type=str, default="8,10,12")
    parser.add_argument("--cases_per_count", type=int, default=10)
    args = parser.parse_args()
    counts = [int(x.strip()) for x in args.subject_counts.split(",") if x.strip()]
    generate_scaling_cases(args.output_path, counts, args.cases_per_count)
