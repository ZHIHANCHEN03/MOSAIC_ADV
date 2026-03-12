import json
import os
import random

def generate_scaling_cases(output_path="adv_work/final_work/scaling_experiment.json"):
    """
    Generates test cases with 8, 10, 12 subjects.
    Uses existing assets from the repo.
    """
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
    subject_counts = [8, 10, 12]
    
    for i, count in enumerate(subject_counts):
        selected_imgs = random.sample(valid_images, count)
        prompt = f"A collection of {count} objects arranged on a wooden table."
        
        cases.append({
            "index": i, 
            "prompt": prompt,
            "image_paths": selected_imgs,
            "num_subjects": count
        })
        
    with open(output_path, 'w') as f:
        json.dump(cases, f, indent=4)
        
    print(f"Generated {len(cases)} scaling test cases in {output_path}")

if __name__ == "__main__":
    generate_scaling_cases()
