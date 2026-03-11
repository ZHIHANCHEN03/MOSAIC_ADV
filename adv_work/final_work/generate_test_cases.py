import json
import os
import random

def generate_scaling_cases(output_path="scaling_experiment.json"):
    """
    Generates test cases with 8, 10, 12 subjects.
    Uses existing assets from the repo.
    """
    # Scan for available assets
    asset_dir = "../../assets" # Relative to final_work
    if not os.path.exists(asset_dir):
        # Fallback to absolute if running from root
        asset_dir = "assets"
        
    valid_images = []
    if os.path.exists(asset_dir):
        for f in os.listdir(asset_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                valid_images.append(os.path.join("assets", f))
    
    if not valid_images:
        print("Warning: No assets found. Using dummy paths.")
        valid_images = [f"assets/dummy_{i}.jpg" for i in range(20)]
        
    # Ensure we have enough unique images, duplicate if needed
    while len(valid_images) < 12:
        valid_images += valid_images
        
    cases = []
    subject_counts = [8, 10, 12]
    
    for i, count in enumerate(subject_counts):
        # Randomly select 'count' images
        selected_imgs = random.sample(valid_images, count)
        
        # Generate a prompt
        # We need a prompt that implies a collection or gathering
        prompt = f"A collection of {count} objects arranged on a wooden table, including "
        # Add some descriptors if we knew what they were, but generic is fine for stress test
        prompt += "various items."
        
        cases.append({
            "index": i, # 0=8, 1=10, 2=12
            "prompt": prompt,
            "image_paths": selected_imgs,
            "num_subjects": count
        })
        
    with open(output_path, 'w') as f:
        json.dump(cases, f, indent=4)
        
    print(f"Generated {len(cases)} scaling test cases in {output_path}")

if __name__ == "__main__":
    generate_scaling_cases()
