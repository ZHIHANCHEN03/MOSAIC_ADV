import json
import os
import random
import argparse
import re
from google import genai

def _extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def _llm_select_case(client, model, candidates, count, interaction_ratio, rng):
    use_interaction = rng.random() < interaction_ratio
    mode = "interaction" if use_interaction else "non_interaction"
    prompt = (
        "You are given a list of candidate subject image filenames. "
        "Pick a coherent subset of size {n} that has a meaningful relationship, "
        "and write a simple, natural prompt that includes all subjects. "
        "Keep the scene easy: single setting, minimal interactions, minimal occlusion, "
        "no fantasy, no complex actions, no crowded background. "
        "Examples of simple scenes: 'objects on a wooden table', 'people standing side by side', "
        "'family sitting on a couch', 'items on a kitchen counter'. "
        "Return JSON only with keys: selected_images (array of filenames), prompt (string). "
        "Mode: {mode}. Candidates: {cands}"
    ).format(n=count, mode=mode, cands=", ".join(candidates))
    resp = client.models.generate_content(model=model, contents=prompt)
    parsed = _extract_json(resp.text or "")
    if not parsed:
        return None
    sel = parsed.get("selected_images", [])
    if len(sel) != count:
        return None
    return {"selected_images": sel, "prompt": parsed.get("prompt", "")}

def generate_scaling_cases(output_path="adv_work/final_work/scaling_experiment.json", subject_counts=None, cases_per_count=10, interaction_ratio=0.3, seed=42, use_llm_selection=False, llm_model="gemini-3.1-flash-lite-preview", candidate_pool_size=30):
    if subject_counts is None:
        subject_counts = [6, 8, 10]
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
    templates_non_interaction = [
        "A collection of {n} objects arranged on a wooden table.",
        "{n} people standing side by side for a group photo.",
        "{n} objects arranged in a circle on the floor.",
        "{n} people walking together in a line."
    ]
    templates_interaction = [
        "{n} people sitting closely together and hugging.",
        "{n} people holding hands in a park.",
        "{n} people dancing together."
    ]
    index = 0
    rng = random.Random(seed)
    client = genai.Client() if use_llm_selection else None
    for count in subject_counts:
        for _ in range(cases_per_count):
            selected_imgs = None
            prompt = None
            if use_llm_selection:
                pool = rng.sample(valid_images, min(candidate_pool_size, len(valid_images)))
                llm_case = _llm_select_case(client, llm_model, [os.path.basename(p) for p in pool], count, interaction_ratio, rng)
                if llm_case:
                    prompt = llm_case["prompt"]
                    selected_imgs = [os.path.join("assets", n) for n in llm_case["selected_images"]]
            if not selected_imgs:
                selected_imgs = rng.sample(valid_images, count)
                use_interaction = rng.random() < interaction_ratio
                template_pool = templates_interaction if use_interaction else templates_non_interaction
                template = rng.choice(template_pool)
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
    parser.add_argument("--subject_counts", type=str, default="6,8,10")
    parser.add_argument("--cases_per_count", type=int, default=10)
    parser.add_argument("--interaction_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_llm_selection", action="store_true")
    parser.add_argument("--llm_model", type=str, default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--candidate_pool_size", type=int, default=30)
    args = parser.parse_args()
    counts = [int(x.strip()) for x in args.subject_counts.split(",") if x.strip()]
    generate_scaling_cases(args.output_path, counts, args.cases_per_count, args.interaction_ratio, args.seed, args.use_llm_selection, args.llm_model, args.candidate_pool_size)
