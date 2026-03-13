
import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import torch.nn.functional as F
import glob
import argparse
import ssl
import certifi

# Fix SSL issue
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

def load_dino_model(device):
    print("Loading DINOv2 model...")
    try:
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    except Exception as e:
        print(f"Failed to load DINOv2 via hub: {e}")
        return None
    dino.to(device)
    dino.eval()
    return dino

def load_clip_model(device):
    print("Loading CLIP model...")
    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

def preprocess_image(image, size=224):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(image).unsqueeze(0)

def get_dino_embedding(model, image_tensor, device):
    if model is None:
        return torch.randn(1, 384).to(device)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        embedding = model(image_tensor)
        return F.normalize(embedding, dim=-1)

def get_clip_score(model, processor, image, text, device):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        score = torch.matmul(image_embeds, text_embeds.T).item()
        return score

def crop_image(image_path, bbox):
    """
    Crops image based on bbox [y1, x1, y2, x2].
    """
    img = Image.open(image_path).convert('RGB')
    # bbox is [y1, x1, y2, x2] (top, left, bottom, right) in numpy-style
    # PIL expects (left, top, right, bottom) -> (x1, y1, x2, y2)
    y1, x1, y2, x2 = bbox
    # Ensure bounds
    w, h = img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return img # Fallback to full image if box invalid
        
    return img.crop((x1, y1, x2, y2))

def evaluate_case(case, result_image_path, dino_model, clip_model, clip_processor, device, layout=None):
    metrics = {}
    prompt = case.get('prompt', '')
    ref_image_paths = case.get('image_paths', [])
    if isinstance(ref_image_paths, str): ref_image_paths = [ref_image_paths]
        
    if not os.path.exists(result_image_path):
        return None

    # 1. CLIP Score (Global Alignment)
    metrics['clip_score'] = get_clip_score(clip_model, clip_processor, result_image_path, prompt, device)
    
    # 2. Identity Score (Local if layout available, Global otherwise)
    identity_scores = []
    if isinstance(layout, dict):
        layout = layout.get("bboxes")
    ref_embeddings = []
    for ref_path in ref_image_paths:
        if not os.path.exists(ref_path):
            ref_embeddings.append(None)
            continue
        ref_tensor = preprocess_image(ref_path)
        ref_embeddings.append(get_dino_embedding(dino_model, ref_tensor, device))
    gen_embeddings = []
    for i in range(len(ref_image_paths)):
        if layout and i < len(layout):
            bbox = layout[i]
            target_img = crop_image(result_image_path, bbox)
        else:
            target_img = Image.open(result_image_path).convert('RGB')
        gen_tensor = preprocess_image(target_img)
        gen_embeddings.append(get_dino_embedding(dino_model, gen_tensor, device))
    for i, ref_emb in enumerate(ref_embeddings):
        if ref_emb is None:
            identity_scores.append(0.0)
            continue
        sim = torch.mm(gen_embeddings[i], ref_emb.T).item()
        identity_scores.append(sim)
            
    metrics['identity_scores'] = identity_scores
    metrics['avg_identity_score'] = sum(identity_scores) / len(identity_scores) if identity_scores else 0
    leakage_scores = []
    if layout:
        for i, gen_emb in enumerate(gen_embeddings):
            for j, ref_emb in enumerate(ref_embeddings):
                if i == j or ref_emb is None:
                    continue
                leakage_scores.append(torch.mm(gen_emb, ref_emb.T).item())
    metrics['leakage_scores'] = leakage_scores
    metrics['avg_leakage_score'] = sum(leakage_scores) / len(leakage_scores) if leakage_scores else None
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="example_cases.json")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dino_model = load_dino_model(device)
    clip_model, clip_processor = load_clip_model(device)
    
    if not os.path.exists(args.json_path):
        print(f"JSON not found: {args.json_path}")
        return

    with open(args.json_path, 'r') as f:
        cases = json.load(f)
    
    # Check for layout_results.json
    layout_path = os.path.join(args.output_dir, "layout_results.json")
    layouts = {}
    if os.path.exists(layout_path):
        print(f"Found layout file: {layout_path}. Will use for Local Identity Score.")
        with open(layout_path, 'r') as f:
            layouts = json.load(f)
    else:
        print("No layout file found. Using Global Identity Score.")

    all_metrics = []
    print(f"Evaluating {len(cases)} cases...")
    
    for case in cases:
        index = case.get('index', 0)
        # Look for result image
        search_pattern = os.path.join(args.output_dir, f"{index}_cfg_*.jpg")
        found_files = glob.glob(search_pattern)
        result_files = [f for f in found_files if "compared" not in f]
        
        if not result_files:
            print(f"Case {index}: No image found.")
            continue
            
        result_image_path = result_files[0]
        
        # Get layout for this case if available
        # Keys in JSON are strings
        case_layout = layouts.get(str(index)) or layouts.get(index)
        
        metrics = evaluate_case(case, result_image_path, dino_model, clip_model, clip_processor, device, layout=case_layout)
        if metrics:
            metrics['index'] = index
            all_metrics.append(metrics)
            layout_info = "(Local)" if case_layout else "(Global)"
            print(f"Case {index}: CLIP={metrics['clip_score']:.3f}, ID{layout_info}={metrics['avg_identity_score']:.3f}")

    if all_metrics:
        avg_clip = sum(m['clip_score'] for m in all_metrics) / len(all_metrics)
        avg_id = sum(m['avg_identity_score'] for m in all_metrics) / len(all_metrics)
        leakage_values = [m['avg_leakage_score'] for m in all_metrics if m.get('avg_leakage_score') is not None]
        
        summary = {"avg_clip": avg_clip, "avg_identity": avg_id}
        if leakage_values:
            summary["avg_leakage"] = sum(leakage_values) / len(leakage_values)
        print("\n=== Summary ===")
        print(summary)
        
        out_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(out_path, 'w') as f:
            json.dump({"summary": summary, "details": all_metrics}, f, indent=4)
        print(f"Saved results to {out_path}")
    else:
        print("No metrics computed.")

if __name__ == "__main__":
    main()
