
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

def evaluate_case(case, result_image_path, dino_model, clip_model, clip_processor, device):
    metrics = {}
    prompt = case.get('prompt', '')
    ref_image_paths = case.get('image_paths', [])
    if isinstance(ref_image_paths, str): ref_image_paths = [ref_image_paths]
        
    if not os.path.exists(result_image_path):
        return None

    metrics['clip_score'] = get_clip_score(clip_model, clip_processor, result_image_path, prompt, device)
    
    gen_img_tensor = preprocess_image(result_image_path)
    gen_dino_emb = get_dino_embedding(dino_model, gen_img_tensor, device)
    
    identity_scores = []
    for ref_path in ref_image_paths:
        if os.path.exists(ref_path):
            ref_tensor = preprocess_image(ref_path)
            ref_dino_emb = get_dino_embedding(dino_model, ref_tensor, device)
            sim = torch.mm(gen_dino_emb, ref_dino_emb.T).item()
            identity_scores.append(sim)
        else:
            identity_scores.append(0.0)
            
    metrics['identity_scores'] = identity_scores
    metrics['avg_identity_score'] = sum(identity_scores) / len(identity_scores) if identity_scores else 0
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
        metrics = evaluate_case(case, result_image_path, dino_model, clip_model, clip_processor, device)
        if metrics:
            metrics['index'] = index
            all_metrics.append(metrics)
            print(f"Case {index}: CLIP={metrics['clip_score']:.3f}, ID={metrics['avg_identity_score']:.3f}")

    if all_metrics:
        avg_clip = sum(m['clip_score'] for m in all_metrics) / len(all_metrics)
        avg_id = sum(m['avg_identity_score'] for m in all_metrics) / len(all_metrics)
        
        summary = {"avg_clip": avg_clip, "avg_identity": avg_id}
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
