import os
import json
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FluxPipeline
from src.flux_omini import Condition, generate
from utils import process_image
import argparse
from types import MethodType
from layout_generator import generate_layout

# ------------------------------------------------------------------------------
# Spatial-Aware Attention Masking Implementation
# ------------------------------------------------------------------------------

def create_soft_mask(bboxes, height, width, downsample_ratio=16, kernel_size=15):
    """
    Creates a soft spatial mask tensor [H_lat*W_lat, Num_Subjects].
    """
    H_lat, W_lat = height // downsample_ratio, width // downsample_ratio
    num_subjects = len(bboxes)
    
    # Init mask with background value (0)
    # Shape: [Num_Subjects, H_lat, W_lat] for Gaussian Blur
    masks = torch.zeros((num_subjects, H_lat, W_lat), dtype=torch.float32)
    
    for i, bbox in enumerate(bboxes):
        # BBox is [y1, x1, y2, x2] in pixel space
        y1, x1, y2, x2 = [c // downsample_ratio for c in bbox]
        # Clamp
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(H_lat, y2), min(W_lat, x2)
        
        if y2 > y1 and x2 > x1:
            masks[i, y1:y2, x1:x2] = 1.0
            
    # Apply Gaussian Blur for Soft Edges
    # We treat each channel independently
    masks = TF.gaussian_blur(masks, kernel_size=kernel_size)
    
    # Flatten spatial dims: [Num_Subjects, Seq_Len] -> [Seq_Len, Num_Subjects]
    masks = masks.flatten(1).transpose(0, 1)
    
    return masks

def masked_attn_forward(
    self,
    attn,
    hidden_states,
    adapters,
    hidden_states2=[],
    position_embs=None,
    group_mask=None,
    **kwargs,
):
    """
    Patched attn_forward with Spatial Mask Injection.
    """
    # ... Standard Flux Attention Logic ...
    bs, _, _ = hidden_states[0].shape
    h2_n = len(hidden_states2)

    queries, keys, values = [], [], []

    # Text Branch
    for i, hidden_state in enumerate(hidden_states2):
        query = attn.add_q_proj(hidden_state)
        key = attn.add_k_proj(hidden_state)
        value = attn.add_v_proj(hidden_state)
        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
        query, key, value = map(reshape_fn, (query, key, value))
        query, key = attn.norm_added_q(query), attn.norm_added_k(key)
        queries.append(query)
        keys.append(key)
        values.append(value)

    # Image Branch
    for i, hidden_state in enumerate(hidden_states):
        # Note: In real training code, lora context is used. Here we assume weights are merged or handled.
        query = attn.to_q(hidden_state)
        key = attn.to_k(hidden_state)
        value = attn.to_v(hidden_state)
        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
        query, key, value = map(reshape_fn, (query, key, value))
        queries.append(query)
        keys.append(key)
        values.append(value)

    # Concat
    query = torch.cat(queries, dim=2)
    key = torch.cat(keys, dim=2)
    value = torch.cat(values, dim=2)

    # Attention Scores: [BS, Heads, Q_Len, K_Len]
    attn_score = torch.matmul(query, key.transpose(-2, -1)) / (query.shape[-1] ** 0.5)

    # --- INJECT MASK ---
    # We access the mask from the transformer instance
    spatial_mask = getattr(self, "spatial_mask", None)
    
    if spatial_mask is not None:
        # spatial_mask: [Img_Seq_Len, Num_Subjects]
        # We need to map it to [BS, Heads, Q_Len, K_Len]
        # Key Structure: [Text(512) | Img(1024) | Ref1(256) | Ref2(256) ...]
        
        # Assume standard sizes
        txt_len = 512
        img_len = 1024 # 32x32 latent for 512x512 img (Flux uses H//16)
        ref_len = 256  # 512x512 ref -> scaled down? Check flux_omini logic. 
                       # Usually Ref is encoded to tokens. Let's assume 256 for now or derive it.
        
        # Actually, let's look at the keys shape
        total_k_len = key.shape[-2]
        
        # We only want to mask: Image Query -> Reference Key
        # Image Query Indices: [txt_len : txt_len + img_len]
        # Reference Key Indices: [txt_len + img_len : ]
        
        num_subjects = spatial_mask.shape[1]
        
        # Calculate Ref Token Count per Subject
        # Total Ref Tokens = total_k_len - txt_len - img_len
        # Assuming equal tokens per subject
        start_ref_idx = txt_len + img_len
        if start_ref_idx < total_k_len:
            tokens_per_ref = (total_k_len - start_ref_idx) // num_subjects
            
            # Expand Mask to [1, 1, Img_Len, Total_K_Len]
            # Init with 0 (No penalty)
            mask_bias = torch.zeros((1, 1, img_len, total_k_len), device=attn_score.device, dtype=attn_score.dtype)
            
            penalty = -10.0 # Soft Penalty
            
            for i in range(num_subjects):
                # Get subject mask: [Img_Len] -> (0.0 to 1.0)
                subj_mask = spatial_mask[:, i] # [Img_Len]
                
                # We want to PENALIZE regions where mask is LOW.
                # bias = (mask - 1.0) * penalty_strength (positive value) -> This encourages attention?
                # No, we want to Discourage attention where mask is 0.
                # So if mask=1, bias=0. If mask=0, bias=-10.
                # Formula: (mask - 1.0) * abs(penalty)
                
                subj_bias = (subj_mask - 1.0) * abs(penalty) # [Img_Len]
                
                # Apply to corresponding Ref Tokens
                ref_start = start_ref_idx + i * tokens_per_ref
                ref_end = ref_start + tokens_per_ref
                
                # Broadcast bias to [1, 1, Img_Len, Ref_Tokens_i]
                mask_bias[:, :, :, ref_start:ref_end] = subj_bias.view(1, 1, -1, 1)
                
            # Apply Bias to Attention Score
            # attn_score slice: [:, :, txt_len:txt_len+img_len, :]
            # We need to match Q indices
            attn_score[:, :, txt_len:txt_len+img_len, :] += mask_bias

    attn_probs = torch.softmax(attn_score, dim=-1)
    attn_output = torch.matmul(attn_probs, value)
    
    # Reshape output ... (Standard Flux Logic)
    attn_output = attn_output.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)
    
    # Split back
    h_out, h2_out = [], []
    # Text
    idx = 0
    for i in range(len(hidden_states2)):
        h2_out.append(attn.to_add_out(attn_output[:, idx:idx+hidden_states2[i].shape[1]]))
        idx += hidden_states2[i].shape[1]
    # Image
    for i in range(len(hidden_states)):
        # Skip refs in output
        h = attn_output[:, idx:idx+hidden_states[i].shape[1]]
        if getattr(attn, "to_out", None) is not None:
             h = attn.to_out[0](h) # Simplified projection
        h_out.append(h)
        idx += hidden_states[i].shape[1]

    return (h_out, h2_out) if h2_n else h_out


def run_inference(pipe, args):
    with open(args.json_path, 'r') as f:
        data_list = json.load(f)

    # Patch the Transformer
    # We bind the custom forward method to the transformer instance
    # Note: This is a hacky way to patch. In production, we'd subclass or use hooks.
    # But for a script, it works.
    # We need to patch `attn_forward` inside `src.flux_omini_mosaic`. 
    # But `transformer_forward` calls `attn_forward` passed as argument.
    # So we need to patch where `transformer_forward` is called or pass it.
    
    # The `generate` function in `src/flux_omini.py` calls `pipe.transformer(...)`
    # FluxTransformer2DModel's forward calls `transformer_forward`.
    # `transformer_forward` uses `attn_forward` from global scope or kwarg.
    # We can inject it via kwargs if supported, or monkey patch the module.
    
    import src.flux_omini_mosaic
    src.flux_omini_mosaic.attn_forward = masked_attn_forward # Monkey Patch Global Function
    
    for item in data_list:
        index = item['index']
        prompt = item['prompt']
        image_paths = item['image_paths']
        if isinstance(image_paths, str): image_paths = [image_paths]
        
        ref_imgs = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                pil_img = process_image(img_path, target_size=512, pad_color=(255,255,255), scale=0.9)
                ref_imgs.append(pil_img)
        
        if not ref_imgs: continue

        # 1. Generate Layout (LLM)
        print(f"Generating Layout for Case {index} ({len(ref_imgs)} subjects)...")
        bboxes = generate_layout(prompt, len(ref_imgs), 512, 512)
        
        # 2. Create Soft Mask
        mask = create_soft_mask(bboxes, 512, 512).to(pipe.device)
        
        # 3. Attach Mask to Transformer (so patched attn_forward can see it)
        pipe.transformer.spatial_mask = mask
        
        # 4. Conditions
        conditions = [Condition(img, "subject", position_delta=[0,0]) for img in ref_imgs]

        print(f"Generating Image...")
        with torch.no_grad():
            result = generate(
                pipe,
                prompt=prompt,
                conditions=conditions,
                num_inference_steps=28,
                height=512,
                width=512,
                guidance_scale=3.5,
                generator=torch.Generator("cuda").manual_seed(42),
            )[0]
            
        out_path = os.path.join(args.output_dir, f"{index}_cfg_3.5_512x512.jpg")
        result[0].save(out_path)
        print(f"Saved to {out_path}")
        
        # Cleanup mask for next iteration
        pipe.transformer.spatial_mask = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="scaling_experiment.json")
    parser.add_argument("--output_dir", type=str, default="./outputs_ours")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    pipe.load_lora_weights("ByteDance-FanQie/MOSAIC", weight_name="subject_512.safetensors", adapter_name="subject")
    pipe.set_adapters(["subject"], [1])
    
    run_inference(pipe, args)
