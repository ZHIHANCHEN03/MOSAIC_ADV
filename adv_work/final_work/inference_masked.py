
import os
import json
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FluxPipeline
from src.flux_omini import Condition, generate
from utils import process_image
import argparse
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
    if kernel_size > 0:
        # Ensure kernel size is odd
        if kernel_size % 2 == 0: kernel_size += 1
        masks = TF.gaussian_blur(masks, kernel_size=kernel_size)
    
    # Flatten spatial dims: [Num_Subjects, Seq_Len] -> [Seq_Len, Num_Subjects]
    masks = masks.flatten(1).transpose(0, 1)
    
    return masks

def install_spatial_attn_patch(penalty_strength: float = 10.0):
    """
    Monkey-patch `src.flux_omini_mosaic.attn_forward` with a version that injects an
    additive attention bias for the *image query branch* attending to *condition/ref branches*,
    using a latent-level spatial mask.

    The spatial mask is read from `src.flux_omini_mosaic.SPATIAL_MASK` and should have shape:
      [image_seq_len, num_subjects]
    where `num_subjects == number of conditions` in the current generation call.
    """
    import math
    import src.flux_omini_mosaic as mosaic

    if getattr(mosaic, "_SPATIAL_ATTN_PATCH_INSTALLED", False):
        return

    from diffusers.models.embeddings import apply_rotary_emb

    orig_specify_lora = mosaic.specify_lora

    def _masked_attn_forward(
        attn,
        hidden_states,
        adapters,
        hidden_states2=[],
        position_embs=None,
        group_mask=None,
        cache_mode=None,
        to_cache=None,
        cache_storage=None,
        get_attn_maps=False,
        **kwargs,
    ):
        bs, _, _ = hidden_states[0].shape
        h2_n = len(hidden_states2)

        queries, keys, values = [], [], []

        # Text branch
        for hidden_state in hidden_states2:
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

        # Image + condition branches
        for i, hidden_state in enumerate(hidden_states):
            with orig_specify_lora((attn.to_q, attn.to_k, attn.to_v), adapters[i + h2_n]):
                query = attn.to_q(hidden_state)
                key = attn.to_k(hidden_state)
                value = attn.to_v(hidden_state)

            head_dim = key.shape[-1] // attn.heads
            reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)
            query, key, value = map(reshape_fn, (query, key, value))
            query, key = attn.norm_q(query), attn.norm_k(key)

            queries.append(query)
            keys.append(key)
            values.append(value)

        if position_embs is not None:
            queries = [apply_rotary_emb(q, position_embs[i]) for i, q in enumerate(queries)]
            keys = [apply_rotary_emb(k, position_embs[i]) for i, k in enumerate(keys)]

        if cache_mode == "write":
            for i, (k, v) in enumerate(zip(keys, values)):
                if to_cache[i]:
                    cache_storage[attn.cache_idx][0].append(k)
                    cache_storage[attn.cache_idx][1].append(v)

        if get_attn_maps:
            # Keep compatibility: this mode is used elsewhere for visualization.
            attn_maps = []
            key_img = keys[1]

        attn_outputs = []
        spatial_mask = getattr(mosaic, "SPATIAL_MASK", None)

        for i, query in enumerate(queries):
            keys_, values_ = [], []
            branch_k_meta = []  # list of (branch_index, k_len)

            for j, (k, v) in enumerate(zip(keys, values)):
                if (group_mask is not None) and not (group_mask[i][j].item()):
                    continue
                keys_.append(k)
                values_.append(v)
                branch_k_meta.append((j, k.shape[2]))

            if cache_mode == "read":
                keys_.extend(cache_storage[attn.cache_idx][0])
                values_.extend(cache_storage[attn.cache_idx][1])

            k_cat = torch.cat(keys_, dim=2)
            v_cat = torch.cat(values_, dim=2)

            attn_mask = None
            # Inject only for: image-query branch (index 1) attending to condition branches (>=2)
            if spatial_mask is not None and i == 1:
                num_subjects = spatial_mask.shape[1]
                # condition branches are expected to be 2..(2+num_subjects-1)
                bias_chunks = []
                img_seq_len = query.shape[2]
                if spatial_mask.shape[0] == img_seq_len and num_subjects > 0:
                    cursor = 0
                    for (branch_idx, k_len) in branch_k_meta:
                        if branch_idx >= 2:
                            subj_idx = branch_idx - 2
                            if 0 <= subj_idx < num_subjects:
                                subj_mask = spatial_mask[:, subj_idx].to(device=query.device, dtype=query.dtype)
                                # mask=1 -> bias 0 ; mask=0 -> bias = -penalty_strength
                                subj_bias = (subj_mask - 1.0) * float(penalty_strength)  # [img_seq_len]
                                bias_chunks.append(subj_bias.view(1, 1, -1, 1).expand(1, 1, img_seq_len, k_len))
                            else:
                                bias_chunks.append(torch.zeros((1, 1, img_seq_len, k_len), device=query.device, dtype=query.dtype))
                        else:
                            bias_chunks.append(torch.zeros((1, 1, img_seq_len, k_len), device=query.device, dtype=query.dtype))
                        cursor += k_len

                    if bias_chunks:
                        attn_mask = torch.cat(bias_chunks, dim=-1)  # [1,1,Q,K]

            # Use torch SDPA if no mask, else manual attention for additive mask.
            if attn_mask is None:
                attn_output = mosaic.F.scaled_dot_product_attention(query, k_cat, v_cat).to(query.dtype)
            else:
                scale = 1.0 / math.sqrt(head_dim)
                scores = torch.matmul(query, k_cat.transpose(-2, -1)) * scale  # [bs,h,Q,K]
                scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)
                probs = torch.softmax(scores, dim=-1)
                attn_output = torch.matmul(probs, v_cat).to(query.dtype)

            attn_output = attn_output.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)
            attn_outputs.append(attn_output)

            if get_attn_maps and i >= 2:
                attn_score = torch.einsum("bhnd,bhmd->bhnm", query, key_img)
                attn_map = mosaic.F.softmax(attn_score / (head_dim ** 0.5), dim=-1)
                attn_map = attn_map.mean(dim=1)
                attn_maps.append(attn_map)

        h_out, h2_out = [], []
        for i in range(len(hidden_states2)):
            h2_out.append(attn.to_add_out(attn_outputs[i]))

        for i in range(len(hidden_states)):
            h = attn_outputs[i + h2_n]
            if getattr(attn, "to_out", None) is not None:
                with orig_specify_lora((attn.to_out[0],), adapters[i + h2_n]):
                    h = attn.to_out[0](h)
            h_out.append(h)

        if get_attn_maps:
            attn_maps = torch.cat(attn_maps, dim=1) if len(attn_maps) else torch.empty(0, device=query.device)
            return (h_out, h2_out, attn_maps) if h2_n else (h_out, attn_maps)

        return (h_out, h2_out) if h2_n else h_out

    mosaic.attn_forward = _masked_attn_forward
    mosaic._SPATIAL_ATTN_PATCH_INSTALLED = True


def run_inference(pipe, args):
    with open(args.json_path, 'r') as f:
        data_list = json.load(f)

    # Install patch once per process.
    install_spatial_attn_patch(penalty_strength=args.penalty_strength)
    import src.flux_omini_mosaic as mosaic
    
    layout_results = {} # Store layouts for evaluation
    
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
        layout_result = generate_layout(prompt, len(ref_imgs), 512, 512)
        if isinstance(layout_result, dict):
            bboxes = layout_result.get("bboxes", [])
            interaction_level = layout_result.get("interaction", "none")
        else:
            bboxes = layout_result
            interaction_level = "none"
        
        layout_results[str(index)] = layout_result
        
        kernel_size = args.kernel_size
        alpha = 1.0
        if interaction_level == "strong":
            kernel_size = kernel_size + 8
            alpha = 0.6
        elif interaction_level == "weak":
            kernel_size = kernel_size + 4
            alpha = 0.8
        
        mask = create_soft_mask(bboxes, 512, 512, kernel_size=kernel_size).to(pipe.device)
        if alpha < 1.0:
            mask = mask * alpha + (1.0 - alpha)
        
        # 3. Publish Mask to Mosaic module (so patched attn_forward can see it)
        mosaic.SPATIAL_MASK = mask
        
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
        mosaic.SPATIAL_MASK = None
    
    # Save all layouts to disk for eval.py
        layout_path = os.path.join(args.output_dir, "layout_results.json")
    with open(layout_path, 'w') as f:
        json.dump(layout_results, f, indent=4)
    print(f"Saved generated layouts to {layout_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="scaling_experiment.json")
    parser.add_argument("--output_dir", type=str, default="./outputs_ours")
    parser.add_argument("--penalty_strength", type=float, default=10.0, help="Strength of attention masking penalty")
    parser.add_argument("--kernel_size", type=int, default=15, help="Gaussian blur kernel size for soft mask")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    pipe.load_lora_weights("ByteDance-FanQie/MOSAIC", weight_name="subject_512.safetensors", adapter_name="subject")
    pipe.set_adapters(["subject"], [1])
    
    run_inference(pipe, args)
