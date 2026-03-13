
import os
import json
import re
import time
import random
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def is_interaction_prompt(prompt):
    text = prompt.lower()
    keywords = [
        "hug", "kiss", "embrace", "hold hands", "holding hands", "handshake",
        "carry", "carries", "riding", "ride", "piggyback", "high five",
        "dancing", "dance", "arm in arm", "leaning on", "sitting closely"
    ]
    return any(k in text for k in keywords)

def classify_interaction(prompt):
    text = prompt.lower()
    strong = [
        "hug", "kiss", "embrace", "holding hands", "hold hands", "arm in arm",
        "piggyback", "carry", "carries", "sitting closely"
    ]
    weak = [
        "dance", "dancing", "handshake", "high five", "leaning on", "walking together"
    ]
    if any(k in text for k in strong):
        return "strong"
    if any(k in text for k in weak):
        return "weak"
    return "none"

def _clamp_bbox(b, height, width):
    y1, x1, y2, x2 = b
    y1 = max(0, min(height, y1))
    x1 = max(0, min(width, x1))
    y2 = max(0, min(height, y2))
    x2 = max(0, min(width, x2))
    if y2 <= y1:
        y2 = min(height, y1 + 50)
    if x2 <= x1:
        x2 = min(width, x1 + 50)
    return [y1, x1, y2, x2]

def _iou(a, b):
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ay2 - ay1) * (ax2 - ax1))
    area_b = max(1, (by2 - by1) * (bx2 - bx1))
    union = area_a + area_b - inter
    return inter / union

def _layout_score(bboxes):
    if len(bboxes) < 2:
        return 1.0, 0.0
    ious = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            ious.append(_iou(bboxes[i], bboxes[j]))
    return sum(ious) / len(ious), max(ious)

def _layout_ok(bboxes, interaction):
    avg_iou, max_iou = _layout_score(bboxes)
    if interaction == "strong":
        return avg_iou >= 0.05 and max_iou <= 0.85
    if interaction == "weak":
        return avg_iou >= 0.02 and max_iou <= 0.65
    return max_iou <= 0.12

def _adjust_bboxes(bboxes, height, width, min_iou, max_iou, steps=3):
    bboxes = [list(b) for b in bboxes]
    for _ in range(steps):
        centers = []
        sizes = []
        for y1, x1, y2, x2 in bboxes:
            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            sizes.append((x2 - x1, y2 - y1))
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                iou = _iou(bboxes[i], bboxes[j])
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                if iou < min_iou:
                    cx_i = (cx_i * 3 + cx_j) / 4
                    cy_i = (cy_i * 3 + cy_j) / 4
                    cx_j = (cx_j * 3 + cx_i) / 4
                    cy_j = (cy_j * 3 + cy_i) / 4
                if iou > max_iou:
                    dx = cx_i - cx_j
                    dy = cy_i - cy_j
                    cx_i += 0.2 * dx
                    cy_i += 0.2 * dy
                    cx_j -= 0.2 * dx
                    cy_j -= 0.2 * dy
                w_i, h_i = sizes[i]
                w_j, h_j = sizes[j]
                bboxes[i] = _clamp_bbox([cy_i - h_i / 2, cx_i - w_i / 2, cy_i + h_i / 2, cx_i + w_i / 2], height, width)
                bboxes[j] = _clamp_bbox([cy_j - h_j / 2, cx_j - w_j / 2, cy_j + h_j / 2, cx_j + w_j / 2], height, width)
    return bboxes

def generate_interaction_layout(num_subjects, height, width, seed_text):
    rng = random.Random(hash(seed_text) & 0xffffffff)
    base_w = max(120, width // 3)
    base_h = max(120, height // 3)
    center_x = width // 2
    center_y = height // 2
    jitter_x = width // 6
    jitter_y = height // 6
    bboxes = []
    for _ in range(num_subjects):
        cx = center_x + rng.randint(-jitter_x, jitter_x)
        cy = center_y + rng.randint(-jitter_y, jitter_y)
        w = base_w + rng.randint(-base_w // 5, base_w // 5)
        h = base_h + rng.randint(-base_h // 5, base_h // 5)
        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(width, cx + w // 2)
        y2 = min(height, cy + h // 2)
        if x2 <= x1:
            x2 = min(width, x1 + 50)
        if y2 <= y1:
            y2 = min(height, y1 + 50)
        bboxes.append([y1, x1, y2, x2])
    return bboxes

def generate_layout(prompt, num_subjects, height=512, width=512, retries=3):
    """
    Generates bounding boxes for subjects using Gemini-3-flash-preview.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    interaction = classify_interaction(prompt)
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment. Using fallback grid layout.")
        if interaction in ["strong", "weak"]:
            bboxes = generate_interaction_layout(num_subjects, height, width, prompt)
        else:
            bboxes = generate_grid_layout(num_subjects, height, width)
        bboxes = _adjust_bboxes(bboxes, height, width, 0.05 if interaction == "strong" else 0.01, 0.5 if interaction == "strong" else 0.15 if interaction == "weak" else 0.02)
        return {"bboxes": bboxes, "interaction": interaction, "layout_ok": _layout_ok(bboxes, interaction)}

    for attempt in range(retries):
        try:
            client = genai.Client(api_key=api_key)
            
            overlap_hint = "Allow overlapping boxes and keep them close." if interaction in ["strong", "weak"] else "Avoid overlap and keep boxes separated."
            system_instruction = f"""
You are an expert image layout planner. Your task is to generate bounding boxes for {num_subjects} subjects in a {height}x{width} canvas based on a text prompt.
The output must be a valid JSON object where keys are subject indices (0 to {num_subjects-1}) and values are [y_min, x_min, y_max, x_max].
Coordinates must be integers within the canvas range [0, {height}] and [0, {width}].
y_min must be less than y_max, and x_min must be less than x_max.
If the prompt implies interaction (e.g., hugging, riding), boxes should overlap appropriately.
If the prompt implies separation (e.g., side by side), boxes should not overlap significantly.
{overlap_hint}
Do not output any markdown formatting, code blocks, or explanation. Just return the raw JSON string.
"""

            user_prompt = f"Prompt: {prompt}\nCanvas Size: {height}x{width}\nNumber of Subjects: {num_subjects}\nGenerate JSON layout:"

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview", 
                contents=user_prompt,
                config={"system_instruction": system_instruction}
            )
            
            # Clean response
            text = response.text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```"):
                text = re.sub(r'^```(json)?', '', text, flags=re.MULTILINE)
                text = re.sub(r'```$', '', text, flags=re.MULTILINE)
            text = text.strip()
            
            layout = json.loads(text)
            
            # Validate layout
            bboxes = []
            for i in range(num_subjects):
                # Handle string or int keys
                box = layout.get(str(i)) or layout.get(i)
                if not box or len(box) != 4:
                    raise ValueError(f"Invalid box format for subject {i}")
                
                # Ensure integers
                box = [int(c) for c in box]
                
                # Clamp coordinates
                y1, x1, y2, x2 = box
                y1 = max(0, min(height, y1))
                x1 = max(0, min(width, x1))
                y2 = max(0, min(height, y2))
                x2 = max(0, min(width, x2))
                
                # Ensure valid area
                if y2 <= y1: y2 = min(height, y1 + 50)
                if x2 <= x1: x2 = min(width, x1 + 50)
                
                bboxes.append([y1, x1, y2, x2])
                
            if interaction == "strong":
                bboxes = _adjust_bboxes(bboxes, height, width, 0.05, 0.6)
            elif interaction == "weak":
                bboxes = _adjust_bboxes(bboxes, height, width, 0.02, 0.3)
            else:
                bboxes = _adjust_bboxes(bboxes, height, width, 0.0, 0.02)
            if not _layout_ok(bboxes, interaction):
                if interaction in ["strong", "weak"]:
                    bboxes = generate_interaction_layout(num_subjects, height, width, prompt)
                else:
                    bboxes = generate_grid_layout(num_subjects, height, width)
                bboxes = _adjust_bboxes(bboxes, height, width, 0.05 if interaction == "strong" else 0.01, 0.5 if interaction == "strong" else 0.15 if interaction == "weak" else 0.02)
            print(f"Successfully generated layout using Gemini (Attempt {attempt+1}).")
            return {"bboxes": bboxes, "interaction": interaction, "layout_ok": _layout_ok(bboxes, interaction)}

        except Exception as e:
            print(f"Error calling Gemini (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(1) # Wait a bit before retry
    
    print("All attempts failed. Falling back to grid layout.")
    if interaction in ["strong", "weak"]:
        bboxes = generate_interaction_layout(num_subjects, height, width, prompt)
    else:
        bboxes = generate_grid_layout(num_subjects, height, width)
    bboxes = _adjust_bboxes(bboxes, height, width, 0.05 if interaction == "strong" else 0.01, 0.5 if interaction == "strong" else 0.15 if interaction == "weak" else 0.02)
    return {"bboxes": bboxes, "interaction": interaction, "layout_ok": _layout_ok(bboxes, interaction)}

def generate_grid_layout(num_subjects, height, width):
    """
    Fallback: Simple grid layout.
    """
    import numpy as np
    cols = int(np.ceil(np.sqrt(num_subjects)))
    rows = int(np.ceil(num_subjects / cols))
    cell_w = width // cols
    cell_h = height // rows
    bboxes = []
    for i in range(num_subjects):
        r = i // cols
        c = i % cols
        pad_h, pad_w = cell_h // 10, cell_w // 10
        y1 = r * cell_h + pad_h
        x1 = c * cell_w + pad_w
        y2 = (r + 1) * cell_h - pad_h
        x2 = (c + 1) * cell_w - pad_w
        # Ensure within bounds
        y1, x1 = max(0, int(y1)), max(0, int(x1))
        y2, x2 = min(height, int(y2)), min(width, int(x2))
        bboxes.append([y1, x1, y2, x2])
    return bboxes

if __name__ == "__main__":
    # Test
    bbox = generate_layout("A cat and a dog sitting on grass", 2)
    print("Generated Layout:", bbox)
