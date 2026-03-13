
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
    interaction = is_interaction_prompt(prompt)
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment. Using fallback grid layout.")
        if interaction:
            return generate_interaction_layout(num_subjects, height, width, prompt)
        return generate_grid_layout(num_subjects, height, width)

    for attempt in range(retries):
        try:
            client = genai.Client(api_key=api_key)
            
            overlap_hint = "Allow overlapping boxes and keep them close." if interaction else "Avoid overlap and keep boxes separated."
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
                
            print(f"Successfully generated layout using Gemini (Attempt {attempt+1}).")
            return bboxes

        except Exception as e:
            print(f"Error calling Gemini (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(1) # Wait a bit before retry
    
    print("All attempts failed. Falling back to grid layout.")
    if interaction:
        return generate_interaction_layout(num_subjects, height, width, prompt)
    return generate_grid_layout(num_subjects, height, width)

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
