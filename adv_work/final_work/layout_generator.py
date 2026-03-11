import os
import json
import re
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_layout(prompt, num_subjects, height=512, width=512):
    """
    Generates bounding boxes for subjects using Gemini-3-flash-preview.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment. Using fallback grid layout.")
        return generate_grid_layout(num_subjects, height, width)

    try:
        client = genai.Client(api_key=api_key)
        
        system_instruction = f"""
You are an expert image layout planner. Your task is to generate bounding boxes for {num_subjects} subjects in a {height}x{width} canvas based on a text prompt.
The output must be a valid JSON object where keys are subject indices (0 to {num_subjects-1}) and values are [y_min, x_min, y_max, x_max].
Coordinates must be integers within the canvas range.
If the prompt implies interaction (e.g., hugging, riding), boxes should overlap.
If the prompt implies separation (e.g., side by side), boxes should not overlap significantly.
Do not output any markdown formatting or explanation, just the raw JSON string.
"""

        user_prompt = f"Prompt: {prompt}\nCanvas Size: {height}x{width}\nNumber of Subjects: {num_subjects}\nGenerate JSON layout:"

        response = client.models.generate_content(
            model="gemini-3.0-flash-preview", 
            contents=user_prompt,
            config={"system_instruction": system_instruction}
        )
        
        # Clean response (remove markdown code blocks if any)
        text = response.text.strip()
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        
        layout = json.loads(text)
        
        # Validate layout
        bboxes = []
        for i in range(num_subjects):
            box = layout.get(str(i)) or layout.get(i)
            if not box or len(box) != 4:
                print(f"Warning: Invalid box for subject {i}, falling back to grid.")
                return generate_grid_layout(num_subjects, height, width)
            bboxes.append(box)
            
        print("Successfully generated layout using Gemini.")
        return bboxes

    except Exception as e:
        print(f"Error calling Gemini: {e}")
        print("Falling back to grid layout.")
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
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(height, y2), min(width, x2)
        bboxes.append([y1, x1, y2, x2])
    return bboxes

if __name__ == "__main__":
    # Test
    bbox = generate_layout("A cat and a dog sitting on grass", 2)
    print("Generated Layout:", bbox)
