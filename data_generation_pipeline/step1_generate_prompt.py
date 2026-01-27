import os
import json
import openai
from tqdm import tqdm
import concurrent.futures
import argparse
import random
import time


def chat(prompt):
    base_url = "YOUR_REQUEST_URL"
    api_version = "YOUR_API_VERSION"
    ak = "YOUR_AK"
    model_name = ""
    max_tokens = 4096  # range: [0, 4096]
    client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            api_version=api_version,
            api_key=ak,
        )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ],
        temperature=1.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}
    )

    result = json.loads(completion.model_dump_json())
    content = result['choices'][0]['message']['content']
    return content


def _process_single_request(idx, save_path, prompt):
    max_retries = 10
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            content = chat(prompt)
            with open(save_path, 'w', encoding='utf-8') as file:
                file.write(content)
            return
        except Exception as e:
            tqdm.write(f"\nError processing prompt {idx} on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                tqdm.write(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                tqdm.write(f"All retries failed for prompt {idx}. Skipping.")


def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    max_workers = args.max_workers

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx in range(args.num_prompts):
            prompt = random.choice(args.prompts)
            prompt += "\n Only generate and output one set of results."
            save_path = os.path.join(output_dir, f'{idx}.json')
            if os.path.exists(save_path):
                continue
            future = executor.submit(_process_single_request, idx, save_path, prompt)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating Prompts"):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate prompts using GPT.")
    parser.add_argument("--output_dir", type=str, default="data/prompts", help="Directory to save the generated prompts.")
    parser.add_argument("--num_prompts", type=int, default=1, help="Number of prompts to generate.")
    parser.add_argument("--ref_num", type=int, default=3, help="Number of reference to generate.")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker threads.")
    parser.add_argument("--prompts", type=list, default=None)
    args = parser.parse_args()

    # prompt example
    prompt = f""" 
    Please generate a prompt for a real-life scene featuring exactly 3 characters, ensuring diversity in gender, age, identity, appearance, etc.
        - **reference1~3**: Fill in only the character's identity.
        - **background**: Fill in only the background description.
        - **prompt**: Write a prompt of about 120 words describing the generated image (including the scene and the characters). The actions and interactions of the characters should be diverse. Note: The scene must not include any characters other than the four specified!!!
        - **masked_prompt**: Replace the description of each character in the prompt with [reference1], [reference2], [reference3] respectively, while keeping the scene description unchanged.
        - The output must be in English JSON format.
    """
    
    args.prompts = prompt 

    main(args)

