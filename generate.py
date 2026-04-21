import csv
import json
import os
import time
from datetime import datetime
import ollama

CSV_FILE = "prompts.csv"
OUTPUT_DIR = "responses"
MODELS = ["llama3.2:3b", "mistral", "deepseek-r1:8b", "qwen2.5:7b", "phi3:mini"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_prompts():
    prompts = {}
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            prompts[row['ID']] = row['Prompt']
    return prompts

def query_model(model, prompt):
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts.")
    
    for model in MODELS:
        model_dir = os.path.join(OUTPUT_DIR, model.replace(":", "_"))
        os.makedirs(model_dir, exist_ok=True)
        print(f"\n>>> Model: {model}")
        
        for concept_id, prompt in prompts.items():
            print(f"  {concept_id}...", end=" ", flush=True)
            response = query_model(model, prompt)
            
            with open(os.path.join(model_dir, f"{concept_id}.json"), 'w', encoding='utf-8') as f:
                json.dump({"id": concept_id, "model": model, "response": response, "timestamp": datetime.now().isoformat()}, f, indent=2)
            print("✓")
            time.sleep(1)

    print("\n Done!")

if __name__ == "__main__":
    main()