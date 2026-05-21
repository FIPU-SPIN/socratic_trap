import csv
import json
import os
import time
from datetime import datetime

import ollama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "prompts.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "responses")
MODELS = [
"mistral:7b",
"ministral-3:14b", 
"gemma4:31b", 
"llama3.3:70b", 
"deepseek-r1:70b", 
"llama4:16x17b",
"qwen3:235b", 
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_prompts():
    prompts = {}
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            prompts[row['ID']] = row['Prompt'].replace('\\n', '\n')
    return prompts

def query_model(model, prompt):
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        return f"ERROR: {str(e)}"

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts.")

    output_file = os.path.join(OUTPUT_DIR, "results.json")

    # Load existing results so a restarted job does not lose previous work
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} results already saved.")
    else:
        results = []

    done = {(r["model"], r["id"]) for r in results}

    for model in MODELS:
        print(f"\n>>> Model: {model}")

        for concept_id, prompt in prompts.items():
            if (model, concept_id) in done:
                print(f"  {concept_id}... skipped")
                continue

            print(f"  {concept_id}...", end=" ", flush=True)
            response = query_model(model, prompt)
            results.append({
                "id": concept_id,
                "model": model,
                "response": response,
                "timestamp": datetime.now().isoformat(),
            })
            save_results(results, output_file)
            print("✓")
            time.sleep(1)

    print(f"\n Done! Results saved to {output_file}")

if __name__ == "__main__":
    main()