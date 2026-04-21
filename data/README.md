# SocraticTrap-CS: LLM Response Generation

Benchmarking strategic misconceptions of large language models in computer science education.

## Project Structure

- `prompts.csv` - 35 structured prompts (one per concept)
- `generate.py` - Python script to query local LLMs via Ollama
- `data/ground_truth.csv` - Ground truth definitions for annotation
- `data/input_questions.csv` - Input questions for each concept
- `responses/` - Generated responses (not tracked in git)

## Models Used

- Llama 3.2 (3B)
- Mistral (7B)
- DeepSeek-R1 (8B)
- Qwen 2.5 (7B)
- Phi-3 Mini (3.8B)

## Setup

```bash
# Install Ollama from https://ollama.com

# Pull models
ollama pull llama3.2:3b
ollama pull mistral
ollama pull deepseek-r1:8b
ollama pull qwen2.5:7b
ollama pull phi3:mini

# Install Python dependencies
pip install -r requirements.txt