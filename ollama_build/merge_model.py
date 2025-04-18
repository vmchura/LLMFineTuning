import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model
base_model_name = "Qwen/Qwen2-0.5B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Path to your adapter model (choose the most recent or best performing one)
adapter_path = "models/finetuned_20250418_035128/final_model"

# Load the adapter model
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge the LoRA weights with the base model
model = model.merge_and_unload()

# Save the merged model
output_dir = "ollama_build"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Merged model saved to {output_dir}")