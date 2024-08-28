import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
base_model_path = "/Users/kenneth/Desktop/lab/memetic.computer/weights/meta-llama/Meta-Llama-3.1-8B-Instruct"
lora_path = "/Users/kenneth/Desktop/lab/memetic.computer/weights/finetunes/Meta-Llama-3.1-8B-Instruct_finetune"
output_path = "/Users/kenneth/Desktop/lab/memetic.computer/weights/merged/Meta-Llama-3.1-8B-Instruct_merged"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Load the base model and the fine-tuned LoRA model
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
lora_model = PeftModel.from_pretrained(base_model, lora_path)

# Merge LoRA weights into the base model
merged_model = lora_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(output_path)

# Copy tokenizer files from the base model directory to the output directory
try:
    shutil.copy(os.path.join(base_model_path, 'tokenizer.json'), os.path.join(output_path, 'tokenizer.json'))
    shutil.copy(os.path.join(base_model_path, 'tokenizer_config.json'), os.path.join(output_path, 'tokenizer_config.json'))
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure that tokenizer files exist in the base model directory.")