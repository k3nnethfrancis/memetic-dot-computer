"""
Fine-tuning Script for Large Language Models

This script fine-tunes a pre-trained language model using the PEFT library
for efficient fine-tuning on a Mac Studio with M2 Max and 96GB of unified memory.

Key Features:
- Uses PEFT with LoRA for parameter-efficient fine-tuning
- Optimized for Apple Silicon (M2 Max) using MPS
- Implements gradient checkpointing for memory efficiency
- Handles variable-length inputs efficiently

Usage:
    python3 -m learning.finetune

Environment Variables:
    MEMETIC_BASE_PATH: Base path for the project (optional)
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Configuration
BASE_PATH = os.environ.get('MEMETIC_BASE_PATH', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"
MODEL_PATH = os.path.join(BASE_PATH, "weights", "meta-llama", MODEL_NAME)
TRAIN_FILE = os.path.join(BASE_PATH, "learning", "payloads", "train.jsonl")
VAL_FILE = os.path.join(BASE_PATH, "learning", "payloads", "val.jsonl")
OUTPUT_DIR = os.path.join(BASE_PATH, "weights", "finetunes", f"{MODEL_NAME}_finetune")

# Training parameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.03

def setup_model_and_tokenizer():
    """Set up the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.gradient_checkpointing_enable()
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    """Prepare and tokenize the dataset."""
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    
    def tokenize_function(examples):
        texts = [f"{prompt} {completion}" for prompt, completion in zip(examples["prompt"], examples["completion"])]
        return tokenizer(texts, truncation=True, padding=True)

    return dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

def main():
    """Main function to run the fine-tuning process."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = setup_model_and_tokenizer()
    tokenized_dataset = prepare_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=10,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()