"""
Fine-tuning script for a small LLM on waste management in Olot dataset
This is a quick test version to verify the workflow works
"""

import os
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model configuration - using a small, open model that doesn't require authentication
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Alternative: "microsoft/phi-2"
OUTPUT_DIR = "output"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
BATCH_SIZE = 2

# QUICK TEST CONFIGURATION
MAX_ITERATIONS = 2  # Just do 2 iterations to test workflow
USE_SUBSET = True   # Use only a subset of data for quick testing
SUBSET_SIZE = 5     # Number of examples to use in the subset

def load_data(file_path="question_answer.csv", use_subset=USE_SUBSET, subset_size=SUBSET_SIZE):
    """Load and preprocess the data from CSV file"""
    df = pd.read_csv(file_path)
    
    # Use subset if specified for quick testing
    if use_subset:
        print(f"Using subset of {subset_size} examples for quick testing")
        df = df.head(subset_size)
    
    # Prepare data in the instruction format for the model
    # Format depends on the model, TinyLlama format:
    df["text"] = df.apply(
        lambda row: f"<|user|>\n{row['Question']}\n<|assistant|>\n{row['Answer']}\n", 
        axis=1
    )
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df[["text"]])
    return dataset

def prepare_model():
    """Initialize and prepare the model for training"""
    # Quantization configuration for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Prepare for training with LoRA
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration - simplified target modules for test
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Target attention modules
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def preprocess_data(dataset, tokenizer):
    """Tokenize and prepare dataset for training"""
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def train_model(model, tokenizer, dataset):
    """Train the model with the prepared dataset"""
    # Tokenize dataset
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    
    # Training arguments for quick test
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,  # Reduced for testing
        learning_rate=LEARNING_RATE,
        max_steps=MAX_ITERATIONS,  # Just do a few steps for testing
        logging_steps=1,
        save_strategy="steps",
        save_steps=MAX_ITERATIONS,  # Save at the end
        report_to="none",  # Disable Wandb reporting
        remove_unused_columns=False,
        fp16=True,  # Mixed precision training
    )
    
    # Initialize trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting QUICK TEST training...")
    print(f"Will run for {MAX_ITERATIONS} iterations only")
    trainer.train()
    
    # Save the model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    print(f"Model saved to {OUTPUT_DIR}/final_model")

def main():
    """Main function to execute the fine-tuning process"""
    print("RUNNING IN QUICK TEST MODE")
    print("This will only run a few iterations to test the workflow")
    
    print("Loading dataset...")
    dataset = load_data()
    
    print("Preparing model...")
    model, tokenizer = prepare_model()
    
    print("Starting training process...")
    train_model(model, tokenizer, dataset)
    
    print("Quick test fine-tuning completed!")

if __name__ == "__main__":
    main()
