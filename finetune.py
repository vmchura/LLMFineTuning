"""
Fine-tuning script for Llama 3.2 1B model on waste management in Olot dataset
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

# Model configuration
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # Small 1B parameter model
OUTPUT_DIR = "output"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
EPOCHS = 3

def load_data(file_path="question_answer.csv"):
    """Load and preprocess the data from CSV file"""
    df = pd.read_csv(file_path)
    
    # Prepare data in the instruction format for the model
    # Format: <s>[INST] {question} [/INST] {answer} </s>
    df["text"] = df.apply(
        lambda row: f"<s>[INST] {row['Question']} [/INST] {row['Answer']} </s>", 
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
    
    # LoRA configuration
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_steps=len(tokenized_dataset) * EPOCHS // BATCH_SIZE,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
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
    print("Starting training...")
    trainer.train()
    
    # Save the model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    print(f"Model saved to {OUTPUT_DIR}/final_model")

def main():
    """Main function to execute the fine-tuning process"""
    print("Loading dataset...")
    dataset = load_data()
    
    print("Preparing model...")
    model, tokenizer = prepare_model()
    
    print("Starting training process...")
    train_model(model, tokenizer, dataset)
    
    print("Fine-tuning completed!")

if __name__ == "__main__":
    main()
