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
class TrainLLM(object):
    def __init__(self, output_path, parameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.output_path = output_path
        self.parameters = {
            'LORA_R': 8,
            'LORA_ALPHA': 16,
            'LORA_DROPOUT': 0.05,
            'LEARNING_RATE': 1e-4,
            'BATCH_SIZE': 2,
            'MAX_ITERATIONS': 100,
            'USE_SUBSET': False, # Test
            'SUBSET_SIZE': None, # Test
            'GRADIENT_ACCUMULATION_STEP': 4,
            'SAVE_STEPS': 50
        } | parameters

    def process_row_question_answer(self, row):
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{row['Question']}<|im_end|>\n<|im_start|>assistant\n{row['Answer']}<|im_end|>\n",

    def load_data(self):
        df = pd.read_csv('question_answer.csv', header=0)
        if self.parameters['USE_SUBSET']:
            df = df.head(self.parameters['SUBSET_SIZE'])
        df['text'] = df.apply(self.process_row_question_answer, axis=1)
        return Dataset.from_pandas(df[['text']])


    def prepare_model(self):
        """Initialize and prepare the model for training"""
        # Quantization configuration for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.parameters['MODEL_NAME'])
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Prepare for training with LoRA
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration - simplified target modules for test
        peft_config = LoraConfig(
            r=self.parameters['LORA_R'],
            lora_alpha=self.parameters['LORA_ALPHA'],
            lora_dropout=self.parameters['LORA_DROPOUT'],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Target attention modules
        )

        # Apply LoRA to model
        model = get_peft_model(model, peft_config)

        return model, tokenizer

    def preprocess_data(self, dataset, tokenizer):
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def train_model(self, model, tokenizer, dataset):
        """Train the model with the prepared dataset"""
        # Tokenize dataset
        tokenized_dataset = self.preprocess_data(dataset, tokenizer)

        # Training arguments for quick test
        training_args = TrainingArguments(
            output_dir=self.output_path,
            per_device_train_batch_size=self.parameters['BATCH_SIZE'],
            gradient_accumulation_steps=self.parameters['GRADIENT_ACCUMULATION_STEP'],  # Reduced for testing
            learning_rate=self.parameters['LEARNING_RATE'],
            max_steps=self.parameters['MAX_ITERATIONS'],  # Just do a few steps for testing
            logging_steps=1,
            save_strategy="steps",
            save_steps=self.parameters['SAVE_STEPS'],  # Save at the end
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
        print(f"Will run for {self.parameters['MAX_ITERATIONS']} iterations only")
        trainer.train()

        # Save the model
        os.makedirs(self.output_path, exist_ok=True)
        model.save_pretrained(f"{self.output_path}/final_model")
        tokenizer.save_pretrained(f"{self.output_path}/final_model")
        print(f"Model saved to {self.output_path}/final_model")

    def run(self):
        dataset = self.load_data()
        print("Preparing model...")
        model, tokenizer = self.prepare_model()

        print("Starting training process...")
        self.train_model(model, tokenizer, dataset)

        print("Fine-tuning completed!")