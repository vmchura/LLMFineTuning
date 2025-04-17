# LLM Fine-Tuning Project for Waste Management in Olot

This project fine-tunes a small LLM (Llama 3.2 1B) on waste management data for Olot using Parameter-Efficient Fine-Tuning (PEFT) techniques.

## Project Description

- **Selected LLM**: Meta's Llama 3.2 1B Instruct model
- **Pre-processing**: Questions and answers are formatted into instruction-tuning format
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Evaluation Metric**: GPTScore to measure similarity between model responses and expected answers

## Main Hyperparameters

- **LoRA rank (r)**: 8
- **LoRA alpha**: 16
- **Learning rate**: 1e-4
- **Batch size**: 2
- **Training epochs**: 3
- **4-bit quantization**: Used to reduce GPU memory requirements

## Prerequisites

1. Python 3.8+
2. CUDA-compatible GPU (recommended, but CPU will work too)
3. Required packages (install via `pip install -r requirements.txt`):
   - torch
   - transformers
   - peft
   - bitsandbytes
   - pandas
   - datasets
   - tqdm
   - gpt-score
   - accelerate

## How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Fine-tune the Model

```bash
python finetune.py
```

This will:
- Load the question_answer.csv dataset
- Initialize Llama 3.2 1B model with 4-bit quantization
- Apply LoRA fine-tuning
- Save the fine-tuned model to the `output` directory

### 3. Evaluate Model Performance

```bash
python evaluate.py
```

This will:
- Evaluate the base model on all questions
- Evaluate the fine-tuned model on all questions
- Calculate GPTScore for both models
- Generate comparison results
- Save detailed results to the `results` directory

## Project Structure

```
├── finetune.py           # Fine-tuning script
├── evaluate.py           # Evaluation script
├── question_answer.csv   # Dataset with questions and expected answers
├── requirements.txt      # Required dependencies
├── README.md             # Project documentation
├── output/               # Fine-tuned model output directory
└── results/              # Evaluation results directory
```

## Notes

- The fine-tuning process is designed to be memory-efficient and run on consumer GPUs
- You may need to adjust batch size or quantization settings based on your hardware
- For improved results, consider increasing the number of training epochs
