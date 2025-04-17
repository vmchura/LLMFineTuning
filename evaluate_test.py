"""
Quick test evaluation script to compare model performance before and after fine-tuning
using ROUGE score as the evaluation metric - only evaluates on a few examples
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm
import json
import os
import numpy as np

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model paths - using TinyLlama for quick testing
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINETUNED_MODEL = "output/final_model"

# Quick test settings
USE_SUBSET = True
SUBSET_SIZE = 3  # Only evaluate on 3 questions

def load_questions(file_path="question_answer.csv", use_subset=USE_SUBSET, subset_size=SUBSET_SIZE):
    """Load questions and expected answers from CSV file"""
    df = pd.read_csv(file_path)
    
    # Use subset if specified for quick testing
    if use_subset:
        print(f"Using subset of {subset_size} questions for quick testing")
        df = df.head(subset_size)
        
    return df

def get_model_response(model, tokenizer, question):
    """Get model's response to a given question"""
    # Using TinyLlama format
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate response - using smaller max_new_tokens for speed
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,  # Reduced for quick testing
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and clean up response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def calculate_rouge_scores(reference, prediction):
    """Calculate ROUGE scores between reference and prediction"""
    # Initialize ROUGE scorer
    # Using ROUGE-1, ROUGE-2, and ROUGE-L metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores
    scores = scorer.score(reference, prediction)
    
    # Return F1 scores as a dictionary
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
        'average': (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    }

def evaluate_model(model_path, questions_df):
    """Evaluate model's performance using ROUGE score"""
    print(f"Quick testing evaluation on model: {model_path}")
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=DEVICE,
        torch_dtype=torch.float16
    )
    
    results = []
    all_scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'average': []
    }
    
    # Process each question
    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
        question = row["Question"]
        expected_answer = row["Answer"]
        
        # Get model's response
        print(f"Getting response for: {question[:30]}...")
        actual_response = get_model_response(model, tokenizer, question)
        print(f"Response: {actual_response[:50]}...")
        
        # Calculate ROUGE scores
        scores = calculate_rouge_scores(expected_answer, actual_response)
        print(f"ROUGE scores: {scores}")
        
        # Save results
        result = {
            "question": question,
            "expected_answer": expected_answer,
            "model_response": actual_response,
            "rouge_scores": scores
        }
        results.append(result)
        
        # Collect all scores for averaging
        for metric in all_scores.keys():
            all_scores[metric].append(scores[metric])
    
    # Calculate average scores
    avg_scores = {metric: np.mean(scores) for metric, scores in all_scores.items()}
    print(f"Average ROUGE scores:")
    print(f"  ROUGE-1: {avg_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {avg_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {avg_scores['rougeL']:.4f}")
    print(f"  Average: {avg_scores['average']:.4f}")
    
    return results, avg_scores

def main():
    """Main function to execute the evaluation process"""
    print("RUNNING QUICK TEST EVALUATION")
    print("This will evaluate only a few examples to test the workflow")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load questions and expected answers
    questions_df = load_questions()
    
    # Evaluate base model
    print("Quick testing base model performance...")
    base_results, base_avg_scores = evaluate_model(BASE_MODEL, questions_df)
    
    # Save base model results
    with open("results/base_model_results.json", "w") as f:
        json.dump({
            "results": base_results,
            "average_scores": base_avg_scores
        }, f, indent=2)
    
    # Evaluate fine-tuned model
    print("Quick testing fine-tuned model performance...")
    ft_results, ft_avg_scores = evaluate_model(FINETUNED_MODEL, questions_df)
    
    # Save fine-tuned model results
    with open("results/finetuned_model_results.json", "w") as f:
        json.dump({
            "results": ft_results,
            "average_scores": ft_avg_scores
        }, f, indent=2)
    
    # Print comparison
    print("\nQuick Test Performance Comparison:")
    print(f"Base Model Average ROUGE: {base_avg_scores['average']:.4f}")
    print(f"Fine-tuned Model Average ROUGE: {ft_avg_scores['average']:.4f}")
    print(f"Improvement: {(ft_avg_scores['average'] - base_avg_scores['average']):.4f} ({((ft_avg_scores['average'] - base_avg_scores['average']) / base_avg_scores['average'] * 100):.2f}%)")
    
    print("Quick test evaluation completed!")

if __name__ == "__main__":
    main()
