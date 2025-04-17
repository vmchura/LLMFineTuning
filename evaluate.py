"""
Evaluation script to compare model performance before and after fine-tuning
using GPTScore as the evaluation metric
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpt_score_scorer import GPTScore  # Using gpt-score-scorer package
from tqdm import tqdm
import json
import os

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model paths
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
FINETUNED_MODEL = "output/final_model"

def load_questions(file_path="question_answer.csv"):
    """Load questions and expected answers from CSV file"""
    df = pd.read_csv(file_path)
    return df

def get_model_response(model, tokenizer, question):
    """Get model's response to a given question"""
    prompt = f"<s>[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and clean up response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def evaluate_model(model_path, questions_df):
    """Evaluate model's performance using GPTScore"""
    print(f"Evaluating model: {model_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=DEVICE,
        torch_dtype=torch.float16
    )
    
    # Initialize GPTScore
    gpt_scorer = GPTScore()
    
    results = []
    scores = []
    
    # Process each question
    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
        question = row["Question"]
        expected_answer = row["Answer"]
        
        # Get model's response
        actual_response = get_model_response(model, tokenizer, question)
        
        # Calculate GPTScore (similarity to expected answer)
        score = gpt_scorer.score(
            references=[expected_answer],
            predictions=[actual_response]
        )
        
        # Save results
        result = {
            "question": question,
            "expected_answer": expected_answer,
            "model_response": actual_response,
            "gpt_score": score
        }
        results.append(result)
        scores.append(score)
    
    # Calculate average score
    avg_score = sum(scores) / len(scores)
    print(f"Average GPTScore: {avg_score:.4f}")
    
    return results, avg_score

def main():
    """Main function to execute the evaluation process"""
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load questions and expected answers
    questions_df = load_questions()
    
    # Evaluate base model
    print("Evaluating base model performance...")
    base_results, base_avg_score = evaluate_model(BASE_MODEL, questions_df)
    
    # Save base model results
    with open("results/base_model_results.json", "w") as f:
        json.dump({
            "results": base_results,
            "average_score": base_avg_score
        }, f, indent=2)
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model performance...")
    ft_results, ft_avg_score = evaluate_model(FINETUNED_MODEL, questions_df)
    
    # Save fine-tuned model results
    with open("results/finetuned_model_results.json", "w") as f:
        json.dump({
            "results": ft_results,
            "average_score": ft_avg_score
        }, f, indent=2)
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"Base Model Average GPTScore: {base_avg_score:.4f}")
    print(f"Fine-tuned Model Average GPTScore: {ft_avg_score:.4f}")
    print(f"Improvement: {(ft_avg_score - base_avg_score):.4f} ({((ft_avg_score - base_avg_score) / base_avg_score * 100):.2f}%)")

if __name__ == "__main__":
    main()
