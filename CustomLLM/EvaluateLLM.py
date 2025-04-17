import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm
import json
import os
import numpy as np
class EvaluateLLM:
    def __init__(self, model_path, results_evaluation_path, parameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.model_path = model_path
        self.results_evaluation_path = results_evaluation_path
        self.parameters = {'MAXIMUM_NEW_TOKENS': 256,
                          } | parameters

    def proces_question(self, question):
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    def load_questions(self):
        return pd.read_csv('question_answer.csv', header=0)

    def get_model_response(self, model, tokenizer, question):
        prompt = self.proces_question(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate response - using smaller max_new_tokens for speed
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=self.parameters['MAXIMUM_NEW_TOKENS'],  # Reduced for quick testing
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode and clean up response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        return response

    def calculate_rouge_scores(self, reference, prediction):
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

    def evaluate_model(self, model_path, questions_df):
        """Evaluate model's performance using ROUGE score"""
        print(f"Evaluating model: {model_path}")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Check if CUDA is available
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.device,
                torch_dtype=torch.float16
            )
        else:
            print("CUDA not available, loading model in standard mode for CPU.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu"
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
            actual_response = self.get_model_response(model, tokenizer, question)

            # Calculate ROUGE scores
            scores = self.calculate_rouge_scores(expected_answer, actual_response)

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

    def run(self):
        """Main function to execute the evaluation process"""
        # Create results directory
        os.makedirs(self.results_evaluation_path, exist_ok=True)

        # Load questions and expected answers
        questions_df = self.load_questions()

        # Evaluate base model
        print("Evaluating base model performance...")
        base_results, base_avg_scores = self.evaluate_model(self.model_name, questions_df)

        # Save base model results
        with open(f"{self.results_evaluation_path}/base_model_results.json", "w") as f:
            json.dump({
                "results": base_results,
                "average_scores": base_avg_scores
            }, f, indent=2)

        # Evaluate fine-tuned model
        print("Evaluating fine-tuned model performance...")
        ft_results, ft_avg_scores = self.evaluate_model(self.model_path, questions_df)

        # Save fine-tuned model results
        with open(f"{self.results_evaluation_path}/finetuned_model_results.json", "w") as f:
            json.dump({
                "results": ft_results,
                "average_scores": ft_avg_scores
            }, f, indent=2)

        # Print comparison
        print("\nPerformance Comparison:")
        print(f"Base Model Average ROUGE: {base_avg_scores['average']:.4f}")
        print(f"Fine-tuned Model Average ROUGE: {ft_avg_scores['average']:.4f}")
        print(f"Improvement: {(ft_avg_scores['average'] - base_avg_scores['average']):.4f} ({((ft_avg_scores['average'] - base_avg_scores['average']) / base_avg_scores['average'] * 100):.2f}%)")
