#!/usr/bin/env python3
"""
Complete ML fine-tuning and evaluation flow.
This script orchestrates the entire process of training, evaluating, and reporting on an LLM model.

Usage:
  For full training:  python complete_flow.py [steps]
  For testing only:   python complete_flow.py -test [steps]

Steps can be 'train', 'evaluate', 'report' or any combination.
If no steps are specified, all three steps will be executed in sequence.
"""

import sys
import os
from datetime import datetime

from CustomLLM.TrainLLM import TrainLLM
from CustomLLM.EvaluateLLM import EvaluateLLM
from CustomLLM.ReportLLM import ReportLLM

def print_header(message):
    """Print a formatted header for steps in the process"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"  {message}")
    print(f"{line}\n")

def run_train(test_mode=False):
    """Run the training step with appropriate parameters"""
    print_header("STARTING TRAINING PROCESS")
    
    # Set up paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"models/finetuned_{timestamp}"
    
    # Set default parameters
    parameters = {}
    
    if test_mode:
        # Use testing parameters for quick runs
        parameters = {
            'MAX_ITERATIONS': 2,
            'USE_SUBSET': True,
            'SUBSET_SIZE': 5,
            'GRADIENT_ACCUMULATION_STEP': 1,
            'SAVE_STEPS': 2
        }
        print("Running in TEST mode with reduced parameters.")
    
    # Initialize and run training
    trainer = TrainLLM(output_path, parameters)
    trainer.run()
    
    # Return the path where the model was saved
    return f"{output_path}/final_model"

def run_evaluate(model_path, test_mode=False):
    """Run the evaluation step with appropriate parameters"""
    print_header("STARTING EVALUATION PROCESS")
    
    # Set up paths for evaluation results
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    
    # Set default parameters
    parameters = {}
    
    if test_mode:
        # Use testing parameters for quick evaluation
        parameters = {
            'MAXIMUM_NEW_TOKENS': 100
        }
        print("Running in TEST mode with reduced parameters.")
    
    # Initialize and run evaluation
    evaluator = EvaluateLLM(model_path, results_path, parameters)
    evaluator.run()
    
    return results_path

def run_report(results_path):
    """Generate the performance report"""
    print_header("GENERATING PERFORMANCE REPORT")
    reporter = ReportLLM(results_path)
    reporter.run()

def main():
    """Main function to orchestrate the complete flow"""
    # Parse command-line arguments
    args = sys.argv[1:]
    
    # Check if test mode is enabled
    test_mode = False
    if "-test" in args:
        test_mode = True
        args.remove("-test")
    
    # Determine which steps to run
    steps = args if args else ["train", "evaluate", "report"]
    
    # Display mode and steps
    mode_str = "TEST" if test_mode else "FULL"
    print(f"Running in {mode_str} mode with steps: {', '.join(steps)}")
    
    model_path = None
    results_path = None
    
    # Execute the requested steps
    if "train" in steps:
        model_path = run_train(test_mode)
        print(f"Training completed. Model saved to: {model_path}")
    else:
        # If not training, use the latest trained model
        # This assumes models are saved in a 'models' directory with timestamp-based names
        try:
            model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d)) and d.startswith("finetuned_")]
            latest_model = sorted(model_dirs)[-1]
            model_path = f"models/{latest_model}/final_model"
            print(f"Using latest trained model: {model_path}")
        except (FileNotFoundError, IndexError):
            print("Error: No trained model found. Please run the 'train' step first.")
            if "evaluate" in steps:
                print("Skipping evaluation step due to missing model.")
                steps.remove("evaluate")
                
    if "evaluate" in steps and model_path:
        results_path = run_evaluate(model_path, test_mode)
        print(f"Evaluation completed. Results saved to: {results_path}")
            
    if "report" in steps and results_path:
        run_report(results_path)
    
    print("\nComplete flow execution finished!")

if __name__ == "__main__":
    main()
