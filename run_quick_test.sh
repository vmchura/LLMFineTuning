#!/bin/bash

# Quick test script to verify the entire LLM fine-tuning workflow

echo "Starting QUICK TEST LLM fine-tuning workflow..."
echo "This will only run a minimal number of iterations to test that everything works"
echo "======================================"

# 1. Fine-tune the model with minimal iterations
echo "Step 1: Quick testing fine-tuning..."
python finetune_test.py

# 2. Evaluate the models with minimal examples
echo "Step 2: Quick testing evaluation..."
python evaluate_test.py

# 3. Generate report
echo "Step 3: Generating performance report..."
python generate_report.py

echo "======================================"
echo "Quick test workflow completed!"
echo "If this worked without errors, you can now run the full training with run_all.sh"
echo "Results can be found in the 'results' directory."
echo "Fine-tuned model is saved in the 'output' directory."
