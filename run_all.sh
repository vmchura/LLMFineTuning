#!/bin/bash

# Run the entire LLM fine-tuning workflow

echo "Starting LLM fine-tuning workflow..."
echo "======================================"

# 1. Fine-tune the model
echo "Step 1: Fine-tuning the model..."
python finetune.py

# 2. Evaluate the models
echo "Step 2: Evaluating model performance..."
python evaluate.py

# 3. Generate report
echo "Step 3: Generating performance report..."
python generate_report.py

echo "======================================"
echo "Workflow completed!"
echo "Results can be found in the 'results' directory."
echo "Fine-tuned model is saved in the 'output' directory."
