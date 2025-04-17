"""
Script to generate a report comparing model performance before and after fine-tuning
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def load_results(base_path="results/base_model_results.json", ft_path="results/finetuned_model_results.json"):
    """Load evaluation results from JSON files"""
    try:
        with open(base_path, 'r') as f:
            base_results = json.load(f)
        
        with open(ft_path, 'r') as f:
            ft_results = json.load(f)
        
        return base_results, ft_results
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run evaluate.py first to generate results.")
        return None, None

def create_comparison_table(base_results, ft_results):
    """Create a comparison table of results"""
    if not base_results or not ft_results:
        return None
    
    data = []
    
    # Extract data for each question
    for i, (base_item, ft_item) in enumerate(zip(base_results["results"], ft_results["results"])):
        question = base_item["question"]
        base_score = base_item["gpt_score"]
        ft_score = ft_item["gpt_score"]
        improvement = ft_score - base_score
        
        data.append({
            "Question": question,
            "Base Score": base_score,
            "Fine-tuned Score": ft_score,
            "Improvement": improvement,
            "Improvement %": (improvement / base_score * 100) if base_score > 0 else 0
        })
    
    return pd.DataFrame(data)

def plot_score_comparison(df):
    """Create a visualization of the score comparison"""
    if df is None:
        return
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    indices = range(len(df))
    width = 0.35
    
    plt.bar(indices, df["Base Score"], width, label="Base Model")
    plt.bar([i + width for i in indices], df["Fine-tuned Score"], width, label="Fine-tuned Model")
    
    plt.xlabel("Question Index")
    plt.ylabel("GPTScore")
    plt.title("Model Performance Comparison: Base vs. Fine-tuned")
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig("results/score_comparison.png")
    print("Created visualization: results/score_comparison.png")

def generate_html_report(df, base_results, ft_results):
    """Generate an HTML report with the comparison results"""
    if df is None or base_results is None or ft_results is None:
        return
    
    avg_base_score = base_results["average_score"]
    avg_ft_score = ft_results["average_score"]
    improvement = avg_ft_score - avg_base_score
    improvement_percent = (improvement / avg_base_score * 100) if avg_base_score > 0 else 0
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert DataFrame to HTML
    table_html = df.to_html(classes="table table-striped", index=False)
    
    # HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Fine-tuning Results</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            .summary-box {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .highlight {{ font-weight: bold; color: #0d6efd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LLM Fine-tuning Results Report</h1>
            <p class="text-muted">Generated on {timestamp}</p>
            
            <div class="summary-box">
                <h2>Performance Summary</h2>
                <p>Base Model Average GPTScore: <span class="highlight">{avg_base_score:.4f}</span></p>
                <p>Fine-tuned Model Average GPTScore: <span class="highlight">{avg_ft_score:.4f}</span></p>
                <p>Improvement: <span class="highlight">{improvement:.4f}</span> ({improvement_percent:.2f}%)</p>
            </div>
            
            <h2>Visualization</h2>
            <img src="score_comparison.png" class="img-fluid" alt="Score Comparison">
            
            <h2>Detailed Results</h2>
            {table_html}
            
            <h2>Model Information</h2>
            <p>Base Model: meta-llama/Llama-3.2-1B-Instruct</p>
            <p>Fine-tuning Method: LoRA (Low-Rank Adaptation)</p>
            <p>Evaluation Metric: GPTScore</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open("results/fine_tuning_report.html", "w") as f:
        f.write(html_content)
    
    print("Generated HTML report: results/fine_tuning_report.html")

def main():
    """Main function to generate the report"""
    print("Generating fine-tuning report...")
    
    # Load results
    base_results, ft_results = load_results()
    if not base_results or not ft_results:
        return
    
    # Create comparison table
    df = create_comparison_table(base_results, ft_results)
    
    # Plot comparison
    plot_score_comparison(df)
    
    # Generate HTML report
    generate_html_report(df, base_results, ft_results)
    
    print("Report generation completed!")

if __name__ == "__main__":
    main()
