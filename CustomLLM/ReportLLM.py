"""
Script to generate a report comparing model performance before and after fine-tuning
using ROUGE scores
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

class ReportLLM:

    def __init__(self, results_path):
        self.results_path = results_path

    def load_results(self):
        base_path = os.path.join(self.results_path, "results","base")
        ft_path = os.path.join(self.results_path, "results","fine_tuning")
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

    def create_comparison_table(self, base_results, ft_results):
        """Create a comparison table of results"""
        if not base_results or not ft_results:
            return None

        data = []

        # Extract data for each question
        for i, (base_item, ft_item) in enumerate(zip(base_results["results"], ft_results["results"])):
            question = base_item["question"]
            base_score = base_item["rouge_scores"]["average"]
            ft_score = ft_item["rouge_scores"]["average"]
            improvement = ft_score - base_score

            data.append({
                "Question": question,
                "Base Score": base_score,
                "Fine-tuned Score": ft_score,
                "Improvement": improvement,
                "Improvement %": (improvement / base_score * 100) if base_score > 0 else 0
            })

        return pd.DataFrame(data)

    def plot_score_comparison(self, df):
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
        plt.ylabel("Average ROUGE Score")
        plt.title("Model Performance Comparison: Base vs. Fine-tuned")
        plt.legend()
        plt.tight_layout()

        # Save figure
        plt.savefig(f"{self.results_path}/score_comparison.png")
        print(f"Created visualization: {self.results_path}/score_comparison.png")

    def plot_rouge_metrics(self, base_results, ft_results):
        """Create a visualization of the different ROUGE metrics"""
        if base_results is None or ft_results is None:
            return

        metrics = ['rouge1', 'rouge2', 'rougeL', 'average']
        base_scores = [base_results["average_scores"][m] for m in metrics]
        ft_scores = [ft_results["average_scores"][m] for m in metrics]

        # Create bar chart
        plt.figure(figsize=(10, 6))
        indices = range(len(metrics))
        width = 0.35

        plt.bar(indices, base_scores, width, label="Base Model")
        plt.bar([i + width for i in indices], ft_scores, width, label="Fine-tuned Model")

        plt.xlabel("ROUGE Metric")
        plt.ylabel("Score")
        plt.title("ROUGE Metrics Comparison")
        plt.xticks([i + width/2 for i in indices], ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Average'])
        plt.legend()
        plt.tight_layout()

        # Save figure
        plt.savefig(f"{self.results_path}/rouge_metrics_comparison.png")
        print(f"Created visualization: {self.results_path}/rouge_metrics_comparison.png")

    def generate_html_report(self, df, base_results, ft_results):
        """Generate an HTML report with the comparison results"""
        if df is None or base_results is None or ft_results is None:
            return

        avg_base_score = base_results["average_scores"]["average"]
        avg_ft_score = ft_results["average_scores"]["average"]
        improvement = avg_ft_score - avg_base_score
        improvement_percent = (improvement / avg_base_score * 100) if avg_base_score > 0 else 0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert DataFrame to HTML
        table_html = df.to_html(classes="table table-striped", index=False)

        # Create detailed ROUGE metrics table
        rouge_metrics = ["rouge1", "rouge2", "rougeL", "average"]
        rouge_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "Average"]

        rouge_data = []
        for metric, name in zip(rouge_metrics, rouge_names):
            base_score = base_results["average_scores"][metric]
            ft_score = ft_results["average_scores"][metric]
            improvement = ft_score - base_score
            improvement_pct = (improvement / base_score * 100) if base_score > 0 else 0

            rouge_data.append({
                "Metric": name,
                "Base Model": f"{base_score:.4f}",
                "Fine-tuned Model": f"{ft_score:.4f}",
                "Improvement": f"{improvement:.4f} ({improvement_pct:.2f}%)"
            })

        rouge_df = pd.DataFrame(rouge_data)
        rouge_table_html = rouge_df.to_html(classes="table table-striped", index=False)

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
                    <p>Base Model Average ROUGE Score: <span class="highlight">{avg_base_score:.4f}</span></p>
                    <p>Fine-tuned Model Average ROUGE Score: <span class="highlight">{avg_ft_score:.4f}</span></p>
                    <p>Improvement: <span class="highlight">{improvement:.4f}</span> ({improvement_percent:.2f}%)</p>
                </div>
                
                <h2>ROUGE Metrics Comparison</h2>
                {rouge_table_html}
                <img src="rouge_metrics_comparison.png" class="img-fluid" alt="ROUGE Metrics Comparison">
                
                <h2>Per-Question Performance</h2>
                <img src="score_comparison.png" class="img-fluid" alt="Score Comparison">
                
                <h2>Detailed Results</h2>
                {table_html}
                
                <h2>Model Information</h2>
                <p>Base Model: meta-llama/Llama-3.2-1B-Instruct</p>
                <p>Fine-tuning Method: LoRA (Low-Rank Adaptation)</p>
                <p>Evaluation Metric: ROUGE (Recall-Oriented Understudy for Gisting Evaluation)</p>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        with open(f"{self.results_path}/fine_tuning_report.html", "w") as f:
            f.write(html_content)

        print(f"Generated HTML report: {self.results_path}/fine_tuning_report.html")

    def run(self):
        """Main function to generate the report"""
        print("Generating fine-tuning report...")

        # Load results
        base_results, ft_results = self.load_results()
        if not base_results or not ft_results:
            return

        # Create comparison table
        df = self.create_comparison_table(base_results, ft_results)

        # Plot comparison
        self.plot_score_comparison(df)

        # Plot ROUGE metrics comparison
        self.plot_rouge_metrics(base_results, ft_results)

        # Generate HTML report
        self.generate_html_report(df, base_results, ft_results)

        print("Report generation completed!")

