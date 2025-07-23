# src/results.py
"""
Results saving module for SCOPE framework.
Handles saving evaluation results in various formats.
"""

import os
import json
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def save_results(
    results: Dict,
    output_dir: str,
    timestamp: str
) -> None:
    """
    Save all evaluation results.
    
    Args:
        results: Complete results dictionary
        output_dir: Directory to save results
        timestamp: Timestamp for filenames
    """
    model_name = results['model'].replace('/', '_')
    
    # Save complete results as JSON
    save_json_results(results, output_dir, model_name, timestamp)
    
    # Save summary metrics as CSV
    save_summary_csv(results, output_dir, model_name, timestamp)
    
    # Save detailed evaluation results
    save_detailed_results(results, output_dir, model_name, timestamp)
    
    # Save position biases
    save_position_biases(results, output_dir, model_name, timestamp)
    
    # Create paper-style tables
    create_paper_tables(results, output_dir, model_name, timestamp)
    
    logger.info(f"All results saved to {output_dir}")


def save_json_results(
    results: Dict,
    output_dir: str,
    model_name: str,
    timestamp: str
) -> None:
    """
    Save complete results as JSON.
    
    Args:
        results: Complete results dictionary
        output_dir: Output directory
        model_name: Model name for filename
        timestamp: Timestamp
    """
    filename = f'complete_results_{model_name}_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types to Python types for JSON serialization
    results_serializable = convert_to_serializable(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved complete results: {filename}")


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy types to Python types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Serializable object
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_summary_csv(
    results: Dict,
    output_dir: str,
    model_name: str,
    timestamp: str
) -> None:
    """
    Save summary metrics as CSV.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        model_name: Model name
        timestamp: Timestamp
    """
    summary_data = []
    
    for dataset_name, dataset_results in results['datasets'].items():
        metrics = dataset_results['metrics']['summary']
        
        row = {
            'Model': model_name,
            'Dataset': dataset_name,
            'Total_Questions': metrics['total_questions'],
            'Pr-T': metrics['Pr-T'],
            'Pr-F': metrics['Pr-F'],
            'Co-T': metrics['Co-T'],
            'Co-F': metrics['Co-F'],
            'Answer_F1': round(metrics['Answer_F1'], 4),
            'Distractor_F1': round(metrics['Distractor_F1'], 4),
            'F1_Gap': round(metrics['Answer_F1'] - metrics['Distractor_F1'], 4),
            'Lucky_hit': round(metrics['Lucky_hit'], 4),
            'Pure_skill': round(metrics['Pure_skill'], 4)
        }
        
        summary_data.append(row)
    
    # Save as CSV
    filename = f'summary_metrics_{model_name}_{timestamp}.csv'
    filepath = os.path.join(output_dir, filename)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(filepath, index=False)
    
    logger.info(f"Saved summary metrics: {filename}")


def save_detailed_results(
    results: Dict,
    output_dir: str,
    model_name: str,
    timestamp: str
) -> None:
    """
    Save detailed evaluation results for each question.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        model_name: Model name
        timestamp: Timestamp
    """
    for dataset_name, dataset_results in results['datasets'].items():
        detailed_data = []
        
        for eval_result in dataset_results['evaluation_results']:
            row = {
                'question_id': eval_result['question_id'],
                'correct_answer': eval_result['correct_answer'],
                'responses': ','.join(eval_result['responses']),
                'original_label_responses': ','.join(eval_result['original_label_responses']),
                'correct_count': eval_result['correct_count'],
                'total_valid_responses': eval_result['total_valid_responses'],
                'most_common_response': eval_result['most_common_response'],
                'most_common_count': eval_result['most_common_count']
            }
            
            detailed_data.append(row)
        
        # Save as CSV
        filename = f'detailed_results_{model_name}_{dataset_name}_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        
        df = pd.DataFrame(detailed_data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved detailed results for {dataset_name}: {filename}")


def save_position_biases(
    results: Dict,
    output_dir: str,
    model_name: str,
    timestamp: str
) -> None:
    """
    Save position bias distributions.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        model_name: Model name
        timestamp: Timestamp
    """
    bias_data = {
        'model': model_name,
        'timestamp': timestamp,
        'datasets': {}
    }
    
    for dataset_name, dataset_results in results['datasets'].items():
        bias_data['datasets'][dataset_name] = {
            'position_bias': dataset_results['position_bias'],
            'inverse_bias_dist': dataset_results['inverse_bias_dist'],
            'lucky_hit_probability': dataset_results['metrics']['lucky_hit_probability']
        }
    
    filename = f'position_biases_{model_name}_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(bias_data, f, indent=2)
    
    logger.info(f"Saved position biases: {filename}")


def create_paper_tables(
    results: Dict,
    output_dir: str,
    model_name: str,
    timestamp: str
) -> None:
    """
    Create tables in paper format (like Tables 6, 7 in the paper).
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        model_name: Model name
        timestamp: Timestamp
    """
    # Table format similar to paper's Table 6 & 7
    for dataset_name, dataset_results in results['datasets'].items():
        metrics = dataset_results['metrics']
        pr_co = metrics['pr_co_metrics']
        
        # Create Pr/Co table
        table_data = [
            ['Metric', '', 'T', 'F'],
            [model_name, '', '', ''],
            ['', 'Pr', f"{pr_co['Pr-T']} ({pr_co['Pr-T']/pr_co['total']:.1%})", 
             f"{pr_co['Pr-F']} ({pr_co['Pr-F']/pr_co['total']:.1%})"],
            ['', 'Co', f"{pr_co['Co-T']} ({pr_co['Co-T']/pr_co['total']:.1%})", 
             f"{pr_co['Co-F']} ({pr_co['Co-F']/pr_co['total']:.1%})"]
        ]
        
        # Save as CSV
        filename = f'paper_table_{model_name}_{dataset_name}_pr_co_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table_data)
        
        # Create Answer/Distractor metrics table
        metrics_table = [
            ['Method', 'Answer Precision', 'Answer Recall', 'Answer F1', 
             'Distractor Precision', 'Distractor Recall', 'Distractor F1'],
            [f"SCOPE ({model_name})", 
             f"{metrics['answer_metrics']['Precision']:.4f}",
             f"{metrics['answer_metrics']['Recall']:.4f}",
             f"{metrics['answer_metrics']['F1']:.4f}",
             f"{metrics['distractor_metrics']['Precision']:.4f}",
             f"{metrics['distractor_metrics']['Recall']:.4f}",
             f"{metrics['distractor_metrics']['F1']:.4f}"]
        ]
        
        filename = f'paper_table_{model_name}_{dataset_name}_f1_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(metrics_table)
        
        logger.info(f"Created paper-style tables for {dataset_name}")


def save_ablation_results(
    ablation_results: Dict[str, Dict],
    output_dir: str,
    timestamp: str
) -> None:
    """
    Save ablation study results.
    
    Args:
        ablation_results: Results from ablation study
        output_dir: Output directory
        timestamp: Timestamp
    """
    # Create summary table
    summary_data = []
    
    for mode, results in ablation_results.items():
        for dataset_name, dataset_results in results['datasets'].items():
            metrics = dataset_results['metrics']
            
            row = {
                'Mode': mode,
                'Dataset': dataset_name,
                'Answer_F1': round(metrics['answer_metrics']['F1'], 4),
                'Distractor_F1': round(metrics['distractor_metrics']['F1'], 4),
                'Lucky_hit': round(metrics['lucky_hit_probability'], 4),
                'Pure_skill': round(metrics['pure_skill'], 4)
            }
            
            summary_data.append(row)
    
    # Save as CSV
    filename = f'ablation_summary_{timestamp}.csv'
    filepath = os.path.join(output_dir, filename)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(filepath, index=False)
    
    # Also save complete ablation results as JSON
    filename_json = f'ablation_complete_{timestamp}.json'
    filepath_json = os.path.join(output_dir, filename_json)
    
    ablation_serializable = convert_to_serializable(ablation_results)
    
    with open(filepath_json, 'w', encoding='utf-8') as f:
        json.dump(ablation_serializable, f, indent=2)
    
    logger.info(f"Saved ablation results: {filename} and {filename_json}")


def create_latex_table(
    results: Dict,
    dataset_name: str,
    output_dir: str,
    timestamp: str
) -> None:
    """
    Create LaTeX formatted table for paper.
    
    Args:
        results: Results dictionary
        dataset_name: Dataset name
        output_dir: Output directory
        timestamp: Timestamp
    """
    if dataset_name not in results['datasets']:
        return
    
    metrics = results['datasets'][dataset_name]['metrics']
    model_name = results['model'].replace('_', '\_')  # Escape underscores for LaTeX
    
    latex_content = f"""\\begin{{table}}[h]
\\centering
\\caption{{{model_name} on {dataset_name}: SCOPE Results}}
\\begin{{tabular}}{{l|cc}}
\\hline
Model & Metric & T & F \\\\
\\hline
\\multirow{{2}}{{*}}{{{model_name}}} & Pr & {metrics['pr_co_metrics']['Pr-T']} & {metrics['pr_co_metrics']['Pr-F']} \\\\
 & Co & {metrics['pr_co_metrics']['Co-T']} & {metrics['pr_co_metrics']['Co-F']} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    filename = f'latex_table_{model_name}_{dataset_name}_{timestamp}.tex'
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    logger.info(f"Created LaTeX table: {filename}")


# Example usage
if __name__ == "__main__":
    # Test with dummy results
    test_results = {
        'model': 'gpt-3.5-turbo',
        'timestamp': '20250101_120000',
        'config': {'test': True},
        'datasets': {
            'CSQA': {
                'num_questions': 10,
                'num_choices': 5,
                'position_bias': {'A': 0.2, 'B': 0.2, 'C': 0.2, 'D': 0.2, 'E': 0.2},
                'inverse_bias_dist': {'A': 0.2, 'B': 0.2, 'C': 0.2, 'D': 0.2, 'E': 0.2},
                'metrics': {
                    'pr_co_metrics': {
                        'Pr-T': 8, 'Pr-F': 2, 'Co-T': 6, 'Co-F': 1, 'total': 10
                    },
                    'answer_metrics': {'Precision': 0.857, 'Recall': 0.750, 'F1': 0.800},
                    'distractor_metrics': {'Precision': 0.143, 'Recall': 0.500, 'F1': 0.222},
                    'f1_gap': 0.578,
                    'lucky_hit_probability': 0.200,
                    'pure_skill': 0.600,
                    'summary': {
                        'total_questions': 10,
                        'Pr-T': 8, 'Pr-F': 2, 'Co-T': 6, 'Co-F': 1,
                        'Answer_F1': 0.800, 'Distractor_F1': 0.222,
                        'Lucky_hit': 0.200, 'Pure_skill': 0.600
                    }
                },
                'evaluation_results': []
            }
        }
    }
    
    save_results(test_results, '.', '20250101_120000')