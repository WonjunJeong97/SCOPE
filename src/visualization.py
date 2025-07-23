# src/visualization.py
"""
Visualization module for SCOPE framework.
Creates plots and tables for evaluation results.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

# Disable LaTeX completely
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.preamble'] = ''

# Reset to default matplotlib style
plt.rcParams.update(plt.rcParamsDefault)

# Set Times New Roman font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

logger = logging.getLogger(__name__)

# Define color palette
COLORS = {
    'primary': ['#5B9BD5', '#70AD47', '#FFC000', '#ED7D31', '#A5A5A5'],  # blue, green, orange, dark orange, gray
    'position_bias': '#5B9BD5',  # blue
    'inverse_bias': '#70AD47',   # green
    'answer': '#70AD47',         # green
    'distractor': '#ED7D31'      # dark orange
}


def plot_position_bias(
    position_bias: Dict[str, float],
    dataset_name: str,
    model_name: str,
    output_dir: str,
    timestamp: str
) -> None:
    """
    Plot position bias distribution.
    
    Args:
        position_bias: Dictionary mapping positions to probabilities
        dataset_name: Name of the dataset
        model_name: Name of the model
        output_dir: Output directory
        timestamp: Timestamp for filename
    """
    # Set figure size
    fig_width = 3.0
    fig_height = 2.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Prepare data
    positions = sorted(position_bias.keys())
    values = [position_bias[pos] * 100 for pos in positions]  # Convert to percentage
    
    # Bar positions and width
    x = np.arange(len(positions))
    width = 0.6
    
    # Create bar plot
    bars = ax.bar(x, values, width,
                  color=COLORS['position_bias'],
                  edgecolor='black',
                  linewidth=0.5)
    
    # Add value labels on bars
    for i, (pos, val) in enumerate(zip(positions, values)):
        ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Set axes
    ax.set_xlabel('Position', fontsize=9)
    ax.set_ylabel('Selection Rate (%)', fontsize=9)
    ax.set_title(f'{model_name} - {dataset_name} Position Bias', fontsize=10, pad=10)
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    
    # Auto-adjust y-axis range
    max_val = max(values)
    ax.set_ylim(0, max_val * 1.2)
    
    # Grid settings
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Spine settings
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = f'position_bias_{model_name}_{dataset_name}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    logger.info(f"Saved position bias plot: {filename}")


def plot_bias_comparison(
    position_bias: Dict[str, float],
    inverse_bias: Dict[str, float],
    dataset_name: str,
    model_name: str,
    output_dir: str,
    timestamp: str
) -> None:
    """
    Plot comparison between position bias and inverse bias.
    
    Args:
        position_bias: Original position bias
        inverse_bias: Inverse bias distribution
        dataset_name: Name of the dataset
        model_name: Name of the model
        output_dir: Output directory
        timestamp: Timestamp for filename
    """
    fig_width = 4.5
    fig_height = 3.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    positions = sorted(position_bias.keys())
    
    # Position bias (left subplot)
    values1 = [position_bias[pos] * 100 for pos in positions]
    x = np.arange(len(positions))
    width = 0.6
    
    bars1 = ax1.bar(x, values1, width,
                    color=COLORS['position_bias'],
                    edgecolor='black',
                    linewidth=0.5)
    
    ax1.set_xlabel('Position', fontsize=9)
    ax1.set_ylabel('Probability (%)', fontsize=9)
    ax1.set_title('Position Bias', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.set_ylim(0, max(values1) * 1.2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Inverse bias (right subplot)
    values2 = [inverse_bias[pos] * 100 for pos in positions]
    
    bars2 = ax2.bar(x, values2, width,
                    color=COLORS['inverse_bias'],
                    edgecolor='black',
                    linewidth=0.5)
    
    ax2.set_xlabel('Position', fontsize=9)
    ax2.set_ylabel('Probability (%)', fontsize=9)
    ax2.set_title('Inverse Bias (for answer placement)', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions)
    ax2.set_ylim(0, max(values2) * 1.2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels
    for i, val in enumerate(values1):
        ax1.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
    for i, val in enumerate(values2):
        ax2.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
    
    plt.suptitle(f'{model_name} - {dataset_name}', fontsize=11)
    plt.tight_layout()
    
    # Save figure
    filename = f'bias_comparison_{model_name}_{dataset_name}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    logger.info(f"Saved bias comparison plot: {filename}")


def plot_answer_distractor_f1(
    results_by_model: Dict[str, Dict],
    dataset_name: str,
    output_dir: str,
    timestamp: str
) -> None:
    """
    Plot Answer F1 vs Distractor F1 comparison across models.
    
    Args:
        results_by_model: Results for multiple models
        dataset_name: Name of the dataset
        output_dir: Output directory
        timestamp: Timestamp for filename
    """
    fig_width = 5.0
    fig_height = 3.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    models = list(results_by_model.keys())
    answer_f1s = []
    distractor_f1s = []
    
    for model in models:
        if dataset_name in results_by_model[model]['datasets']:
            metrics = results_by_model[model]['datasets'][dataset_name]['metrics']
            answer_f1s.append(metrics['answer_metrics']['F1'])
            distractor_f1s.append(metrics['distractor_metrics']['F1'])
        else:
            answer_f1s.append(0)
            distractor_f1s.append(0)
    
    x = np.arange(len(models))
    width = 0.35
    
    # Create grouped bar plot
    bars1 = ax.bar(x - width/2, answer_f1s, width, 
                   label='Answer F1',
                   color=COLORS['answer'],
                   edgecolor='black',
                   linewidth=0.5)
    
    bars2 = ax.bar(x + width/2, distractor_f1s, width,
                   label='Distractor F1',
                   color=COLORS['distractor'],
                   edgecolor='black',
                   linewidth=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Set axes
    ax.set_xlabel('Model', fontsize=9)
    ax.set_ylabel('F1 Score', fontsize=9)
    ax.set_title(f'{dataset_name} - Answer vs Distractor F1', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.1)
    
    # Grid settings
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Spine settings
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'answer_distractor_f1_{dataset_name}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    logger.info(f"Saved Answer/Distractor F1 plot: {filename}")


def create_metrics_table(
    results: Dict,
    dataset_name: str,
    output_dir: str,
    timestamp: str
) -> None:
    """
    Create a formatted table of metrics.
    
    Args:
        results: Evaluation results
        dataset_name: Name of the dataset
        output_dir: Output directory
        timestamp: Timestamp for filename
    """
    if dataset_name not in results['datasets']:
        logger.warning(f"No results for dataset {dataset_name}")
        return
    
    metrics = results['datasets'][dataset_name]['metrics']
    model_name = results['model']
    
    # Create table data
    table_data = {
        'Metric': ['Pr-T', 'Pr-F', 'Co-T', 'Co-F', 
                   'Answer Precision', 'Answer Recall', 'Answer F1',
                   'Distractor Precision', 'Distractor Recall', 'Distractor F1',
                   'F1 Gap', 'Lucky-hit probability', 'Pure skill'],
        'Value': [
            f"{metrics['pr_co_metrics']['Pr-T']} ({metrics['pr_co_metrics']['Pr-T']/metrics['pr_co_metrics']['total']:.1%})",
            f"{metrics['pr_co_metrics']['Pr-F']} ({metrics['pr_co_metrics']['Pr-F']/metrics['pr_co_metrics']['total']:.1%})",
            f"{metrics['pr_co_metrics']['Co-T']} ({metrics['pr_co_metrics']['Co-T']/metrics['pr_co_metrics']['total']:.1%})",
            f"{metrics['pr_co_metrics']['Co-F']} ({metrics['pr_co_metrics']['Co-F']/metrics['pr_co_metrics']['total']:.1%})",
            f"{metrics['answer_metrics']['Precision']:.3f}",
            f"{metrics['answer_metrics']['Recall']:.3f}",
            f"{metrics['answer_metrics']['F1']:.3f}",
            f"{metrics['distractor_metrics']['Precision']:.3f}",
            f"{metrics['distractor_metrics']['Recall']:.3f}",
            f"{metrics['distractor_metrics']['F1']:.3f}",
            f"{metrics['f1_gap']:.3f}",
            f"{metrics['lucky_hit_probability']:.4f}",
            f"{metrics['pure_skill']:.3f}"
        ]
    }
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    filename = f'metrics_table_{model_name}_{dataset_name}_{timestamp}.csv'
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    logger.info(f"Saved metrics table: {filename}")


def create_visualizations(
    results: Dict,
    output_dir: str,
    timestamp: str
) -> None:
    """
    Create all visualizations for the evaluation results.
    
    Args:
        results: Complete results dictionary
        output_dir: Directory to save figures
        timestamp: Timestamp for filenames
    """
    model_name = results['model'].replace('/', '_')
    
    # Create visualization for each dataset
    for dataset_name, dataset_results in results['datasets'].items():
        # 1. Position bias distribution
        plot_position_bias(
            dataset_results['position_bias'],
            dataset_name,
            model_name,
            output_dir,
            timestamp
        )
        
        # 2. Position bias vs Inverse bias comparison
        plot_bias_comparison(
            dataset_results['position_bias'],
            dataset_results['inverse_bias_dist'],
            dataset_name,
            model_name,
            output_dir,
            timestamp
        )
        
        # 3. Create metrics table
        create_metrics_table(
            results,
            dataset_name,
            output_dir,
            timestamp
        )
    
    logger.info(f"All visualizations created for {model_name}")


# Example usage
if __name__ == "__main__":
    # Test position bias plot
    test_bias = {'A': 0.237, 'B': 0.247, 'C': 0.253, 'D': 0.263}
    plot_position_bias(test_bias, 'MMLU', 'test_model', '.', '20250101_120000')