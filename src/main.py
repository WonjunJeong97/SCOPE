# src/main.py
"""
Main entry point for SCOPE evaluation framework.
Implements the complete pipeline from the paper.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SCOPE modules
from src.data_preprocessing import load_fixed_datasets, prepare_evaluation_data, get_dataset_statistics
from src.scope.ip_module import (
    generate_null_prompts, 
    measure_position_bias, 
    calculate_inverse_bias_distribution
)
from src.scope.ss_module import create_scope_processor
from src.models import get_model
from src.evaluate import run_evaluation_batch, calculate_all_metrics
from src.visualization import create_visualizations
from src.results import save_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_full_evaluation(
    model_name: str,
    dataset_name: str = "both",
    config_path: str = "configs/default.yaml",
    output_dir: Optional[str] = None,
    test_mode: bool = False,
    ablation_mode: Optional[str] = None
) -> Dict:
    """
    Run complete SCOPE evaluation pipeline.
    
    Args:
        model_name: Name of the model to evaluate
        dataset_name: "csqa", "mmlu", or "both"
        config_path: Path to configuration file
        output_dir: Directory for outputs (default: current directory)
        test_mode: If True, use small samples for testing
        ablation_mode: None, "IP+SS", "¬IP+SS", or "IP+¬SS"
        
    Returns:
        Dictionary with all results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting SCOPE evaluation")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Test mode: {test_mode}")
    logger.info(f"Ablation mode: {ablation_mode}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = get_model(model_name)
    
    # Load datasets
    logger.info("Loading datasets...")
    csqa_data, mmlu_data = load_fixed_datasets(
        config['data']['csqa_path'],
        config['data']['mmlu_path']
    )
    
    # Prepare evaluation data based on dataset choice
    evaluation_data = []
    
    if dataset_name in ["csqa", "both"]:
        csqa_sample_size = 10 if test_mode else None
        csqa_eval = prepare_evaluation_data(csqa_data, "CSQA", csqa_sample_size, random_seed=42)
        evaluation_data.extend(csqa_eval)
        logger.info(f"Prepared {len(csqa_eval)} CSQA questions")
    
    if dataset_name in ["mmlu", "both"]:
        mmlu_sample_size = 10 if test_mode else None
        mmlu_eval = prepare_evaluation_data(mmlu_data, "MMLU", mmlu_sample_size, random_seed=42)
        evaluation_data.extend(mmlu_eval)
        logger.info(f"Prepared {len(mmlu_eval)} MMLU questions")
    
    # Store results by dataset type
    results_by_dataset = {
        'CSQA': {'questions': [], 'num_choices': 5},
        'MMLU': {'questions': [], 'num_choices': 4}
    }
    
    # Separate questions by dataset
    for q in evaluation_data:
        dataset = q.get('dataset', 'CSQA')
        results_by_dataset[dataset]['questions'].append(q)
    
    # Process each dataset separately (different number of choices)
    all_results = {
        'model': model_name,
        'timestamp': timestamp,
        'config': config,
        'datasets': {}
    }
    
    for dataset_type, dataset_info in results_by_dataset.items():
        questions = dataset_info['questions']
        num_choices = dataset_info['num_choices']
        
        if not questions:
            continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {dataset_type} ({len(questions)} questions, {num_choices} choices)")
        logger.info(f"{'='*50}")
        
        # Step 1: Measure position bias using null prompts
        logger.info(f"\n Measuring position bias for {dataset_type}...")
        
        null_prompt_count = config['null_prompt']['count']
        if test_mode:
            null_prompt_count = min(null_prompt_count, 50)  # Reduce for testing
        
        null_prompts = generate_null_prompts(num_choices, null_prompt_count)
        position_bias = measure_position_bias(model, null_prompts)
        
        # Step 2: Calculate inverse bias distribution
        logger.info(f"\n Calculating inverse bias distribution...")
        inverse_bias_dist = calculate_inverse_bias_distribution(position_bias)
        
        # Step 3: Create SCOPE processor based on ablation mode
        use_ip = ablation_mode != "¬IP+SS"
        use_ss = ablation_mode != "IP+¬SS"
        
        logger.info(f"\n SCOPE configuration - IP: {use_ip}, SS: {use_ss}")
        
        if use_ip:
            scope_processor = create_scope_processor(
                inverse_bias_dist,
                config['semantic_similarity']['model']
            )
        else:
            # Use uniform distribution if IP is disabled
            uniform_dist = {chr(65+i): 1/num_choices for i in range(num_choices)}
            scope_processor = create_scope_processor(
                uniform_dist,
                config['semantic_similarity']['model']
            )
        
        # Step 4: Apply SCOPE to questions
        logger.info(f"\n Applying SCOPE to rearrange choices...")
        processed_questions = []
        
        for question in questions:
            processed_q = scope_processor(question, use_ss=use_ss)
            processed_questions.append(processed_q)
        
        # Step 5: Evaluate with multiple trials
        logger.info(f"\n Running evaluation with {config['evaluation']['num_trials']} trials per question...")
        
        evaluation_results = run_evaluation_batch(
            model,
            processed_questions,
            num_trials=config['evaluation']['num_trials'],
            temperature=config['evaluation']['temperature'],
            show_progress=True
        )
        
        # Step 6: Calculate metrics
        logger.info(f"\n Calculating metrics...")
        
        metrics = calculate_all_metrics(
            evaluation_results,
            position_bias,
            inverse_bias_dist if use_ip else uniform_dist
        )
        
        # Store results
        all_results['datasets'][dataset_type] = {
            'num_questions': len(questions),
            'num_choices': num_choices,
            'position_bias': position_bias,
            'inverse_bias_dist': inverse_bias_dist if use_ip else uniform_dist,
            'metrics': metrics,
            'evaluation_results': evaluation_results,
            'ablation_config': {
                'mode': ablation_mode,
                'use_ip': use_ip,
                'use_ss': use_ss
            }
        }
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"{dataset_type} Results Summary")
        print(f"{'='*50}")
        print(f"Total questions: {metrics['summary']['total_questions']}")
        print(f"Pr-T: {metrics['summary']['Pr-T']} ({metrics['summary']['Pr-T']/metrics['summary']['total_questions']:.1%})")
        print(f"Pr-F: {metrics['summary']['Pr-F']} ({metrics['summary']['Pr-F']/metrics['summary']['total_questions']:.1%})")
        print(f"Co-T: {metrics['summary']['Co-T']} ({metrics['summary']['Co-T']/metrics['summary']['total_questions']:.1%})")
        print(f"Co-F: {metrics['summary']['Co-F']} ({metrics['summary']['Co-F']/metrics['summary']['total_questions']:.1%})")
        print(f"\nAnswer Precision: {metrics['answer_metrics']['Precision']:.3f}")
        print(f"Answer Recall: {metrics['answer_metrics']['Recall']:.3f}")
        print(f"Answer F1: {metrics['answer_metrics']['F1']:.3f}")
        print(f"\nDistractor Precision: {metrics['distractor_metrics']['Precision']:.3f}")
        print(f"Distractor Recall: {metrics['distractor_metrics']['Recall']:.3f}")
        print(f"Distractor F1: {metrics['distractor_metrics']['F1']:.3f}")
        print(f"\nF1 Gap (Answer - Distractor): {metrics['f1_gap']:.3f}")
        print(f"Lucky-hit probability: {metrics['lucky_hit_probability']:.4f}")
        print(f"Pure skill: {metrics['pure_skill']:.3f}")
    
    # Step 7: Save results and create visualizations
    logger.info(f"\n Saving results...")
    
    save_results(all_results, output_dir, timestamp)
    
    if config['output']['save_figures']:
        logger.info(f"\n📊 Creating visualizations...")
        create_visualizations(all_results, output_dir, timestamp)
    
    logger.info(f"\n Evaluation complete")
    
    return all_results


def run_ablation_study(
    model_name: str,
    dataset_name: str = "both",
    config_path: str = "configs/default.yaml",
    output_dir: Optional[str] = None,
    test_mode: bool = False
) -> Dict:
    """
    Run ablation study with three conditions.
    
    Args:
        model_name: Name of the model to evaluate
        dataset_name: "csqa", "mmlu", or "both"
        config_path: Path to configuration file
        output_dir: Directory for outputs
        test_mode: If True, use small samples
        
    Returns:
        Dictionary with ablation results
    """
    ablation_modes = ["IP+SS", "¬IP+SS", "IP+¬SS"]
    ablation_results = {}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running ablation study for {model_name}")
    logger.info(f"Modes: {ablation_modes}")
    logger.info(f"{'='*60}")
    
    for mode in ablation_modes:
        logger.info(f"\n\n{'*'*50}")
        logger.info(f"Ablation mode: {mode}")
        logger.info(f"{'*'*50}")
        
        results = run_full_evaluation(
            model_name=model_name,
            dataset_name=dataset_name,
            config_path=config_path,
            output_dir=output_dir,
            test_mode=test_mode,
            ablation_mode=mode
        )
        
        ablation_results[mode] = results
    
    # Create comparison summary
    print(f"\n\n{'='*60}")
    print(f"Ablation Study Summary - {model_name}")
    print(f"{'='*60}")
    
    for dataset_type in ["CSQA", "MMLU"]:
        print(f"\n{dataset_type}:")
        print(f"{'Mode':<10} {'Answer F1':<12} {'Distractor F1':<15} {'Lucky-hit':<12} {'Pure Skill':<12}")
        print("-" * 60)
        
        for mode in ablation_modes:
            if dataset_type in ablation_results[mode]['datasets']:
                metrics = ablation_results[mode]['datasets'][dataset_type]['metrics']
                print(f"{mode:<10} "
                      f"{metrics['answer_metrics']['F1']:<12.3f} "
                      f"{metrics['distractor_metrics']['F1']:<15.3f} "
                      f"{metrics['lucky_hit_probability']:<12.4f} "
                      f"{metrics['pure_skill']:<12.3f}")
    
    return ablation_results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="SCOPE: Structural Corrections for Position and Option biases in Evaluation"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-3.5-turbo, claude-3-haiku)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["csqa", "mmlu", "both"],
        default="both",
        help="Dataset to evaluate on"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: current directory)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with small samples"
    )
    
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study"
    )
    
    parser.add_argument(
        "--ablation-mode",
        type=str,
        choices=["IP+SS", "¬IP+SS", "IP+¬SS"],
        default=None,
        help="Specific ablation mode (if not running full ablation)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.ablation:
            # Run full ablation study
            results = run_ablation_study(
                model_name=args.model,
                dataset_name=args.dataset,
                config_path=args.config,
                output_dir=args.output_dir,
                test_mode=args.test
            )
        else:
            # Run single evaluation
            results = run_full_evaluation(
                model_name=args.model,
                dataset_name=args.dataset,
                config_path=args.config,
                output_dir=args.output_dir,
                test_mode=args.test,
                ablation_mode=args.ablation_mode
            )
        
        print("\n All evaluations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()