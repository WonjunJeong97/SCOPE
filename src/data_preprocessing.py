# src/data_preprocessing.py
"""
Data preprocessing module for SCOPE framework.
Handles loading and preparing fixed datasets for evaluation.
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional
import random

logger = logging.getLogger(__name__)


def load_fixed_datasets(
    csqa_path: str = "data/fixed/csqa_500_fixed.json",
    mmlu_path: str = "data/fixed/mmlu_500_fixed.json"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load fixed CSQA and MMLU datasets.
    
    Args:
        csqa_path: Path to CSQA fixed dataset
        mmlu_path: Path to MMLU fixed dataset
        
    Returns:
        Tuple of (csqa_data, mmlu_data)
    """
    # Load CSQA
    if os.path.exists(csqa_path):
        logger.info(f"Loading CSQA data from {csqa_path}")
        with open(csqa_path, 'r', encoding='utf-8') as f:
            csqa_data = json.load(f)
        logger.info(f"Loaded {len(csqa_data)} CSQA questions")
    else:
        logger.warning(f"CSQA file not found at {csqa_path}")
        csqa_data = []
    
    # Load MMLU
    if os.path.exists(mmlu_path):
        logger.info(f"Loading MMLU data from {mmlu_path}")
        with open(mmlu_path, 'r', encoding='utf-8') as f:
            mmlu_data = json.load(f)
        logger.info(f"Loaded {len(mmlu_data)} MMLU questions")
    else:
        logger.warning(f"MMLU file not found at {mmlu_path}")
        mmlu_data = []
    
    # Validate data format
    csqa_data = validate_dataset(csqa_data, "CSQA", expected_choices=5)
    mmlu_data = validate_dataset(mmlu_data, "MMLU", expected_choices=4)
    
    return csqa_data, mmlu_data


def validate_dataset(
    data: List[Dict],
    dataset_name: str,
    expected_choices: int
) -> List[Dict]:
    """
    Validate and clean dataset.
    
    Args:
        data: List of question dictionaries
        dataset_name: Name of the dataset for logging
        expected_choices: Expected number of choices
        
    Returns:
        Validated dataset
    """
    validated_data = []
    
    for i, item in enumerate(data):
        try:
            # Check required fields
            required_fields = ['id', 'question', 'choices', 'answer']
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate choices
            choices = item['choices']
            if not isinstance(choices, dict):
                raise ValueError("Choices must be a dictionary")
            
            # Check number of choices
            if len(choices) != expected_choices:
                logger.warning(
                    f"{dataset_name} item {i} has {len(choices)} choices, expected {expected_choices}"
                )
            
            # Check answer is in choices
            if item['answer'] not in choices:
                raise ValueError(f"Answer '{item['answer']}' not in choices")
            
            # Add num_choices if missing
            if 'num_choices' not in item:
                item['num_choices'] = len(choices)
            
            validated_data.append(item)
            
        except Exception as e:
            logger.error(f"Error validating {dataset_name} item {i}: {e}")
            continue
    
    logger.info(f"Validated {len(validated_data)}/{len(data)} items from {dataset_name}")
    
    return validated_data


def prepare_evaluation_data(
    dataset: List[Dict],
    dataset_name: str,
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Prepare dataset for evaluation.
    
    Args:
        dataset: Full dataset
        dataset_name: Name of the dataset
        sample_size: Optional sample size (None for full dataset)
        random_seed: Random seed for sampling
        
    Returns:
        Prepared dataset
    """
    if sample_size and sample_size < len(dataset):
        if random_seed is not None:
            random.seed(random_seed)
        
        logger.info(f"Sampling {sample_size} questions from {dataset_name}")
        sampled_data = random.sample(dataset, sample_size)
    else:
        sampled_data = dataset
    
    # Add dataset tag
    for item in sampled_data:
        item['dataset'] = dataset_name
    
    return sampled_data


def get_dataset_statistics(data: List[Dict], dataset_name: str) -> Dict:
    """
    Calculate statistics for a dataset.
    
    Args:
        data: Dataset
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with statistics
    """
    if not data:
        return {
            'dataset': dataset_name,
            'total_questions': 0,
            'error': 'No data'
        }
    
    # Count answer distribution
    answer_counts = {}
    choice_counts = {}
    
    for item in data:
        answer = item.get('answer')
        if answer:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        num_choices = len(item.get('choices', {}))
        choice_counts[num_choices] = choice_counts.get(num_choices, 0) + 1
    
    # Calculate average question and choice lengths
    avg_question_length = sum(len(item.get('question', '')) for item in data) / len(data)
    
    all_choice_lengths = []
    for item in data:
        for choice_text in item.get('choices', {}).values():
            all_choice_lengths.append(len(choice_text))
    
    avg_choice_length = sum(all_choice_lengths) / len(all_choice_lengths) if all_choice_lengths else 0
    
    return {
        'dataset': dataset_name,
        'total_questions': len(data),
        'answer_distribution': answer_counts,
        'choice_count_distribution': choice_counts,
        'avg_question_length': avg_question_length,
        'avg_choice_length': avg_choice_length
    }


def combine_datasets(
    csqa_data: List[Dict],
    mmlu_data: List[Dict],
    shuffle: bool = True,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Combine CSQA and MMLU datasets.
    
    Args:
        csqa_data: CSQA dataset
        mmlu_data: MMLU dataset
        shuffle: Whether to shuffle combined data
        random_seed: Random seed for shuffling
        
    Returns:
        Combined dataset
    """
    combined = csqa_data + mmlu_data
    
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(combined)
    
    logger.info(f"Combined dataset: {len(combined)} questions total")
    
    return combined


def filter_by_subject(
    data: List[Dict],
    subjects: List[str],
    dataset_name: str = "MMLU"
) -> List[Dict]:
    """
    Filter dataset by subject (mainly for MMLU).
    
    Args:
        data: Dataset to filter
        subjects: List of subjects to include
        dataset_name: Dataset name for logging
        
    Returns:
        Filtered dataset
    """
    if dataset_name != "MMLU":
        logger.warning(f"Subject filtering requested for {dataset_name}, but subjects are MMLU-specific")
        return data
    
    filtered = [
        item for item in data
        if item.get('subject') in subjects
    ]
    
    logger.info(f"Filtered {dataset_name}: {len(filtered)}/{len(data)} questions from subjects: {subjects}")
    
    return filtered


# Utility functions for data inspection
def print_sample_questions(data: List[Dict], n: int = 3, dataset_name: str = "Dataset"):
    """Print sample questions from dataset."""
    print(f"\n=== Sample questions from {dataset_name} ===")
    
    for i, item in enumerate(data[:n]):
        print(f"\nQuestion {i+1} (ID: {item.get('id', 'N/A')}):")
        print(f"Q: {item.get('question', 'N/A')}")
        print("Choices:")
        
        choices = item.get('choices', {})
        for label in sorted(choices.keys()):
            print(f"  {label}) {choices[label]}")
        
        print(f"Answer: {item.get('answer', 'N/A')}")
        
        if 'subject' in item:
            print(f"Subject: {item['subject']}")


def check_data_files():
    """Check if required data files exist."""
    csqa_path = "data/fixed/csqa_500_fixed.json"
    mmlu_path = "data/fixed/mmlu_500_fixed.json"
    
    files_exist = {
        'CSQA': os.path.exists(csqa_path),
        'MMLU': os.path.exists(mmlu_path)
    }
    
    print("\n=== Data File Check ===")
    for dataset, exists in files_exist.items():
        status = "✓ Found" if exists else "✗ Missing"
        print(f"{dataset}: {status}")
    
    if not all(files_exist.values()):
        print("\n⚠️  Missing data files! Please ensure you have:")
        print(f"  - {csqa_path}")
        print(f"  - {mmlu_path}")
        print("\nYou can download the raw data using scripts/download_data.sh")
    
    return all(files_exist.values())


# Main function for testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check data files
    if not check_data_files():
        exit(1)
    
    # Load datasets
    csqa_data, mmlu_data = load_fixed_datasets()
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    csqa_stats = get_dataset_statistics(csqa_data, "CSQA")
    mmlu_stats = get_dataset_statistics(mmlu_data, "MMLU")
    
    for stats in [csqa_stats, mmlu_stats]:
        print(f"\n{stats['dataset']}:")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Answer distribution: {stats['answer_distribution']}")
        print(f"  Avg question length: {stats['avg_question_length']:.1f} chars")
        print(f"  Avg choice length: {stats['avg_choice_length']:.1f} chars")
    
    # Print samples
    if csqa_data:
        print_sample_questions(csqa_data, n=2, dataset_name="CSQA")
    if mmlu_data:
        print_sample_questions(mmlu_data, n=2, dataset_name="MMLU")
    
    # Test combining
    combined = combine_datasets(
        prepare_evaluation_data(csqa_data, "CSQA", sample_size=10),
        prepare_evaluation_data(mmlu_data, "MMLU", sample_size=10),
        shuffle=True,
        random_seed=42
    )
    print(f"\n=== Combined Dataset ===")
    print(f"Total: {len(combined)} questions")