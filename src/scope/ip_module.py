# src/scope/ip_module.py
"""
Inverse-Positioning (IP) Module
Implements the position bias measurement and inverse-bias sampling from the SCOPE paper.
"""

import random
import string
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def generate_random_string(min_length: int = 4, max_length: int = 12) -> str:
    """Generate random meaningless string for null prompts."""
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def generate_null_prompts(num_choices: int, num_prompts: int = 1000) -> List[Dict]:
    """
    Generate null prompts for position bias measurement.
    
    Args:
        num_choices: Number of choices (4 for MMLU, 5 for CSQA)
        num_prompts: Number of null prompts to generate
        
    Returns:
        List of null prompt dictionaries
    """
    null_prompt_template = "You must choose one. If you had to pick, which would it be?"
    prompts = []
    
    for _ in range(num_prompts):
        # Generate random meaningless strings for each choice
        choices = {}
        for i in range(num_choices):
            label = chr(65 + i)  # A, B, C, D, (E)
            # Generate random string, sometimes with space for variety
            if random.random() < 0.3:
                # Two words
                text = f"{generate_random_string()} {generate_random_string()}"
            else:
                # Single word
                text = generate_random_string()
            choices[label] = text
            
        prompts.append({
            'question': null_prompt_template,
            'choices': choices,
            'type': 'null_prompt'
        })
    
    return prompts


def measure_position_bias(model, null_prompts: List[Dict]) -> Dict[str, float]:
    """
    Measure position bias P = (p1, ..., pn) using null prompts.
    
    Args:
        model: Model instance with generate() method
        null_prompts: List of null prompts
        
    Returns:
        Dictionary mapping position labels to selection probabilities
    """
    position_counts = Counter()
    total_valid_responses = 0
    
    logger.info(f"Measuring position bias with {len(null_prompts)} null prompts...")
    
    for i, prompt in enumerate(null_prompts):
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{len(null_prompts)}")
            
        try:
            # Format prompt according to paper
            prompt_text = prompt['question'] + "\n"
            for label, text in sorted(prompt['choices'].items()):
                prompt_text += f"{text}\n"
            
            # Get model response
            response = model.generate(prompt_text, max_tokens=1, temperature=1.0)
            
            # Extract answer - look for single letter response
            answer = response.strip().upper()
            
            # Check if response is valid position
            if answer in prompt['choices']:
                position_counts[answer] += 1
                total_valid_responses += 1
            else:
                logger.debug(f"Invalid response: {response}")
                
        except Exception as e:
            logger.error(f"Error in null prompt {i}: {e}")
            continue
    
    # Calculate probabilities
    position_probs = {}
    num_choices = len(null_prompts[0]['choices']) if null_prompts else 0
    
    for i in range(num_choices):
        label = chr(65 + i)
        count = position_counts.get(label, 0)
        # Add small epsilon to avoid division by zero
        prob = (count / total_valid_responses) if total_valid_responses > 0 else (1.0 / num_choices)
        position_probs[label] = max(prob, 1e-6)  # Ensure no zero probabilities
    
    # Normalize to ensure sum = 1
    total_prob = sum(position_probs.values())
    position_probs = {k: v/total_prob for k, v in position_probs.items()}
    
    logger.info(f"Position bias measured: {position_probs}")
    logger.info(f"Valid responses: {total_valid_responses}/{len(null_prompts)}")
    
    return position_probs


def calculate_inverse_bias_distribution(position_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate inverse bias distribution Q for answer placement.
    
    From the paper: qi = (1/pi) / Σ(1/pj)
    
    Args:
        position_probs: Position bias distribution P
        
    Returns:
        Inverse bias distribution Q
    """
    inverse_probs = {}
    
    # Calculate inverse weights
    inverse_weights = {}
    for label, prob in position_probs.items():
        # Add small epsilon to avoid division by zero
        inverse_weights[label] = 1.0 / (prob + 1e-10)
    
    # Normalize to get distribution
    total_inverse_weight = sum(inverse_weights.values())
    
    for label, weight in inverse_weights.items():
        inverse_probs[label] = weight / total_inverse_weight
    
    # Verify the theoretical property: Σ(pi * qi) ≤ 1/n
    lucky_hit_prob = sum(position_probs[label] * inverse_probs[label] 
                        for label in position_probs)
    n = len(position_probs)
    theoretical_bound = 1.0 / n
    
    logger.info(f"Inverse bias distribution calculated: {inverse_probs}")
    logger.info(f"Lucky-hit probability ℓ = {lucky_hit_prob:.4f} (theoretical bound: {theoretical_bound:.4f})")
    
    return inverse_probs


def sample_answer_position(inverse_bias_dist: Dict[str, float]) -> str:
    """
    Sample answer position from inverse bias distribution.
    
    Args:
        inverse_bias_dist: Inverse bias distribution Q
        
    Returns:
        Sampled position label
    """
    positions = list(inverse_bias_dist.keys())
    probabilities = list(inverse_bias_dist.values())
    
    # Ensure probabilities sum to 1 (handle floating point errors)
    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()
    
    sampled_position = np.random.choice(positions, p=probabilities)
    
    return sampled_position


def apply_inverse_positioning(question_data: Dict, inverse_bias_dist: Dict[str, float]) -> Tuple[str, Dict[str, str]]:
    """
    Apply inverse positioning to place the correct answer.
    
    Args:
        question_data: Question with choices and correct answer
        inverse_bias_dist: Inverse bias distribution Q
        
    Returns:
        Tuple of (answer_position, position_to_original_label_mapping)
    """
    # Sample position for correct answer
    answer_position = sample_answer_position(inverse_bias_dist)
    
    # Create mapping from new positions to original labels
    correct_answer = question_data['answer']
    choices = question_data['choices']
    
    # Get all positions
    all_positions = sorted(choices.keys())
    
    # Remove the position where we'll place the answer
    available_positions = [pos for pos in all_positions if pos != answer_position]
    
    # Get all distractor labels
    distractor_labels = [label for label in choices.keys() if label != correct_answer]
    
    # Randomly assign distractors to remaining positions
    random.shuffle(available_positions)
    
    # Create position mapping
    position_mapping = {answer_position: correct_answer}
    
    for i, distractor_label in enumerate(distractor_labels):
        if i < len(available_positions):
            position_mapping[available_positions[i]] = distractor_label
    
    return answer_position, position_mapping


# Example usage and testing
if __name__ == "__main__":
    # Test null prompt generation
    null_prompts_5 = generate_null_prompts(num_choices=5, num_prompts=10)
    print("Sample null prompt (5 choices):")
    print(null_prompts_5[0])
    
    # Test inverse bias calculation
    sample_bias = {'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1}
    inverse_dist = calculate_inverse_bias_distribution(sample_bias)
    print(f"\nSample bias: {sample_bias}")
    print(f"Inverse distribution: {inverse_dist}")
    
    # Test answer position sampling
    positions = [sample_answer_position(inverse_dist) for _ in range(100)]
    position_counts = Counter(positions)
    print(f"\nSampled positions (100 trials): {dict(position_counts)}")