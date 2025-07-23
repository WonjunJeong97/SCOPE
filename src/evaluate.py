# src/evaluate.py
"""
Evaluation module for SCOPE framework.
Implements metrics from the paper: Pr-T/F, Co-T/F, Answer/Distractor F1, and lucky-hit probability.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate_single_question(
    model,
    question_data: Dict,
    num_trials: int = 5,
    temperature: float = 1.0
) -> Dict:
    """
    Evaluate a single question multiple times.
    
    Args:
        model: Model instance with generate() method
        question_data: Processed question with rearranged choices
        num_trials: Number of repetitions (default: 5)
        temperature: Temperature for generation
        
    Returns:
        Dictionary with question_id, responses, and analysis
    """
    responses = []
    response_details = []
    
    # Format prompt according to paper
    prompt_text = question_data['question'] + "\n"
    for label in sorted(question_data['choices'].keys()):
        prompt_text += f"{question_data['choices'][label]}\n"
    
    # Run multiple trials
    for trial in range(num_trials):
        try:
            response = model.generate(prompt_text, max_tokens=1, temperature=temperature)
            answer = response.strip().upper()
            
            # Record response
            responses.append(answer)
            
            # Check if response is valid
            is_valid = answer in question_data['choices']
            
            # Map back to original label if valid
            if is_valid and 'position_mapping' in question_data:
                original_label = question_data['position_mapping'].get(answer, answer)
            else:
                original_label = answer if is_valid else None
                
            response_details.append({
                'trial': trial + 1,
                'response': answer,
                'is_valid': is_valid,
                'original_label': original_label
            })
            
        except Exception as e:
            logger.error(f"Error in trial {trial + 1}: {e}")
            responses.append(None)
            response_details.append({
                'trial': trial + 1,
                'response': None,
                'is_valid': False,
                'original_label': None
            })
            time.sleep(1)  # Brief pause on error
    
    # Analyze responses
    valid_responses = [r for r in responses if r and r in question_data['choices']]
    
    # Map responses back to original labels
    original_labels = []
    for resp in valid_responses:
        if 'position_mapping' in question_data and resp in question_data['position_mapping']:
            original_labels.append(question_data['position_mapping'][resp])
        else:
            original_labels.append(resp)
    
    # Count correct responses
    correct_answer = question_data['original_correct_answer']
    correct_count = sum(1 for label in original_labels if label == correct_answer)
    
    # Determine most frequent response
    if original_labels:
        response_counter = Counter(original_labels)
        most_common_response, most_common_count = response_counter.most_common(1)[0]
    else:
        most_common_response, most_common_count = None, 0
    
    return {
        'question_id': question_data.get('id', 'unknown'),
        'correct_answer': correct_answer,
        'responses': responses,
        'original_label_responses': original_labels,
        'response_details': response_details,
        'correct_count': correct_count,
        'total_valid_responses': len(valid_responses),
        'most_common_response': most_common_response,
        'most_common_count': most_common_count
    }


def calculate_pr_co_metrics(evaluation_results: List[Dict]) -> Dict:
    """
    Calculate Preference (Pr) and Consistency (Co) metrics.
    
    Pr-T: Correct answer chosen ≥3 times out of 5
    Pr-F: Correct answer chosen ≤2 times out of 5
    Co-T: Correct answer chosen 5 times out of 5
    Co-F: Same incorrect answer chosen 5 times out of 5
    
    Args:
        evaluation_results: List of evaluation results for all questions
        
    Returns:
        Dictionary with Pr and Co metrics
    """
    pr_t_count = 0
    pr_f_count = 0
    co_t_count = 0
    co_f_count = 0
    
    detailed_results = []
    
    for result in evaluation_results:
        correct_count = result['correct_count']
        total_valid = result['total_valid_responses']
        most_common_response = result['most_common_response']
        most_common_count = result['most_common_count']
        correct_answer = result['correct_answer']
        
        # Skip if no valid responses
        if total_valid == 0:
            continue
        
        # Preference metrics (based on majority)
        is_pr_t = correct_count >= 3  # At least 3 out of 5 correct
        is_pr_f = not is_pr_t
        
        # Consistency metrics (all 5 responses the same)
        is_co_t = correct_count == 5  # All 5 correct
        is_co_f = (most_common_count == 5 and most_common_response != correct_answer)  # All 5 same wrong
        
        if is_pr_t:
            pr_t_count += 1
        if is_pr_f:
            pr_f_count += 1
        if is_co_t:
            co_t_count += 1
        if is_co_f:
            co_f_count += 1
        
        detailed_results.append({
            'question_id': result['question_id'],
            'correct_count': correct_count,
            'pr_result': 'T' if is_pr_t else 'F',
            'co_result': 'T' if is_co_t else ('F' if is_co_f else 'Neither'),
            'responses': result['original_label_responses']
        })
    
    total_questions = len(detailed_results)
    
    return {
        'Pr-T': pr_t_count,
        'Pr-F': pr_f_count,
        'Co-T': co_t_count,
        'Co-F': co_f_count,
        'total': total_questions,
        'detailed_results': detailed_results
    }


def calculate_answer_distractor_f1(pr_co_metrics: Dict) -> Dict:
    """
    Calculate Answer and Distractor Precision, Recall, and F1 scores.
    
    From the paper:
    - Answer Precision (AP) = Co-T / (Co-T + Co-F)
    - Answer Recall (AR) = Co-T / (Pr-T + Co-T)
    - Answer F1 = 2 * (AP * AR) / (AP + AR)
    
    Similar for Distractor metrics with Co-F and Pr-F.
    
    Args:
        pr_co_metrics: Dictionary with Pr and Co counts
        
    Returns:
        Dictionary with Answer and Distractor metrics
    """
    pr_t = pr_co_metrics['Pr-T']
    pr_f = pr_co_metrics['Pr-F']
    co_t = pr_co_metrics['Co-T']
    co_f = pr_co_metrics['Co-F']
    
    # Answer metrics
    # Precision: Of all consistent responses, how many were correct?
    ap = co_t / (co_t + co_f) if (co_t + co_f) > 0 else 0.0
    
    # Recall: Of all questions where correct was preferred, how many were consistent?
    # Note: Co-T is a subset of Pr-T, so we use just Pr-T in denominator
    ar = co_t / pr_t if pr_t > 0 else 0.0
    
    # F1
    answer_f1 = 2 * (ap * ar) / (ap + ar) if (ap + ar) > 0 else 0.0
    
    # Distractor metrics
    # Precision: Of all consistent responses, how many were incorrect?
    dp = co_f / (co_t + co_f) if (co_t + co_f) > 0 else 0.0
    
    # Recall: Of all questions where incorrect was preferred, how many were consistent?
    dr = co_f / pr_f if pr_f > 0 else 0.0
    
    # F1
    distractor_f1 = 2 * (dp * dr) / (dp + dr) if (dp + dr) > 0 else 0.0
    
    return {
        'Answer': {
            'Precision': ap,
            'Recall': ar,
            'F1': answer_f1
        },
        'Distractor': {
            'Precision': dp,
            'Recall': dr,
            'F1': distractor_f1
        },
        'F1_gap': answer_f1 - distractor_f1  # Higher gap is better
    }


def calculate_lucky_hit_probability(
    position_bias: Dict[str, float],
    inverse_bias_dist: Dict[str, float]
) -> float:
    """
    Calculate lucky-hit probability ℓ = Σ(pi * qi).
    
    This represents the probability of getting correct answer by position bias alone.
    
    Args:
        position_bias: Original position bias distribution P
        inverse_bias_dist: Inverse bias distribution Q
        
    Returns:
        Lucky-hit probability ℓ
    """
    lucky_hit_prob = 0.0
    
    for position in position_bias:
        if position in inverse_bias_dist:
            pi = position_bias[position]
            qi = inverse_bias_dist[position]
            lucky_hit_prob += pi * qi
    
    # Theoretical bound is 1/n
    n = len(position_bias)
    theoretical_bound = 1.0 / n
    
    logger.info(f"Lucky-hit probability ℓ = {lucky_hit_prob:.4f} (theoretical bound: {theoretical_bound:.4f})")
    
    return lucky_hit_prob


def run_evaluation_batch(
    model,
    questions: List[Dict],
    num_trials: int = 5,
    temperature: float = 1.0,
    show_progress: bool = True
) -> List[Dict]:
    """
    Run evaluation on a batch of questions.
    
    Args:
        model: Model instance
        questions: List of processed questions
        num_trials: Number of repetitions per question
        temperature: Temperature for generation
        show_progress: Whether to show progress bar
        
    Returns:
        List of evaluation results
    """
    evaluation_results = []
    
    iterator = tqdm(questions, desc="Evaluating") if show_progress else questions
    
    for question in iterator:
        result = evaluate_single_question(
            model,
            question,
            num_trials=num_trials,
            temperature=temperature
        )
        evaluation_results.append(result)
        
        # Brief pause to avoid rate limiting
        time.sleep(0.1)
    
    return evaluation_results


def calculate_all_metrics(
    evaluation_results: List[Dict],
    position_bias: Dict[str, float],
    inverse_bias_dist: Dict[str, float]
) -> Dict:
    """
    Calculate all metrics for the evaluation.
    
    Args:
        evaluation_results: List of evaluation results
        position_bias: Original position bias
        inverse_bias_dist: Inverse bias distribution
        
    Returns:
        Dictionary with all metrics
    """
    # Calculate Pr/Co metrics
    pr_co_metrics = calculate_pr_co_metrics(evaluation_results)
    
    # Calculate Answer/Distractor F1
    f1_metrics = calculate_answer_distractor_f1(pr_co_metrics)
    
    # Calculate lucky-hit probability
    lucky_hit = calculate_lucky_hit_probability(position_bias, inverse_bias_dist)
    
    # Calculate pure skill (Answer F1 - lucky hit)
    pure_skill = f1_metrics['Answer']['F1'] - lucky_hit
    
    # Compile all metrics
    return {
        'pr_co_metrics': pr_co_metrics,
        'answer_metrics': f1_metrics['Answer'],
        'distractor_metrics': f1_metrics['Distractor'],
        'f1_gap': f1_metrics['F1_gap'],
        'lucky_hit_probability': lucky_hit,
        'pure_skill': pure_skill,
        'summary': {
            'total_questions': pr_co_metrics['total'],
            'Pr-T': pr_co_metrics['Pr-T'],
            'Pr-F': pr_co_metrics['Pr-F'],
            'Co-T': pr_co_metrics['Co-T'],
            'Co-F': pr_co_metrics['Co-F'],
            'Answer_F1': f1_metrics['Answer']['F1'],
            'Distractor_F1': f1_metrics['Distractor']['F1'],
            'Lucky_hit': lucky_hit,
            'Pure_skill': pure_skill
        }
    }


# Example usage
if __name__ == "__main__":
    # Test metrics calculation with dummy data
    test_results = [
        {
            'question_id': 'q1',
            'correct_answer': 'A',
            'correct_count': 5,
            'total_valid_responses': 5,
            'most_common_response': 'A',
            'most_common_count': 5,
            'original_label_responses': ['A', 'A', 'A', 'A', 'A']
        },
        {
            'question_id': 'q2',
            'correct_answer': 'B',
            'correct_count': 3,
            'total_valid_responses': 5,
            'most_common_response': 'B',
            'most_common_count': 3,
            'original_label_responses': ['B', 'B', 'B', 'C', 'D']
        },
        {
            'question_id': 'q3',
            'correct_answer': 'C',
            'correct_count': 0,
            'total_valid_responses': 5,
            'most_common_response': 'A',
            'most_common_count': 5,
            'original_label_responses': ['A', 'A', 'A', 'A', 'A']
        }
    ]
    
    # Test Pr/Co metrics
    pr_co = calculate_pr_co_metrics(test_results)
    print("Pr/Co metrics:", pr_co)
    
    # Test F1 metrics
    f1_metrics = calculate_answer_distractor_f1(pr_co)
    print("F1 metrics:", f1_metrics)
    
    # Test lucky hit
    test_bias = {'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1}
    test_inverse = {'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.4}
    lucky_hit = calculate_lucky_hit_probability(test_bias, test_inverse)
    print(f"Lucky hit probability: {lucky_hit:.4f}")