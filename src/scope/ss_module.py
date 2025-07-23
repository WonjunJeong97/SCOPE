# src/scope/ss_module.py
"""
Semantic-Spread (SS) Module
Implements semantic similarity-based distractor placement from the SCOPE paper.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class SemanticSimilarityCalculator:
    """Handle semantic similarity calculations using Sentence-BERT."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Sentence-BERT model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        embeddings = self.get_embeddings([text1, text2])
        
        # Calculate cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        cosine_similarity = dot_product / (norm1 * norm2)
        
        return float(cosine_similarity)


def find_ssd(choices: Dict[str, str], correct_answer_label: str, 
             similarity_calculator: SemanticSimilarityCalculator) -> Tuple[str, Dict[str, float]]:
    """
    Find the Semantically Similar Distractor (SSD).
    
    Args:
        choices: Dictionary mapping labels to choice texts
        correct_answer_label: Label of the correct answer
        similarity_calculator: Instance of SemanticSimilarityCalculator
        
    Returns:
        Tuple of (SSD label, similarity scores for all distractors)
    """
    correct_text = choices[correct_answer_label]
    similarities = {}
    
    # Calculate similarity for each distractor
    for label, text in choices.items():
        if label != correct_answer_label:
            similarity = similarity_calculator.calculate_similarity(correct_text, text)
            similarities[label] = similarity
    
    # Find the most similar distractor
    if similarities:
        ssd_label = max(similarities, key=similarities.get)
        logger.debug(f"SSD found: {ssd_label} with similarity {similarities[ssd_label]:.3f}")
    else:
        ssd_label = None
        logger.warning("No distractors found to calculate SSD")
    
    return ssd_label, similarities


def calculate_distance_weights(correct_position: str, available_positions: List[str]) -> Dict[str, float]:
    """
    Calculate exponential distance weights for SSD placement.
    
    From the paper: wj = exp(|i* - j|)
    
    Args:
        correct_position: Position where correct answer is placed
        available_positions: List of available positions for distractors
        
    Returns:
        Dictionary mapping positions to weights
    """
    weights = {}
    
    # Convert position labels to indices
    correct_idx = ord(correct_position) - ord('A')
    
    for pos in available_positions:
        pos_idx = ord(pos) - ord('A')
        distance = abs(correct_idx - pos_idx)
        
        # Exponential weight based on distance
        weight = np.exp(distance)
        weights[pos] = weight
    
    logger.debug(f"Distance weights from position {correct_position}: {weights}")
    
    return weights


def sample_ssd_position(weights: Dict[str, float]) -> str:
    """
    Sample SSD position based on distance weights.
    
    Args:
        weights: Dictionary mapping positions to weights
        
    Returns:
        Sampled position for SSD
    """
    positions = list(weights.keys())
    weight_values = list(weights.values())
    
    # Normalize weights to probabilities
    total_weight = sum(weight_values)
    probabilities = [w / total_weight for w in weight_values]
    
    # Sample position
    sampled_position = np.random.choice(positions, p=probabilities)
    
    return sampled_position


def place_choices_with_debiasing(
    question_data: Dict,
    inverse_bias_dist: Dict[str, float],
    similarity_calculator: Optional[SemanticSimilarityCalculator] = None,
    use_ss: bool = True
) -> Dict:
    """
    Apply both IP and SS modules to place answer choices.
    
    Args:
        question_data: Question with choices and correct answer
        inverse_bias_dist: Inverse bias distribution from IP module
        similarity_calculator: Instance for calculating semantic similarity
        use_ss: Whether to use SS module (for ablation study)
        
    Returns:
        Dictionary with rearranged choices and metadata
    """
    choices = question_data['choices']
    correct_answer = question_data['answer']
    
    # Step 1: Apply IP - Sample position for correct answer
    from .ip_module import sample_answer_position
    correct_position = sample_answer_position(inverse_bias_dist)
    
    # Initialize position mapping
    position_mapping = {correct_position: correct_answer}
    
    # Get remaining positions
    all_positions = sorted(choices.keys())
    available_positions = [pos for pos in all_positions if pos != correct_position]
    
    # Get distractor labels
    distractor_labels = [label for label in choices.keys() if label != correct_answer]
    
    # Step 2: Apply SS if enabled
    if use_ss and similarity_calculator and len(distractor_labels) > 0:
        # Find SSD
        ssd_label, similarities = find_ssd(choices, correct_answer, similarity_calculator)
        
        if ssd_label and len(available_positions) > 0:
            # Calculate distance weights for SSD placement
            distance_weights = calculate_distance_weights(correct_position, available_positions)
            
            # Sample position for SSD based on distance weights
            ssd_position = sample_ssd_position(distance_weights)
            position_mapping[ssd_position] = ssd_label
            
            # Remove SSD from distractors and its position from available
            distractor_labels.remove(ssd_label)
            available_positions.remove(ssd_position)
    
    # Step 3: Randomly place remaining distractors
    np.random.shuffle(available_positions)
    
    for i, distractor_label in enumerate(distractor_labels):
        if i < len(available_positions):
            position_mapping[available_positions[i]] = distractor_label
    
    # Create rearranged choices
    rearranged_choices = {}
    for position in sorted(all_positions):
        if position in position_mapping:
            original_label = position_mapping[position]
            rearranged_choices[position] = choices[original_label]
    
    # Create result with metadata
    result = {
        'question': question_data['question'],
        'choices': rearranged_choices,
        'correct_position': correct_position,
        'position_mapping': position_mapping,  # Maps new position -> original label
        'original_correct_answer': correct_answer,
        'id': question_data.get('id', 'unknown')
    }
    
    # Add SSD info if SS was used
    if use_ss and similarity_calculator and 'ssd_label' in locals():
        result['ssd_info'] = {
            'ssd_label': ssd_label,
            'ssd_position': ssd_position if 'ssd_position' in locals() else None,
            'similarities': similarities if 'similarities' in locals() else {}
        }
    
    return result


# Utility function for creating complete SCOPE pipeline
def create_scope_processor(inverse_bias_dist: Dict[str, float], 
                          sentence_bert_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create a SCOPE processor with both IP and SS modules.
    
    Args:
        inverse_bias_dist: Inverse bias distribution from IP module
        sentence_bert_model: Model name for semantic similarity
        
    Returns:
        Function that applies SCOPE to a question
    """
    similarity_calculator = SemanticSimilarityCalculator(sentence_bert_model)
    
    def process_question(question_data: Dict, use_ss: bool = True) -> Dict:
        return place_choices_with_debiasing(
            question_data,
            inverse_bias_dist,
            similarity_calculator,
            use_ss=use_ss
        )
    
    return process_question


# Example usage and testing
if __name__ == "__main__":
    # Test semantic similarity calculation
    sim_calc = SemanticSimilarityCalculator()
    
    # Test with sample choices
    test_choices = {
        "A": "newspaper",
        "B": "magazine",
        "C": "book",
        "D": "television",
        "E": "radio"
    }
    
    correct_answer = "A"
    
    # Find SSD
    ssd_label, similarities = find_ssd(test_choices, correct_answer, sim_calc)
    print(f"Correct answer: {correct_answer} - {test_choices[correct_answer]}")
    print(f"SSD: {ssd_label} - {test_choices[ssd_label]}")
    print(f"All similarities: {similarities}")
    
    # Test distance weights
    weights = calculate_distance_weights("B", ["A", "C", "D", "E"])
    print(f"\nDistance weights from position B: {weights}")
    
    # Test full placement
    test_question = {
        "id": "test1",
        "question": "What is printed with ink and distributed daily?",
        "choices": test_choices,
        "answer": correct_answer
    }
    
    # Dummy inverse bias for testing
    inverse_bias = {"A": 0.15, "B": 0.20, "C": 0.25, "D": 0.20, "E": 0.20}
    
    result = place_choices_with_debiasing(test_question, inverse_bias, sim_calc)
    print(f"\nRearranged question:")
    print(f"Question: {result['question']}")
    print(f"Choices: {result['choices']}")
    print(f"Correct position: {result['correct_position']}")
    print(f"Position mapping: {result['position_mapping']}")
    if 'ssd_info' in result:
        print(f"SSD info: {result['ssd_info']}")