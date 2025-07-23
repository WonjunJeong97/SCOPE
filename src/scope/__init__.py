# src/scope/__init__.py
"""
SCOPE modules for position bias and semantic similarity handling.
"""

from .ip_module import (
    generate_null_prompts,
    measure_position_bias,
    calculate_inverse_bias_distribution,
    sample_answer_position,
    apply_inverse_positioning
)

from .ss_module import (
    SemanticSimilarityCalculator,
    find_ssd,
    calculate_distance_weights,
    sample_ssd_position,
    place_choices_with_debiasing,
    create_scope_processor
)

__all__ = [
    'generate_null_prompts',
    'measure_position_bias',
    'calculate_inverse_bias_distribution',
    'sample_answer_position',
    'apply_inverse_positioning',
    'SemanticSimilarityCalculator',
    'find_ssd',
    'calculate_distance_weights',
    'sample_ssd_position',
    'place_choices_with_debiasing',
    'create_scope_processor'
]