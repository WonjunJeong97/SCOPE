# README.md

# SCOPE: STOCHASTIC AND COUNTERBIASED OPTION PLACEMENT FOR EVALUATING LARGE LANGUAGE MODELS

Official implementation of the SCOPE framework for mitigating selection biases in Large Language Model (LLM) multiple-choice evaluations.

## Overview

SCOPE addresses two selection biases in LLM evaluations:
1. Position Bias: LLMs tend to favor certain positions (e.g., first or last options)
2. Semantic Bias: Models may select semantically similar distractors when uncertain

The framework consists of two modules:
- Inverse-Positioning (IP): Measures position bias and places answers in less-preferred positions
- Semantic-Spread (SS): Identifies and spatially separates semantically similar distractors from correct answers

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/WonjunJeong97/SCOPE-repo
cd scope-repo