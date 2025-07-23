#!/bin/bash
# scripts/download_data.sh

echo "======================================"
echo "SCOPE Data Download Instructions"
echo "======================================"
echo ""
echo "This framework uses fixed datasets that should be placed in data/fixed/"
echo ""
echo "Required files:"
echo "1. data/fixed/csqa_500_fixed.json - 500 CommonsenseQA questions"
echo "2. data/fixed/mmlu_500_fixed.json - 500 MMLU questions"
echo ""
echo "To create these files from raw data:"
echo ""
echo "1. Download CommonsenseQA:"
echo "   wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl"
echo ""
echo "2. Download MMLU from Hugging Face:"
echo "   https://huggingface.co/datasets/cais/mmlu"
echo ""
echo "3. Use the provided preprocessing script to create fixed datasets"
echo ""
echo "Note: The fixed datasets ensure reproducibility by using the same"
echo "      question samples across all experiments."
echo ""

# Create data directories if they don't exist
mkdir -p data/fixed
mkdir -p data/raw

echo "Data directories created:"
echo "- data/raw/ (for original datasets)"
echo "- data/fixed/ (for preprocessed datasets)"