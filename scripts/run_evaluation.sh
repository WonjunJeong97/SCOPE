#!/bin/bash
# scripts/run_evaluation.sh

# SCOPE Evaluation Runner Script

# Default values
MODEL="gpt-3.5-turbo"
DATASET="both"
MODE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -d|--dataset)
      DATASET="$2"
      shift 2
      ;;
    -t|--test)
      MODE="--test"
      shift
      ;;
    -a|--ablation)
      MODE="--ablation"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-m MODEL] [-d DATASET] [-t] [-a]"
      echo "  -m, --model     Model name (default: gpt-3.5-turbo)"
      echo "  -d, --dataset   Dataset: csqa, mmlu, or both (default: both)"
      echo "  -t, --test      Run in test mode"
      echo "  -a, --ablation  Run ablation study"
      exit 1
      ;;
  esac
done

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
  echo "Error: .env file not found!"
  echo "Please create .env from .env.example and add your API keys"
  exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Run evaluation
echo "======================================"
echo "Running SCOPE Evaluation"
echo "======================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Mode: ${MODE:-normal}"
echo ""

python src/main.py --model "$MODEL" --dataset "$DATASET" $MODE

echo ""
echo "Evaluation complete!"