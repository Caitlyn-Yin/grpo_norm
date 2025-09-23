#!/bin/bash

# Activate conda environment
source /home/heqiy/miniconda3/etc/profile.d/conda.sh
conda activate grpo_env

echo "Starting MEDIUM experiments..."
echo "====================================="

# Configuration
MODEL="Qwen/Qwen2.5-Math-1.5B"
MAX_STEPS=600
DATA_DIR="data/gsm8k_difficulty_subsets"

echo "Model: $MODEL"
echo "Max steps: $MAX_STEPS"
echo "Data directory: $DATA_DIR"
echo ""

# Create output directories
echo "Creating directories..."
mkdir -p outputs/iclr1_difficulty/medium/standard/Qwen2.5-Math-1.5B
mkdir -p outputs/iclr1_difficulty/medium/no_std/Qwen2.5-Math-1.5B
mkdir -p logs/iclr1_difficulty

echo "Starting MEDIUM experiments..."
echo "WANDB project: iclr02_medium"
echo ""

# Run medium experiments
run_medium_exp() {
    local norm=$1
    
    echo "Starting: medium-$norm"
    
    python train_difficulty.py \
        --difficulty medium \
        --data_dir $DATA_DIR \
        --normalization $norm \
        --max_steps $MAX_STEPS \
        --model_name $MODEL \
        --use_wandb \
        --wandb_project iclr02_medium \
        --exp_name "medium_${norm}" \
        > "logs/iclr1_difficulty/medium_${norm}.log" 2>&1
    
    echo "Completed: medium-$norm"
}

# Run medium experiments in parallel
run_medium_exp "standard" &
run_medium_exp "no_std" &

# Wait for medium experiments to complete
wait

echo "MEDIUM experiments completed!"
echo ""

echo "====================================="
echo "All MEDIUM experiments completed!"
echo ""
echo "Summary:"
echo "- Medium experiments: iclr02_medium"
echo "- Max steps: $MAX_STEPS"
echo "- Model: $MODEL"
echo ""
echo "Check WandB project: iclr02_medium"
echo "Outputs saved in: outputs/iclr1_difficulty/medium/"
echo "Logs saved in: logs/iclr1_difficulty/"
