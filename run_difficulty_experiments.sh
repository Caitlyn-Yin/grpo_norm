#!/bin/bash

# Activate conda environment
source /home/heqiy/miniconda3/etc/profile.d/conda.sh
conda activate grpo_env

# Configuration
MODEL="Qwen/Qwen2.5-Math-1.5B"
MAX_STEPS=800
DATA_DIR="data/gsm8k_difficulty_subsets"
DIFFICULTY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --difficulty)
            DIFFICULTY="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --difficulty <easy|medium|hard> [--max_steps <steps>]"
            echo ""
            echo "Options:"
            echo "  --difficulty    Choose difficulty level: easy, medium, or hard"
            echo "  --max_steps     Number of training steps (default: 1000)"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --difficulty easy"
            echo "  $0 --difficulty medium --max_steps 500"
            echo "  $0 --difficulty hard"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if difficulty is provided
if [ -z "$DIFFICULTY" ]; then
    echo "Error: --difficulty is required"
    echo "Usage: $0 --difficulty <easy|medium|hard>"
    echo "Use --help for more information"
    exit 1
fi

# Validate difficulty
if [[ "$DIFFICULTY" != "easy" && "$DIFFICULTY" != "medium" && "$DIFFICULTY" != "hard" ]]; then
    echo "Error: --difficulty must be one of: easy, medium, hard"
    exit 1
fi

# Set WANDB project name based on difficulty
if [[ "$DIFFICULTY" == "easy" ]]; then
    WANDB_PROJECT="iclr01_easy"
elif [[ "$DIFFICULTY" == "medium" ]]; then
    WANDB_PROJECT="iclr02_medium"
elif [[ "$DIFFICULTY" == "hard" ]]; then
    WANDB_PROJECT="iclr01_hard"
fi

echo "Creating directories..."
mkdir -p logs/iclr1_difficulty
mkdir -p outputs/iclr1_difficulty

echo "Starting 3 normalization experiments for $DIFFICULTY difficulty"
echo "Max steps: $MAX_STEPS"
echo "WANDB project: $WANDB_PROJECT"

# Create experiments array for the selected difficulty
experiments=(
    "$DIFFICULTY standard"
    "$DIFFICULTY no_std"
)

run_exp() {
    local diff=$1
    local norm=$2
    
    echo "Starting: $diff-$norm"
    
    python train_difficulty.py \
        --difficulty $diff \
        --data_dir $DATA_DIR \
        --normalization $norm \
        --max_steps $MAX_STEPS \
        --use_wandb \
        --wandb_project $WANDB_PROJECT \
        --exp_name "${diff}_${norm}" \
        > "logs/iclr1_difficulty/${diff}_${norm}.log" 2>&1
    
    echo "Completed: $diff-$norm"
}

if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for execution"
    export -f run_exp
    export MODEL MAX_STEPS WANDB_PROJECT DATA_DIR
    
    printf '%s\n' "${experiments[@]}" | parallel -j 3 --colsep ' ' run_exp {1} {2}
else
    echo "GNU parallel not found, using background jobs"
    
    max_jobs=3
    job_count=0
    
    for exp in "${experiments[@]}"; do
        read diff norm <<< "$exp"
        
        while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
            sleep 10
        done
        
        run_exp $diff $norm &
        
        ((job_count++))
        echo "Started $job_count/${#experiments[@]}: $diff-$norm"
        
        sleep 5
    done
    
    wait
fi

echo ""
echo "All experiments completed!"


echo ""
echo "Results Summary:"
echo "----------------"
for exp in "${experiments[@]}"; do
    read diff norm <<< "$exp"
    log="logs/iclr1_difficulty/${diff}_${norm}.log"
    if [ -f "$log" ]; then
        echo ""
        echo "[$diff - $norm]:"
        tail -10 "$log" | grep -E "Final|Pass@" || echo "  Check log file for details"
    fi
done

echo ""
echo "Check WandB project: $WANDB_PROJECT"
echo "Logs saved in: logs/iclr1_difficulty/"