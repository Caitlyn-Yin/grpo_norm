#!/bin/bash
MODEL="Qwen/Qwen2.5-Math-1.5B"
MAX_STEPS=400
WANDB_PROJECT="grpo-final3-difficulty"
DATA_DIR="data/gsm8k_difficulty_subsets"

echo "Creating directories..."
mkdir -p logs/final3_difficulty
mkdir -p logs/final3_difficulty
mkdir -p outputs/GRPO_final3_difficulty

echo "Starting 9 experiments with auto GPU allocation"
echo "==========================="

experiments=(
    "easy standard"
    "easy no_std"
    "easy batch_std"
    "medium standard"
    "medium no_std"
    "medium batch_std"
    "hard standard"
    "hard no_std"
    "hard batch_std"
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
        > "logs/final3_difficulty/${diff}_${norm}.log" 2>&1
    
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
    log="logs/final3_difficulty/${diff}_${norm}.log"
    if [ -f "$log" ]; then
        echo ""
        echo "[$diff - $norm]:"
        tail -10 "$log" | grep -E "Final|Pass@" || echo "  Check log file for details"
    fi
done

echo ""
echo "Check WandB project: $WANDB_PROJECT"
echo "Logs saved in: logs/final3_difficulty/"
echo "==========================="