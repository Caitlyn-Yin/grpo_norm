#!/bin/bash
MODEL="Qwen/Qwen2.5-Math-1.5B"
FORMAT="qa"
NUM_SHOTS=2
NUM_GENERATIONS=8
LEARNING_RATE=1e-5
KL_BETA=0.02
MAX_STEPS=600  # 完整训练步数
WANDB_PROJECT="grpo-normalization-comparison"

GPU_STANDARD=4
GPU_NO_STD=5
GPU_BATCH_STD=6

mkdir -p logs
mkdir -p outputs

echo "======================================"
echo "GRPO Normalization Comparison"
echo "======================================"
echo "Model: $MODEL"
echo "Training steps: $MAX_STEPS"
echo "K (num_generations): $NUM_GENERATIONS"
echo "GPUs: $GPU_STANDARD, $GPU_NO_STD, $GPU_BATCH_STD"
echo "======================================"

echo "Starting all three experiments in parallel..."

CUDA_VISIBLE_DEVICES=$GPU_STANDARD python train.py \
    --format $FORMAT \
    --model_name $MODEL \
    --num_shots $NUM_SHOTS \
    --num_generations $NUM_GENERATIONS \
    --learning_rate $LEARNING_RATE \
    --kl_beta $KL_BETA \
    --max_steps $MAX_STEPS \
    --normalization standard \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "standard_norm_${MAX_STEPS}steps" \
    --exp_name "standard_norm" \
    > logs/standard_norm.log 2>&1 &
PID1=$!
echo "✓ Started Standard normalization (PID: $PID1)"

CUDA_VISIBLE_DEVICES=$GPU_NO_STD python train.py \
    --format $FORMAT \
    --model_name $MODEL \
    --num_shots $NUM_SHOTS \
    --num_generations $NUM_GENERATIONS \
    --learning_rate $LEARNING_RATE \
    --kl_beta $KL_BETA \
    --max_steps $MAX_STEPS \
    --normalization no_std \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "no_std_norm_${MAX_STEPS}steps" \
    --exp_name "no_std_norm" \
    > logs/no_std_norm.log 2>&1 &
PID2=$!
echo "✓ Started No-std normalization (PID: $PID2)"

CUDA_VISIBLE_DEVICES=$GPU_BATCH_STD python train.py \
    --format $FORMAT \
    --model_name $MODEL \
    --num_shots $NUM_SHOTS \
    --num_generations $NUM_GENERATIONS \
    --learning_rate $LEARNING_RATE \
    --kl_beta $KL_BETA \
    --max_steps $MAX_STEPS \
    --normalization batch_std \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "batch_std_norm_${MAX_STEPS}steps" \
    --exp_name "batch_std_norm" \
    > logs/batch_std_norm.log 2>&1 &
PID3=$!
echo "✓ Started Batch-std normalization (PID: $PID3)"


echo "All experiments started!"
echo "Process IDs: $PID1, $PID2, $PID3"
echo "Logs are being written to ./logs/"
echo "==========================="


echo "Monitoring experiments..."
while true; do

    if ! kill -0 $PID1 2>/dev/null && ! kill -0 $PID2 2>/dev/null && ! kill -0 $PID3 2>/dev/null; then
        echo "All experiments completed!"
        break
    fi
    

    echo -n "Progress: "
    for log_file in logs/*.log; do
        if [ -f "$log_file" ]; then
            progress=$(grep -o "[0-9]*%" "$log_file" | tail -1)
            name=$(basename "$log_file" .log)
            echo -n "$name:$progress "
        fi
    done
    echo ""
    
    sleep 60  
done

echo "All experiments finished!"
echo "Check WandB project: $WANDB_PROJECT"
echo "Logs saved in ./logs/"
echo "Models saved in ./outputs/"
echo "==========================="

for log_file in logs/*.log; do
    echo ""
    echo "Results from $(basename $log_file):"
    tail -20 "$log_file" | grep -E "Final Pass@|Final Sample Accuracy|FINAL"
done