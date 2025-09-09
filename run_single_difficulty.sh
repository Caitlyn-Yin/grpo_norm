#!/bin/bash

# configuration
MODEL="Qwen/Qwen2.5-Math-7B"
FORMAT="qa"
NUM_SHOTS=2
NUM_GENERATIONS=8
LEARNING_RATE=1e-5
KL_BETA=0.02
MAX_STEPS=400
WANDB_PROJECT="grpo-final-hard-difficulty"
DATA_DIR="data/gsm8k_difficulty_subsets"

# GPU allocation
GPU_STANDARD=4
GPU_NO_STD=5
GPU_BATCH_STD=6

# only one difficulty
DIFFICULTY="hard"  

mkdir -p logs/final_hard_difficulty
mkdir -p outputs/GRPO_final_hard_difficulty

echo "GRPO Final - $DIFFICULTY difficulty"


# Standard normalization
CUDA_VISIBLE_DEVICES=$GPU_STANDARD python train_difficulty.py \
    --difficulty $DIFFICULTY \
    --data_dir $DATA_DIR \
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
    --wandb_run_name "${DIFFICULTY}_standard_${MAX_STEPS}steps" \
    --exp_name "${DIFFICULTY}_standard" \
    > logs/final_hard_difficulty/${DIFFICULTY}_standard.log 2>&1 &
PID1=$!

# No std normalization  
CUDA_VISIBLE_DEVICES=$GPU_NO_STD python train_difficulty.py \
    --difficulty $DIFFICULTY \
    --data_dir $DATA_DIR \
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
    --wandb_run_name "${DIFFICULTY}_no_std_${MAX_STEPS}steps" \
    --exp_name "${DIFFICULTY}_no_std" \
    > logs/final_hard_difficulty/${DIFFICULTY}_no_std.log 2>&1 &
PID2=$!

# Batch std normalization
CUDA_VISIBLE_DEVICES=$GPU_BATCH_STD python train_difficulty.py \
    --difficulty $DIFFICULTY \
    --data_dir $DATA_DIR \
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
    --wandb_run_name "${DIFFICULTY}_batch_std_${MAX_STEPS}steps" \
    --exp_name "${DIFFICULTY}_batch_std" \
    > logs/final_hard_difficulty/${DIFFICULTY}_batch_std.log 2>&1 &
PID3=$!

echo "Started PIDs: $PID1, $PID2, $PID3"
wait
echo "All $DIFFICULTY experiments completed!"