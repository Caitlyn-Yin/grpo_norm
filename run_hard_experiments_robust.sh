#!/bin/bash

# Activate conda environment
source /home/heqiy/miniconda3/etc/profile.d/conda.sh
conda activate grpo_env

echo "Starting HARD experiments from scratch (600 steps)..."
echo "====================================="

# Configuration
MODEL="Qwen/Qwen2.5-Math-1.5B"
MAX_STEPS=600
DATA_DIR="data/gsm8k_difficulty_subsets"

echo "Model: $MODEL"
echo "Max steps: $MAX_STEPS"
echo "Data directory: $DATA_DIR"
echo ""

# Clean up any existing processes
echo "Cleaning up any existing processes..."
pkill -f train_difficulty 2>/dev/null || true
sleep 2

# Create fresh output directories
echo "Creating fresh output directories..."
rm -rf outputs/iclr1_difficulty/hard/
mkdir -p outputs/iclr1_difficulty/hard/standard/Qwen2.5-Math-1.5B
mkdir -p outputs/iclr1_difficulty/hard/no_std/Qwen2.5-Math-1.5B
mkdir -p logs/iclr1_difficulty

echo "Starting HARD experiments from scratch..."
echo "WANDB project: iclr1_hard"
echo "WANDB run names:"
echo "  - grpo-iclr1_hard-standard-hard-hard_standard"
echo "  - grpo-iclr1_hard-no_std-hard-hard_no_std"
echo ""

# Function to run experiments with enhanced robustness
run_hard_exp_robust() {
    local norm=$1
    
    echo "Starting: hard-$norm (from scratch) - using nohup with enhanced monitoring"
    
    # Use unique WandB run names for fresh start
    local wandb_run_name="grpo-iclr1_hard-${norm}-hard-hard_${norm}"
    
    # Create a wrapper script to ensure completion
    cat > "logs/iclr1_difficulty/run_${norm}_wrapper.sh" << EOF
#!/bin/bash
source /home/heqiy/miniconda3/etc/profile.d/conda.sh
conda activate grpo_env

# Set up signal handling
trap 'echo "Received signal, but continuing training..."' INT TERM

# Run training with retry logic
max_retries=3
retry_count=0

while [ \$retry_count -lt \$max_retries ]; do
    echo "Attempt \$((retry_count + 1)) of \$max_retries for hard-$norm"
    
    python train_difficulty.py \\
        --difficulty hard \\
        --data_dir $DATA_DIR \\
        --normalization $norm \\
        --max_steps $MAX_STEPS \\
        --model_name $MODEL \\
        --use_wandb \\
        --wandb_project iclr1_hard \\
        --wandb_run_name "$wandb_run_name" \\
        --exp_name "hard_${norm}"
    
    exit_code=\$?
    
    if [ \$exit_code -eq 0 ]; then
        echo "Training completed successfully for hard-$norm"
        break
    else
        echo "Training failed with exit code \$exit_code for hard-$norm"
        retry_count=\$((retry_count + 1))
        if [ \$retry_count -lt \$max_retries ]; then
            echo "Retrying in 10 seconds..."
            sleep 10
        fi
    fi
done

if [ \$retry_count -eq \$max_retries ]; then
    echo "Training failed after \$max_retries attempts for hard-$norm"
    exit 1
fi
EOF

    chmod +x "logs/iclr1_difficulty/run_${norm}_wrapper.sh"
    
    # Run with nohup and enhanced monitoring
    nohup "logs/iclr1_difficulty/run_${norm}_wrapper.sh" \
        > "logs/iclr1_difficulty/hard_${norm}_robust.log" 2>&1 &
    
    local pid=$!
    echo "Started hard-$norm with PID: $pid"
    echo "$pid" > "logs/iclr01_difficulty/hard_${norm}_pid.txt"
    
    # Start a monitoring process
    (
        while kill -0 $pid 2>/dev/null; do
            sleep 30
            if ! kill -0 $pid 2>/dev/null; then
                echo "Process $pid died, checking if training completed..."
                if [ -f "outputs/iclr1_difficulty/hard/$norm/Qwen2.5-Math-1.5B/hard_$norm/checkpoint-600" ]; then
                    echo "Training completed successfully for hard-$norm"
                else
                    echo "Training did not complete for hard-$norm"
                fi
                break
            fi
        done
    ) &
    
    return $pid
}

# Run hard experiments in background with enhanced monitoring
run_hard_exp_robust "standard"
run_hard_exp_robust "no_std"

echo ""
echo "Both experiments started in background with enhanced monitoring"
echo "You can safely disconnect from the terminal"
echo ""
echo "To monitor progress:"
echo "  tail -f logs/iclr01_difficulty/hard_standard_robust.log"
echo "  tail -f logs/iclr01_difficulty/hard_no_std_robust.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep train_difficulty"
echo "  ps aux | grep run_.*_wrapper"
echo ""
echo "To check progress:"
echo "  find outputs/iclr01_difficulty/hard/ -name 'checkpoint-*' | sort -V | tail -5"
echo ""
echo "To stop experiments:"
echo "  pkill -f train_difficulty"
echo "  pkill -f run_.*_wrapper"
echo ""
echo "Check WandB project: iclr01_hard"
echo "Outputs saved in: outputs/iclr01_difficulty/hard/"
echo "Logs saved in: logs/iclr01_difficulty/"
echo ""
echo "The experiments will run until completion (600 steps) with automatic retry on failure."