#!/bin/bash

# Create a new conda environment for GRPO
echo "Creating new conda environment for GRPO..."
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n grpo_env python=3.10 -y
conda activate grpo_env

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with compatible versions (2.1.0 or newer)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related libraries with compatible versions
pip install transformers==4.36.0 peft==0.6.0

# Install TRL from source to get GRPO
pip install git+https://github.com/huggingface/trl.git@main

# Install other dependencies
pip install numpy pandas matplotlib scipy wandb accelerate bitsandbytes

# Install flash-attn if available (optional, for faster training)
pip install flash-attn --no-build-isolation || echo "Flash attention not installed, continuing without it"

echo "Environment setup complete!"
echo "=================================="
echo "To use this environment, run: conda activate grpo_env"
echo "Then run your training script: python train_difficulty.py --difficulty easy --normalization standard --num_generations 8 --max_steps 50 --exp_name test_run"
echo "=================================="
