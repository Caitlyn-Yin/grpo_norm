#!/bin/bash

echo "Creating new conda environment for GRPO..."
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n grpo_env python=3.10 -y
conda activate grpo_env
pip install --upgrade pip
pip install seaborn
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

pip install transformers==4.36.0 peft==0.6.0

pip install git+https://github.com/huggingface/trl.git@main

pip install numpy pandas matplotlib scipy wandb accelerate bitsandbytes

pip install flash-attn --no-build-isolation || echo "Flash attention not installed, continuing without it"

echo "Environment setup complete!"
echo "=================================="
echo "To use this environment, run: conda activate grpo_env"
echo "Then run your training script: python train_difficulty.py --difficulty easy --normalization standard --num_generations 8 --max_steps 50 --exp_name test_run"
echo "=================================="
