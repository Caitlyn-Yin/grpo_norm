#!/usr/bin/env python3
"""
Script to monitor training progress and analyze the latest checkpoint.
Can be run while training is ongoing.
"""

import os
import argparse
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def get_latest_checkpoint(output_dir: Path) -> Path:
    """Get the latest checkpoint directory"""
    checkpoint_dirs = []
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoint_dirs.append((step, item))
            except:
                continue
    
    if not checkpoint_dirs:
        return None
    
    # Sort by step number and return the latest
    checkpoint_dirs.sort(key=lambda x: x[0])
    return checkpoint_dirs[-1][1]


def read_trainer_state(checkpoint_dir: Path) -> dict:
    """Read the trainer state from a checkpoint"""
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path, 'r') as f:
            return json.load(f)
    return None


def plot_training_metrics(output_dir: Path):
    """Plot training metrics from all available checkpoints"""
    checkpoint_dirs = []
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoint_dirs.append((step, item))
            except:
                continue
    
    if not checkpoint_dirs:
        print("No checkpoints found yet")
        return
    
    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: x[0])
    
    # Collect metrics
    steps = []
    losses = []
    learning_rates = []
    
    for step, checkpoint_dir in checkpoint_dirs:
        state = read_trainer_state(checkpoint_dir)
        if state and 'log_history' in state:
            # Get the last logged loss for this checkpoint
            for log_entry in reversed(state['log_history']):
                if 'loss' in log_entry:
                    steps.append(step)
                    losses.append(log_entry['loss'])
                    if 'learning_rate' in log_entry:
                        learning_rates.append(log_entry['learning_rate'])
                    break
    
    if not steps:
        print("No training metrics found in checkpoints")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Loss plot
    axes[0].plot(steps, losses, 'b-', marker='o')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate plot
    if learning_rates:
        axes[1].plot(steps[:len(learning_rates)], learning_rates, 'r-', marker='o')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "training_progress.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training progress plot saved to {plot_path}")
    print(f"Latest step: {steps[-1]}, Latest loss: {losses[-1]:.4f}")
    
    return steps, losses


def monitor_loop(output_dir: Path, interval: int = 30):
    """Monitor training progress in a loop"""
    print(f"Monitoring training in {output_dir}")
    print(f"Checking every {interval} seconds. Press Ctrl+C to stop.")
    
    last_checkpoint = None
    
    while True:
        try:
            latest_checkpoint = get_latest_checkpoint(output_dir)
            
            if latest_checkpoint and latest_checkpoint != last_checkpoint:
                print(f"\nNew checkpoint found: {latest_checkpoint.name}")
                
                # Read trainer state
                state = read_trainer_state(latest_checkpoint)
                if state:
                    current_step = state.get('global_step', 0)
                    best_metric = state.get('best_metric', None)
                    
                    print(f"  Step: {current_step}")
                    if best_metric:
                        print(f"  Best metric: {best_metric:.4f}")
                    
                    # Get latest metrics from log history
                    if 'log_history' in state and state['log_history']:
                        latest_log = state['log_history'][-1]
                        if 'loss' in latest_log:
                            print(f"  Latest loss: {latest_log['loss']:.4f}")
                        if 'learning_rate' in latest_log:
                            print(f"  Learning rate: {latest_log['learning_rate']:.2e}")
                
                # Update plots
                plot_training_metrics(output_dir)
                
                last_checkpoint = latest_checkpoint
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to the training output directory')
    parser.add_argument('--interval', type=int, default=30,
                       help='Check interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true',
                       help='Run once instead of monitoring continuously')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist")
        return
    
    if args.once:
        # Run once and exit
        latest_checkpoint = get_latest_checkpoint(output_dir)
        if latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint.name}")
            state = read_trainer_state(latest_checkpoint)
            if state:
                print(f"Step: {state.get('global_step', 0)}")
        
        plot_training_metrics(output_dir)
    else:
        # Monitor continuously
        monitor_loop(output_dir, args.interval)


if __name__ == "__main__":
    main()
