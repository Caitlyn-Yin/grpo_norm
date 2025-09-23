#!/usr/bin/env python3
"""
Monitor training progress in real-time
"""

import os
import time
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob

class TrainingMonitor:
    """Monitor training progress in real-time"""
    
    def __init__(self, output_dir: str, update_interval: int = 30):
        """
        Initialize the monitor
        
        Args:
            output_dir: Directory containing model checkpoints
            update_interval: Update interval in seconds
        """
        self.output_dir = Path(output_dir)
        self.update_interval = update_interval
        self.last_step = 0
        
        # Initialize plot
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Progress Monitor', fontsize=16)
        
    def find_latest_checkpoint(self):
        """Find the latest checkpoint directory"""
        checkpoint_dirs = glob.glob(str(self.output_dir / "checkpoint-*"))
        if not checkpoint_dirs:
            return None
            
        # Sort by step number
        checkpoints = []
        for checkpoint_dir in checkpoint_dirs:
            try:
                step = int(checkpoint_dir.split("-")[-1])
                checkpoints.append((step, checkpoint_dir))
            except:
                continue
        
        if not checkpoints:
            return None
            
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]
    
    def load_trainer_state(self, checkpoint_dir):
        """Load trainer state from checkpoint"""
        trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                return json.load(f)
        return None
    
    def extract_metrics(self, trainer_state):
        """Extract metrics from trainer state"""
        metrics = {
            'steps': [],
            'loss': [],
            'kl_loss': [],
            'reward': [],
            'accuracy': [],
            'pass_at_k': [],
            'learning_rate': []
        }
        
        log_history = trainer_state.get('log_history', [])
        
        for entry in log_history:
            step = entry.get('step')
            if step is not None and step > 0:
                metrics['steps'].append(step)
                metrics['loss'].append(entry.get('loss', np.nan))
                metrics['kl_loss'].append(entry.get('objective/kl', np.nan))
                metrics['reward'].append(entry.get('objective/scores', np.nan))
                metrics['accuracy'].append(entry.get('train/sample_accuracy', np.nan))
                metrics['pass_at_k'].append(entry.get('train/pass_at_k', np.nan))
                metrics['learning_rate'].append(entry.get('learning_rate', np.nan))
        
        return metrics
    
    def update_plots(self, metrics):
        """Update the plots with new metrics"""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Loss curves
        ax = self.axes[0, 0]
        if metrics['loss']:
            ax.plot(metrics['steps'], metrics['loss'], 'b-', label='Total Loss', linewidth=2)
        if metrics['kl_loss']:
            ax.plot(metrics['steps'], metrics['kl_loss'], 'r-', label='KL Loss', linewidth=2, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Reward
        ax = self.axes[0, 1]
        if metrics['reward']:
            ax.plot(metrics['steps'], metrics['reward'], 'g-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Reward Evolution')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy metrics
        ax = self.axes[1, 0]
        if metrics['accuracy']:
            valid_acc = [(s, a*100) for s, a in zip(metrics['steps'], metrics['accuracy']) if not np.isnan(a)]
            if valid_acc:
                steps, accs = zip(*valid_acc)
                ax.plot(steps, accs, 'b-', label='Sample Accuracy', linewidth=2)
        if metrics['pass_at_k']:
            valid_pass = [(s, p*100) for s, p in zip(metrics['steps'], metrics['pass_at_k']) if not np.isnan(p)]
            if valid_pass:
                steps, passes = zip(*valid_pass)
                ax.plot(steps, passes, 'r-', label='Pass@K', linewidth=2, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Metrics')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax = self.axes[1, 1]
        ax.axis('off')
        
        # Calculate current statistics
        summary_text = f"Live Training Monitor\n"
        summary_text += f"{'='*30}\n"
        summary_text += f"Last Update: {datetime.now().strftime('%H:%M:%S')}\n\n"
        
        if metrics['steps']:
            current_step = metrics['steps'][-1]
            summary_text += f"Current Step: {current_step}\n"
            
            if metrics['loss'] and not np.isnan(metrics['loss'][-1]):
                summary_text += f"Current Loss: {metrics['loss'][-1]:.4f}\n"
            
            if metrics['kl_loss'] and not np.isnan(metrics['kl_loss'][-1]):
                summary_text += f"Current KL Loss: {metrics['kl_loss'][-1]:.4f}\n"
            
            if metrics['reward'] and not np.isnan(metrics['reward'][-1]):
                summary_text += f"Current Reward: {metrics['reward'][-1]:.4f}\n"
            
            if metrics['accuracy']:
                valid_acc = [a for a in metrics['accuracy'] if not np.isnan(a)]
                if valid_acc:
                    summary_text += f"Current Accuracy: {valid_acc[-1]*100:.2f}%\n"
                    summary_text += f"Max Accuracy: {max(valid_acc)*100:.2f}%\n"
            
            if metrics['pass_at_k']:
                valid_pass = [p for p in metrics['pass_at_k'] if not np.isnan(p)]
                if valid_pass:
                    summary_text += f"Current Pass@K: {valid_pass[-1]*100:.2f}%\n"
                    summary_text += f"Max Pass@K: {max(valid_pass)*100:.2f}%\n"
            
            if metrics['learning_rate'] and not np.isnan(metrics['learning_rate'][-1]):
                summary_text += f"Learning Rate: {metrics['learning_rate'][-1]:.2e}\n"
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               fontfamily='monospace')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def monitor(self):
        """Main monitoring loop"""
        print(f"Monitoring training in {self.output_dir}")
        print(f"Update interval: {self.update_interval} seconds")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Find latest checkpoint
                latest_checkpoint = self.find_latest_checkpoint()
                
                if latest_checkpoint:
                    # Load trainer state
                    trainer_state = self.load_trainer_state(latest_checkpoint)
                    
                    if trainer_state:
                        # Extract metrics
                        metrics = self.extract_metrics(trainer_state)
                        
                        # Update plots
                        if metrics['steps']:
                            current_step = metrics['steps'][-1]
                            if current_step != self.last_step:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Updated: Step {current_step}")
                                self.last_step = current_step
                                self.update_plots(metrics)
                        
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No checkpoints found yet...")
                
                # Wait for next update
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            plt.ioff()
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Monitor training progress in real-time")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--update_interval', type=int, default=30,
                       help='Update interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = TrainingMonitor(args.output_dir, args.update_interval)
    
    # Start monitoring
    monitor.monitor()

if __name__ == "__main__":
    main()
