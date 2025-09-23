#!/usr/bin/env python3
"""
Plot training metrics from wandb logs or checkpoint data
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import glob

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TrainingMetricsPlotter:
    """Plot training metrics from checkpoints and logs"""
    
    def __init__(self, output_dir: str, save_dir: str = "training_plots"):
        """
        Initialize the plotter
        
        Args:
            output_dir: Directory containing model checkpoints
            save_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.metrics = {
            'step': [],
            'kl_loss': [],
            'reward_mean': [],
            'reward_std': [],
            'accuracy': [],
            'pass_at_k': [],
            'learning_rate': [],
            'loss': [],
            'cosine_similarity': [],
            'variance_diff': [],
            'positive_cosine_ratio': []
        }
        
        # Load trainer state if available
        self.trainer_states = self._load_trainer_states()
        
    def _load_trainer_states(self) -> Dict[int, Dict]:
        """Load trainer states from all checkpoints"""
        trainer_states = {}
        
        # Find all checkpoint directories
        checkpoint_dirs = sorted(glob.glob(str(self.output_dir / "checkpoint-*")))
        
        for checkpoint_dir in checkpoint_dirs:
            # Extract step number from directory name
            try:
                step = int(checkpoint_dir.split("-")[-1])
            except:
                continue
                
            # Load trainer state
            trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"
            if trainer_state_path.exists():
                with open(trainer_state_path, 'r') as f:
                    trainer_states[step] = json.load(f)
                    
        return trainer_states
    
    def extract_metrics_from_trainer_states(self):
        """Extract metrics from trainer states"""
        for step, state in self.trainer_states.items():
            # Get log history
            log_history = state.get('log_history', [])
            
            # Find the log entry for this step
            for log_entry in log_history:
                if log_entry.get('step') == step:
                    self.metrics['step'].append(step)
                    
                    # Extract various metrics
                    self.metrics['loss'].append(log_entry.get('loss', np.nan))
                    self.metrics['learning_rate'].append(log_entry.get('learning_rate', np.nan))
                    
                    # GRPO specific metrics
                    self.metrics['kl_loss'].append(log_entry.get('objective/kl', np.nan))
                    self.metrics['reward_mean'].append(log_entry.get('objective/scores', np.nan))
                    
                    # Custom metrics from our implementation
                    self.metrics['accuracy'].append(log_entry.get('train/sample_accuracy', np.nan))
                    self.metrics['pass_at_k'].append(log_entry.get('train/pass_at_k', np.nan))
                    
                    # Variance analysis metrics
                    self.metrics['cosine_similarity'].append(
                        log_entry.get('analysis/avg_cosine_similarity', np.nan)
                    )
                    self.metrics['positive_cosine_ratio'].append(
                        log_entry.get('analysis/positive_cosine_ratio', np.nan)
                    )
                    self.metrics['variance_diff'].append(
                        log_entry.get('analysis/avg_variance_diff', np.nan)
                    )
                    
                    break
    
    def load_wandb_data(self, wandb_dir: Optional[str] = None):
        """Load metrics from wandb logs if available"""
        if wandb_dir is None:
            wandb_dir = self.output_dir.parent / "wandb"
        
        wandb_dir = Path(wandb_dir)
        
        # Find the latest run directory
        run_dirs = sorted(glob.glob(str(wandb_dir / "run-*")))
        if not run_dirs:
            print("No wandb runs found")
            return
        
        latest_run = run_dirs[-1]
        
        # Try to load wandb summary
        summary_file = Path(latest_run) / "files" / "wandb-summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                print(f"Loaded wandb summary from {summary_file}")
                
        # Try to load wandb history
        history_file = Path(latest_run) / "files" / "wandb-history.jsonl"
        if history_file.exists():
            import jsonlines
            
            with jsonlines.open(history_file) as reader:
                for obj in reader:
                    step = obj.get('_step', obj.get('step'))
                    if step is not None:
                        self.metrics['step'].append(step)
                        
                        # Extract metrics
                        self.metrics['loss'].append(obj.get('loss', np.nan))
                        self.metrics['learning_rate'].append(obj.get('learning_rate', np.nan))
                        self.metrics['kl_loss'].append(obj.get('objective/kl', np.nan))
                        self.metrics['reward_mean'].append(obj.get('objective/scores', np.nan))
                        self.metrics['accuracy'].append(obj.get('train/sample_accuracy', np.nan))
                        self.metrics['pass_at_k'].append(obj.get('train/pass_at_k', np.nan))
                        
                        # Variance analysis metrics
                        self.metrics['cosine_similarity'].append(
                            obj.get('analysis/avg_cosine_similarity', np.nan)
                        )
                        self.metrics['positive_cosine_ratio'].append(
                            obj.get('analysis/positive_cosine_ratio', np.nan)
                        )
                        self.metrics['variance_diff'].append(
                            obj.get('analysis/avg_variance_diff', np.nan)
                        )
            
            print(f"Loaded {len(self.metrics['step'])} data points from wandb history")
    
    def plot_training_curves(self):
        """Create comprehensive training curve plots"""
        # Convert metrics to DataFrame for easier plotting
        df = pd.DataFrame(self.metrics)
        df = df.sort_values('step')
        df = df[df['step'] > 0]  # Remove step 0 if present
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Training Metrics Overview', fontsize=16, y=1.02)
        
        # 1. Loss curves
        ax = axes[0, 0]
        if not df['loss'].isna().all():
            ax.plot(df['step'], df['loss'], label='Total Loss', linewidth=2)
        if not df['kl_loss'].isna().all():
            ax.plot(df['step'], df['kl_loss'], label='KL Loss', linewidth=2, alpha=0.7)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Reward metrics
        ax = axes[0, 1]
        if not df['reward_mean'].isna().all():
            ax.plot(df['step'], df['reward_mean'], label='Mean Reward', linewidth=2)
            if not df['reward_std'].isna().all():
                ax.fill_between(df['step'], 
                               df['reward_mean'] - df['reward_std'],
                               df['reward_mean'] + df['reward_std'],
                               alpha=0.3, label='±1 std')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Accuracy metrics
        ax = axes[0, 2]
        if not df['accuracy'].isna().all():
            ax.plot(df['step'], df['accuracy'] * 100, label='Sample Accuracy', linewidth=2)
        if not df['pass_at_k'].isna().all():
            ax.plot(df['step'], df['pass_at_k'] * 100, label='Pass@K', linewidth=2, alpha=0.7)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Metrics')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Learning rate
        ax = axes[1, 0]
        if not df['learning_rate'].isna().all():
            ax.plot(df['step'], df['learning_rate'], linewidth=2, color='orange')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        # 5. KL Loss vs Reward (scatter)
        ax = axes[1, 1]
        valid_data = df[~df['kl_loss'].isna() & ~df['reward_mean'].isna()]
        if len(valid_data) > 0:
            scatter = ax.scatter(valid_data['kl_loss'], valid_data['reward_mean'], 
                               c=valid_data['step'], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Step')
        ax.set_xlabel('KL Loss')
        ax.set_ylabel('Mean Reward')
        ax.set_title('KL-Reward Trade-off')
        ax.grid(True, alpha=0.3)
        
        # 6. Cosine Similarity Analysis
        ax = axes[1, 2]
        if not df['cosine_similarity'].isna().all():
            ax.plot(df['step'], df['cosine_similarity'], label='Avg Cosine Sim', linewidth=2)
        if not df['positive_cosine_ratio'].isna().all():
            ax2 = ax.twinx()
            ax2.plot(df['step'], df['positive_cosine_ratio'] * 100, 
                    label='Positive Ratio (%)', linewidth=2, color='orange', alpha=0.7)
            ax2.set_ylabel('Positive Cosine Ratio (%)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Cosine Similarity', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_title('Cosine Similarity Evolution')
        ax.grid(True, alpha=0.3)
        
        # 7. Variance Difference
        ax = axes[2, 0]
        if not df['variance_diff'].isna().all():
            ax.plot(df['step'], df['variance_diff'], linewidth=2, color='purple')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Variance Difference')
        ax.set_title('Variance Difference Between Questions')
        ax.grid(True, alpha=0.3)
        
        # 8. Pass@K vs Accuracy correlation
        ax = axes[2, 1]
        valid_data = df[~df['pass_at_k'].isna() & ~df['accuracy'].isna()]
        if len(valid_data) > 0:
            ax.scatter(valid_data['accuracy'] * 100, valid_data['pass_at_k'] * 100,
                      c=valid_data['step'], cmap='plasma', alpha=0.6)
            # Add diagonal line
            ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)
        ax.set_xlabel('Sample Accuracy (%)')
        ax.set_ylabel('Pass@K (%)')
        ax.set_title('Pass@K vs Sample Accuracy')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        
        # 9. Summary statistics table
        ax = axes[2, 2]
        ax.axis('off')
        
        # Calculate summary statistics
        summary_text = "Summary Statistics\n" + "="*30 + "\n"
        
        if len(df) > 0:
            final_step = df['step'].max()
            summary_text += f"Total Steps: {final_step}\n"
            
            if not df['accuracy'].isna().all():
                final_acc = df[df['step'] == final_step]['accuracy'].values[0]
                summary_text += f"Final Accuracy: {final_acc*100:.2f}%\n"
                summary_text += f"Max Accuracy: {df['accuracy'].max()*100:.2f}%\n"
            
            if not df['pass_at_k'].isna().all():
                final_pass = df[df['step'] == final_step]['pass_at_k'].values[0]
                summary_text += f"Final Pass@K: {final_pass*100:.2f}%\n"
                summary_text += f"Max Pass@K: {df['pass_at_k'].max()*100:.2f}%\n"
            
            if not df['reward_mean'].isna().all():
                final_reward = df[df['step'] == final_step]['reward_mean'].values[0]
                summary_text += f"Final Reward: {final_reward:.4f}\n"
            
            if not df['kl_loss'].isna().all():
                final_kl = df[df['step'] == final_step]['kl_loss'].values[0]
                summary_text += f"Final KL Loss: {final_kl:.4f}\n"
            
            if not df['positive_cosine_ratio'].isna().all():
                pos_ratio = df['positive_cosine_ratio'].mean()
                summary_text += f"Avg Positive Cosine: {pos_ratio*100:.1f}%\n"
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='center',
               fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save the figure
        save_path = self.save_dir / "training_metrics_overview.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training metrics plot to {save_path}")
        
        # Also save as PDF for publication quality
        save_path_pdf = self.save_dir / "training_metrics_overview.pdf"
        plt.savefig(save_path_pdf, bbox_inches='tight')
        print(f"Saved training metrics plot (PDF) to {save_path_pdf}")
        
        plt.show()
        
        # Save metrics to CSV for further analysis
        csv_path = self.save_dir / "training_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved training metrics to {csv_path}")
        
        return df
    
    def plot_individual_metrics(self, df: pd.DataFrame):
        """Create individual plots for each metric"""
        
        # Create individual plots directory
        individual_dir = self.save_dir / "individual_plots"
        individual_dir.mkdir(exist_ok=True)
        
        # 1. KL Loss plot
        if not df['kl_loss'].isna().all():
            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['kl_loss'], linewidth=2, color='blue')
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('KL Loss', fontsize=12)
            plt.title('KL Divergence Loss Over Training', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.savefig(individual_dir / "kl_loss.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Reward plot
        if not df['reward_mean'].isna().all():
            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['reward_mean'], linewidth=2, color='green')
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Mean Reward', fontsize=12)
            plt.title('Mean Reward Over Training', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.savefig(individual_dir / "reward.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Accuracy plot
        if not df['accuracy'].isna().all():
            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['accuracy'] * 100, linewidth=2, color='red')
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title('Sample Accuracy Over Training', fontsize=14)
            plt.ylim([0, 100])
            plt.grid(True, alpha=0.3)
            plt.savefig(individual_dir / "accuracy.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Cosine similarity plot
        if not df['cosine_similarity'].isna().all():
            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['cosine_similarity'], linewidth=2, color='purple')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Average Cosine Similarity', fontsize=12)
            plt.title('Cosine Similarity Between Questions Over Training', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.savefig(individual_dir / "cosine_similarity.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Saved individual plots to {individual_dir}")

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from checkpoints")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--wandb_dir', type=str, default=None,
                       help='Directory containing wandb logs')
    parser.add_argument('--save_dir', type=str, default='training_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = TrainingMetricsPlotter(args.output_dir, args.save_dir)
    
    # Extract metrics from trainer states
    print("Extracting metrics from trainer states...")
    plotter.extract_metrics_from_trainer_states()
    
    # Load wandb data if available
    if args.wandb_dir or (Path(args.output_dir).parent / "wandb").exists():
        print("Loading wandb data...")
        plotter.load_wandb_data(args.wandb_dir)
    
    # Create plots
    print("Creating plots...")
    df = plotter.plot_training_curves()
    
    # Create individual plots
    plotter.plot_individual_metrics(df)
    
    print("Done!")

if __name__ == "__main__":
    main()
