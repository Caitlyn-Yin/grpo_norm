#!/usr/bin/env python3
"""
Comprehensive analysis of all training experiments
Calculates variance differences, cosine similarities, and training metrics
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import glob
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ExperimentAnalyzer:
    """Analyze all experiments and compute metrics"""
    
    def __init__(self, base_dir: str = "outputs/GRPO_final3_difficulty"):
        self.base_dir = Path(base_dir)
        self.results = {}
        self.metrics_data = []
        
    def find_all_experiments(self):
        """Find all experiment directories"""
        experiments = []
        
        for difficulty in ['easy', 'medium', 'hard']:
            for normalization in ['standard', 'no_std']:
                exp_dir = self.base_dir / difficulty / normalization / "Qwen2.5-Math-7B" / f"{difficulty}_{normalization}"
                if exp_dir.exists():
                    experiments.append({
                        'difficulty': difficulty,
                        'normalization': normalization,
                        'path': exp_dir
                    })
                    print(f"Found experiment: {difficulty}-{normalization}")
        
        return experiments
    
    def load_trainer_state(self, checkpoint_path: Path) -> Dict:
        """Load trainer state from checkpoint"""
        trainer_state_path = checkpoint_path / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                return json.load(f)
        return None
    
    def extract_metrics_from_experiment(self, exp_path: Path) -> pd.DataFrame:
        """Extract all metrics from an experiment"""
        metrics = []
        
        # Find all checkpoints
        checkpoints = sorted(glob.glob(str(exp_path / "checkpoint-*")))
        
        for checkpoint_dir in checkpoints:
            checkpoint_path = Path(checkpoint_dir)
            step = int(checkpoint_dir.split("-")[-1])
            
            # Load trainer state
            trainer_state = self.load_trainer_state(checkpoint_path)
            
            if trainer_state:
                # Find metrics for this step
                log_history = trainer_state.get('log_history', [])
                
                # Get the last entry in log_history for this checkpoint
                # (sometimes there are multiple entries per checkpoint)
                for entry in reversed(log_history):
                    if entry.get('step') == step or (len(log_history) > 0 and entry == log_history[-1]):
                        metric_dict = {
                            'step': step,
                            'loss': entry.get('loss', np.nan),
                            'kl_loss': entry.get('kl', np.nan),  # Changed from 'objective/kl' to 'kl'
                            'reward_mean': entry.get('reward', np.nan),  # Changed from 'objective/scores' to 'reward'
                            'reward_std': entry.get('reward_std', np.nan),  # Changed key
                            'learning_rate': entry.get('learning_rate', np.nan),
                            'entropy': entry.get('entropy', np.nan),
                            'grad_norm': entry.get('grad_norm', np.nan),
                        }
                        
                        # Extract accuracy from rewards subdictionaries if available
                        correctness_mean = entry.get('rewards/correctness_reward_func_qa/mean', np.nan)
                        if not np.isnan(correctness_mean):
                            metric_dict['accuracy'] = correctness_mean
                        else:
                            metric_dict['accuracy'] = np.nan
                            
                        metrics.append(metric_dict)
                        break
        
        return pd.DataFrame(metrics)
    
    def calculate_variance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate variance-related metrics"""
        
        # Variance of rewards across training
        reward_variance = df['reward_mean'].var() if 'reward_mean' in df else np.nan
        
        # Variance of accuracy across training
        accuracy_variance = df['accuracy'].var() if 'accuracy' in df else np.nan
        
        # Variance differences between consecutive steps
        if 'reward_mean' in df and len(df) > 1:
            reward_diffs = df['reward_mean'].diff().dropna()
            variance_between_steps = reward_diffs.var()
        else:
            variance_between_steps = np.nan
        
        return {
            'reward_variance': reward_variance,
            'accuracy_variance': accuracy_variance,
            'variance_between_steps': variance_between_steps
        }
    
    def analyze_convergence(self, df: pd.DataFrame) -> Dict:
        """Analyze convergence behavior"""
        if len(df) < 10:
            return {}
        
        # Take last 20% of training for stability analysis
        last_portion = int(len(df) * 0.2)
        if last_portion < 5:
            last_portion = min(5, len(df))
        
        late_stage = df.tail(last_portion)
        
        # Calculate stability metrics
        metrics = {}
        
        if 'reward_mean' in df:
            metrics['final_reward'] = df['reward_mean'].iloc[-1]
            metrics['late_stage_reward_std'] = late_stage['reward_mean'].std()
            metrics['reward_improvement'] = df['reward_mean'].iloc[-1] - df['reward_mean'].iloc[0]
        
        if 'accuracy' in df:
            metrics['final_accuracy'] = df['accuracy'].iloc[-1] if not pd.isna(df['accuracy'].iloc[-1]) else np.nan
            metrics['max_accuracy'] = df['accuracy'].max()
            metrics['accuracy_improvement'] = (
                df['accuracy'].iloc[-1] - df['accuracy'].iloc[0] 
                if not pd.isna(df['accuracy'].iloc[-1]) and not pd.isna(df['accuracy'].iloc[0])
                else np.nan
            )
        
        if 'kl_loss' in df:
            metrics['final_kl'] = df['kl_loss'].iloc[-1]
            metrics['kl_change'] = df['kl_loss'].iloc[-1] - df['kl_loss'].iloc[0]
        
        return metrics
    
    def run_comprehensive_analysis(self):
        """Run analysis on all experiments"""
        experiments = self.find_all_experiments()
        
        all_results = []
        
        for exp in experiments:
            print(f"\nAnalyzing {exp['difficulty']}-{exp['normalization']}...")
            
            # Extract metrics
            df = self.extract_metrics_from_experiment(exp['path'])
            
            if len(df) == 0:
                print(f"  No data found for {exp['difficulty']}-{exp['normalization']}")
                continue
            
            # Calculate various metrics
            variance_metrics = self.calculate_variance_metrics(df)
            convergence_metrics = self.analyze_convergence(df)
            
            # Combine results
            result = {
                'difficulty': exp['difficulty'],
                'normalization': exp['normalization'],
                'num_steps': len(df),
                **variance_metrics,
                **convergence_metrics
            }
            
            all_results.append(result)
            
            # Store dataframe for plotting
            df['difficulty'] = exp['difficulty']
            df['normalization'] = exp['normalization']
            self.metrics_data.append(df)
        
        # Create summary DataFrame
        self.summary_df = pd.DataFrame(all_results)
        
        return self.summary_df
    
    def plot_comparison_grid(self):
        """Create comparison plots for all experiments"""
        if not self.metrics_data:
            print("No data to plot")
            return
        
        # Combine all data
        combined_df = pd.concat(self.metrics_data, ignore_index=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define plot grid
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)
        
        # 1. KL Loss comparison
        ax1 = fig.add_subplot(gs[0, :])
        for (diff, norm), group in combined_df.groupby(['difficulty', 'normalization']):
            label = f"{diff}-{norm}"
            ax1.plot(group['step'], group['kl_loss'], label=label, linewidth=2, alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('KL Loss')
        ax1.set_title('KL Divergence Loss Across All Experiments')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward comparison
        ax2 = fig.add_subplot(gs[1, :])
        for (diff, norm), group in combined_df.groupby(['difficulty', 'normalization']):
            label = f"{diff}-{norm}"
            ax2.plot(group['step'], group['reward_mean'], label=label, linewidth=2, alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Mean Reward')
        ax2.set_title('Mean Reward Across All Experiments')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy comparison
        ax3 = fig.add_subplot(gs[2, :])
        for (diff, norm), group in combined_df.groupby(['difficulty', 'normalization']):
            valid_acc = group.dropna(subset=['accuracy'])
            if len(valid_acc) > 0:
                label = f"{diff}-{norm}"
                ax3.plot(valid_acc['step'], valid_acc['accuracy'] * 100, 
                        label=label, linewidth=2, alpha=0.7, marker='o', markersize=3)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Sample Accuracy Across All Experiments')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 100])
        
        # 4. Final metrics heatmap
        ax4 = fig.add_subplot(gs[3, :])
        
        # Prepare data for heatmap
        if not self.summary_df.empty and 'final_accuracy' in self.summary_df.columns:
            # Filter out NaN values
            valid_data = self.summary_df.dropna(subset=['final_accuracy'])
            
            if not valid_data.empty:
                pivot_data = valid_data.pivot_table(
                    values='final_accuracy',
                    index='difficulty',
                    columns='normalization',
                    aggfunc='mean'
                )
                
                # Create heatmap only if we have data
                if not pivot_data.empty and pivot_data.notna().any().any():
                    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                               ax=ax4, cbar_kws={'label': 'Final Accuracy'})
                    ax4.set_title('Final Accuracy Heatmap')
                else:
                    ax4.text(0.5, 0.5, 'No accuracy data available for heatmap', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Final Accuracy Heatmap (No Data)')
            else:
                ax4.text(0.5, 0.5, 'No valid accuracy data', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Final Accuracy Heatmap (No Data)')
        else:
            ax4.text(0.5, 0.5, 'Summary data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Final Accuracy Heatmap (No Data)')
        
        plt.suptitle('Comprehensive Experiment Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = "experiment_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
        plt.show()
        
        return fig
    
    def calculate_variance_differences(self):
        """Calculate variance differences as discussed"""
        print("\n" + "="*60)
        print("VARIANCE DIFFERENCE ANALYSIS")
        print("="*60)
        
        if not self.metrics_data:
            print("No data available")
            return
        
        combined_df = pd.concat(self.metrics_data, ignore_index=True)
        
        # 1. Variance between different questions (difficulties)
        print("\n1. Variance Between Different Difficulties:")
        print("-" * 40)
        
        for norm in ['standard', 'no_std']:
            norm_data = combined_df[combined_df['normalization'] == norm]
            
            if len(norm_data) > 0:
                print(f"\n  Normalization: {norm}")
                
                # Group by difficulty and calculate mean reward for each
                diff_rewards = norm_data.groupby('difficulty')['reward_mean'].mean()
                
                if len(diff_rewards) > 1:
                    variance_between_difficulties = diff_rewards.var()
                    print(f"    Variance between difficulties: {variance_between_difficulties:.6f}")
                    print(f"    Mean rewards by difficulty:")
                    for diff, reward in diff_rewards.items():
                        print(f"      {diff}: {reward:.4f}")
        
        # 2. Variance between iterations (training steps)
        print("\n2. Variance Between Iterations:")
        print("-" * 40)
        
        for (diff, norm), group in combined_df.groupby(['difficulty', 'normalization']):
            if len(group) > 1:
                # Calculate variance of rewards across iterations
                reward_var = group['reward_mean'].var()
                print(f"  {diff}-{norm}: {reward_var:.6f}")
        
        # 3. Variance stability over time
        print("\n3. Variance Stability (Late Stage vs Early Stage):")
        print("-" * 40)
        
        for (diff, norm), group in combined_df.groupby(['difficulty', 'normalization']):
            if len(group) >= 20:
                early = group.head(10)['reward_mean'].var()
                late = group.tail(10)['reward_mean'].var()
                change = late - early
                print(f"  {diff}-{norm}:")
                print(f"    Early variance: {early:.6f}")
                print(f"    Late variance: {late:.6f}")
                print(f"    Change: {change:+.6f}")
    
    def generate_summary_table(self):
        """Generate a comprehensive summary table"""
        if self.summary_df.empty:
            print("No summary data available")
            return
        
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY TABLE")
        print("="*60)
        
        # Format and display summary
        display_cols = [
            'difficulty', 'normalization', 
            'final_accuracy', 'max_accuracy', 'accuracy_improvement',
            'final_reward', 'reward_improvement',
            'final_kl', 'kl_change',
            'reward_variance', 'accuracy_variance'
        ]
        
        # Filter columns that exist
        available_cols = [col for col in display_cols if col in self.summary_df.columns]
        
        # Create formatted table
        formatted_df = self.summary_df[available_cols].copy()
        
        # Format numeric columns
        for col in formatted_df.columns:
            if formatted_df[col].dtype in [np.float64, np.float32]:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        
        print(formatted_df.to_string(index=False))
        
        # Save to CSV
        self.summary_df.to_csv("experiment_summary.csv", index=False)
        print(f"\nFull summary saved to experiment_summary.csv")
    
    def analyze_normalization_effect(self):
        """Analyze the effect of different normalization methods"""
        print("\n" + "="*60)
        print("NORMALIZATION METHOD COMPARISON")
        print("="*60)
        
        if self.summary_df.empty:
            return
        
        for difficulty in ['easy', 'medium', 'hard']:
            diff_data = self.summary_df[self.summary_df['difficulty'] == difficulty]
            
            if len(diff_data) > 0:
                print(f"\n{difficulty.upper()} Difficulty:")
                print("-" * 40)
                
                for _, row in diff_data.iterrows():
                    print(f"  {row['normalization']}:")
                    if not pd.isna(row.get('final_accuracy', np.nan)):
                        print(f"    Final Accuracy: {row['final_accuracy']:.4f}")
                    if not pd.isna(row.get('final_reward', np.nan)):
                        print(f"    Final Reward: {row['final_reward']:.4f}")
                    if not pd.isna(row.get('reward_variance', np.nan)):
                        print(f"    Reward Variance: {row['reward_variance']:.6f}")

def main():
    print("Starting comprehensive experiment analysis...")
    
    # Create analyzer
    analyzer = ExperimentAnalyzer()
    
    # Run analysis
    summary = analyzer.run_comprehensive_analysis()
    
    # Generate outputs
    analyzer.calculate_variance_differences()
    analyzer.analyze_normalization_effect()
    analyzer.generate_summary_table()
    
    # Create plots
    analyzer.plot_comparison_grid()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - experiment_summary.csv")
    print("  - experiment_comparison.png")

if __name__ == "__main__":
    main()
