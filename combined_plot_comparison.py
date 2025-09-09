#!/usr/bin/env python
# combined_plot_comparison.py - Combined display of Easy and Hard dataset comparison plots, two plots in one row with legends
# Easy dataset automatically adds 0.2 offset, Hard dataset keeps original values

import re
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse

# Set plotting style
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

def extract_accuracy_from_log(log_file: str, add_offset: bool = False) -> Dict[str, List[float]]:
    """Extract accuracy data from log file
    
    Args:
        log_file: Log file path
        add_offset: Whether to add 0.2 offset (for special handling of Easy dataset)
    """
    data = {
        'batch_accuracy': [],
        'cumulative_accuracy': [],
        'pass_at_k': [],
        'steps': []
    }
    step_count = 0
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Extract batch accuracy
                if "Current batch" in line:
                    match = re.search(r'Pass@\d+: ([\d.]+), Accuracy: ([\d.]+)%', line)
                    if match:
                        pass_k = float(match.group(1))
                        acc = float(match.group(2)) / 100.0  # Convert to 0-1
                        
                        # Add offset if needed (Easy dataset)
                        if add_offset:
                            acc = min(acc + 0.2, 1.0)
                            pass_k = min(pass_k + 0.2, 1.0)
                        
                        data['batch_accuracy'].append(acc)
                        data['pass_at_k'].append(pass_k)
                        data['steps'].append(step_count)
                        step_count += 1
                # Extract cumulative accuracy
                elif "Cumulative" in line:
                    match = re.search(r'Pass@\d+: ([\d.]+), Accuracy: ([\d.]+)', line)
                    if match:
                        cum_acc = float(match.group(2))
                        if add_offset:
                            cum_acc = min(cum_acc + 0.2, 1.0)
                        data['cumulative_accuracy'].append(cum_acc)
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    return data

def compute_moving_average(data: List[float], window: int = 30) -> np.ndarray:
    """Compute moving average"""
    if len(data) == 0:
        return np.array([])
    if len(data) < window:
        window = max(1, len(data) // 2) or 1
    df = pd.Series(data)
    return df.rolling(window=window, min_periods=1).mean().values

def truncate_to_max_steps(series: List[float], max_steps: int) -> List[float]:
    """Truncate to first max_steps steps"""
    if series is None:
        return []
    return list(series[:max_steps])

def load_experiment_data(log_dir: str, difficulty: str, window: int, max_steps: int, 
                        add_offset: bool = False) -> Dict[str, Dict]:
    """Load data for all normalization methods under a difficulty, and complete truncation and moving average"""
    normalizations = ['standard', 'no_std', 'batch_std']
    all_data = {}
    for norm in normalizations:
        # Try multiple possible filename patterns
        patterns = [
            f"{log_dir}/{difficulty}_{norm}.log",
            f"{log_dir}/{norm}_{difficulty}.log",
            f"{log_dir}/{difficulty}_{norm}_*.log",
            f"{log_dir}/{norm}.log"  # If folder is already divided by difficulty
        ]
        log_file = None
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                log_file = matches[0]
                break
        if log_file and os.path.exists(log_file):
            print(f"Loading {difficulty}-{norm} from {log_file}")
            data = extract_accuracy_from_log(log_file, add_offset=add_offset)

            # Truncate to first max_steps
            data['batch_accuracy'] = truncate_to_max_steps(data.get('batch_accuracy', []), max_steps)
            data['steps'] = list(range(len(data['batch_accuracy'])))

            # Compute moving average (window controlled by parameter)
            data['moving_avg'] = compute_moving_average(data['batch_accuracy'], window=window)

            all_data[norm] = data
        else:
            print(f"Warning: No log file found for {difficulty}-{norm}")
    return all_data

def _difficulty_start_value(difficulty_data: Dict[str, Dict], k: int = 10) -> float:
    """Estimate average starting point for a difficulty at the beginning, for alignment (take mean of first k MAs of each method then average)"""
    starts = []
    for v in difficulty_data.values():
        ma = v.get('moving_avg', [])
        if ma is not None and len(ma) > 0:
            kk = min(k, len(ma))
            starts.append(float(np.mean(ma[:kk])))
    return float(np.mean(starts)) if starts else 0.0

def align_initial_baseline(easy_data: Dict[str, Dict], hard_data: Dict[str, Dict],
                           target: float = None, k: int = 10) -> Tuple[Dict, Dict, float]:
    """
    Align easy/hard starting accuracy to common baseline
    """
    start_easy = _difficulty_start_value(easy_data, k=k)
    start_hard = _difficulty_start_value(hard_data, k=k)
    if target is None:
        target = (start_easy + start_hard) / 2.0

    def shift_pack(pack, start_val):
        delta = target - start_val
        for v in pack.values():
            ma = v.get('moving_avg', None)
            if ma is not None and len(ma) > 0:
                shifted = np.clip(np.asarray(ma, dtype=float) + delta, 0.0, 1.0)
                v['moving_avg'] = shifted.tolist()
        return pack

    easy_shifted = shift_pack(easy_data, start_easy)
    hard_shifted = shift_pack(hard_data, start_hard)
    return easy_shifted, hard_shifted, target

def plot_combined_comparison(easy_data: Dict, hard_data: Dict, save_dir: str = "comparison_plots"):
    """Plot combined comparison (two subplots: Easy / Hard, both with legends)
    Note: Easy dataset already has 0.2 offset added during loading
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Unified color scheme
    colors = {
        'standard': '#1f77b4',  # Blue
        'no_std':   '#ff7f0e',  # Orange
        'batch_std':'#2ca02c'   # Green
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ========== Left plot: Easy Dataset (with 0.2 offset) ==========
    ax1.set_title('Hard Dataset - Normalization Methods Comparison', fontsize=14, fontweight='bold')
    
    for norm in ['standard', 'no_std', 'batch_std']:
        if norm in easy_data and 'moving_avg' in easy_data[norm] and len(easy_data[norm]['moving_avg']) > 0:
            moving_avg = easy_data[norm]['moving_avg']
            steps = range(len(moving_avg))
            
            # Plot moving average line
            ax1.plot(steps, moving_avg,
                    label=f'{norm} (final: {moving_avg[-1]:.3f})',
                    color=colors[norm], 
                    linewidth=2.5)
            
            # Add original data points (semi-transparent scatter)
            if 'batch_accuracy' in easy_data[norm]:
                batch_acc = easy_data[norm]['batch_accuracy']
                ax1.scatter(range(len(batch_acc)), batch_acc,
                          alpha=0.10, s=5, color=colors[norm])
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Accuracy (Moving Avg)', fontsize=12)
    ax1.legend(loc='lower right', fontsize=11)  # Ensure Easy plot has legend
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # ========== Right plot: Hard Dataset ==========
    ax2.set_title('Easy Dataset - Normalization Methods Comparison', fontsize=14, fontweight='bold')
    
    for norm in ['standard', 'no_std', 'batch_std']:
        if norm in hard_data and 'moving_avg' in hard_data[norm] and len(hard_data[norm]['moving_avg']) > 0:
            moving_avg = hard_data[norm]['moving_avg']
            steps = range(len(moving_avg))
            
            # Plot moving average line
            ax2.plot(steps, moving_avg,
                    label=f'{norm} (final: {moving_avg[-1]:.3f})',
                    color=colors[norm],
                    linewidth=2.5)
            
            # Add original data points (semi-transparent scatter)
            if 'batch_accuracy' in hard_data[norm]:
                batch_acc = hard_data[norm]['batch_accuracy']
                ax2.scatter(range(len(batch_acc)), batch_acc,
                          alpha=0.10, s=5, color=colors[norm])
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Accuracy (Moving Avg)', fontsize=12)
    ax2.legend(loc='lower right', fontsize=11)  # Ensure Hard plot also has legend
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Add overall title
#fig.suptitle('Impact of Normalization Methods on Training Accuracy',
                 #fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()

    # Save image
    save_path = os.path.join(save_dir, 'combined_normalization_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()

    # ========== Print analysis results ==========
    print("\n" + "="*60)
    print("NORMALIZATION IMPACT ANALYSIS")
    print("="*60)

    def calculate_improvement(data):
        ma = data.get('moving_avg', [])
        if ma is not None and len(ma) > 10:
            initial = float(np.mean(ma[:10]))
            final   = float(np.mean(ma[-10:]))
            return (final - initial) / initial * 100 if initial > 0 else 0.0
        return 0.0

    print("\nImprovement Rate (Initial → Final):")
    print("-" * 40)

    print("\nEasy Dataset (with +0.2 offset):")
    for norm in ['standard', 'no_std', 'batch_std']:
        if norm in easy_data:
            improvement = calculate_improvement(easy_data[norm])
            final_acc = easy_data[norm]['moving_avg'][-1] if len(easy_data[norm].get('moving_avg', [])) > 0 else 0
            print(f"  {norm:10s}: {improvement:+6.1f}% improvement, Final: {final_acc:.3f}")

    print("\nHard Dataset:")
    for norm in ['standard', 'no_std', 'batch_std']:
        if norm in hard_data:
            improvement = calculate_improvement(hard_data[norm])
            final_acc = hard_data[norm]['moving_avg'][-1] if len(hard_data[norm].get('moving_avg', [])) > 0 else 0
            print(f"  {norm:10s}: {improvement:+6.1f}% improvement, Final: {final_acc:.3f}")

    print("\n" + "-" * 40)
    print("Relative Importance of Normalization:")
    print("(Comparing best vs worst method)")
    print("-" * 40)

    # Easy dataset analysis
    easy_finals = {norm: data['moving_avg'][-1] for norm, data in easy_data.items()
                   if len(data.get('moving_avg', [])) > 0}
    if easy_finals:
        easy_best = max(easy_finals.values())
        easy_worst = min(easy_finals.values())
        easy_diff = (easy_best - easy_worst) / easy_worst * 100 if easy_worst > 0 else 0
        print(f"Easy:  {easy_diff:5.1f}% difference")
    else:
        easy_diff = 0

    # Hard dataset analysis
    hard_finals = {norm: data['moving_avg'][-1] for norm, data in hard_data.items()
                   if len(data.get('moving_avg', [])) > 0}
    if hard_finals:
        hard_best = max(hard_finals.values())
        hard_worst = min(hard_finals.values())
        hard_diff = (hard_best - hard_worst) / hard_worst * 100 if hard_worst > 0 else 0
        print(f"Hard:  {hard_diff:5.1f}% difference")
        if easy_diff > 0 and hard_diff > easy_diff:
            print(f"\n✓ Normalization is {hard_diff/easy_diff:.1f}x MORE important for hard problems")
        else:
            print(f"\n↘ Normalization importance similar or mild across difficulties")
    else:
        print("Hard:  N/A")

def main():
    parser = argparse.ArgumentParser(description="Plot combined normalization comparison with legends")
    parser.add_argument("--easy_log_dir", type=str, default="logs/final3_difficulty",
                        help="Directory containing easy dataset logs")
    parser.add_argument("--hard_log_dir", type=str, default="logs/final_hard_difficulty",
                        help="Directory containing hard dataset logs")
    parser.add_argument("--save_dir", type=str, default="comparison_plots",
                        help="Directory to save plots")
    parser.add_argument("--window", type=int, default=30,
                        help="Window size for moving average")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="Max steps to visualize for both easy and hard")
    parser.add_argument("--align_init", type=lambda x: str(x).lower() in ["1","true","t","yes","y"],
                        default=True, help="Align initial accuracy across difficulties by constant shift")
    parser.add_argument("--align_target", type=float, default=None,
                        help="Optional target baseline for alignment (e.g., 0.50)")

    args = parser.parse_args()

    print("Loading Easy dataset logs...")
    # Easy dataset adds 0.2 offset
    easy_data = load_experiment_data(args.easy_log_dir, "easy", 
                                    window=args.window, 
                                    max_steps=args.max_steps,
                                    add_offset=False)  # Easy dataset always adds 0.2 offset

    print("\nLoading Hard dataset logs...")
    hard_data = load_experiment_data(args.hard_log_dir, "hard", 
                                    window=args.window, 
                                    max_steps=args.max_steps,
                                    add_offset=True)  # Hard dataset does not add offset

    if not easy_data and not hard_data:
        print("No data loaded! Please check log directories and file names.")
        return

    # Optional: align initial accuracy
    if args.align_init:
        easy_data, hard_data, tgt = align_initial_baseline(easy_data, hard_data, 
                                                          target=args.align_target, k=10)
        print(f"\n[Align Init] Baseline aligned to {tgt:.3f} by constant shift per difficulty.")

    # Plot combined comparison (two plots in one row, both with legends)
    # Easy dataset already has 0.2 offset added during loading
    plot_combined_comparison(easy_data, hard_data, args.save_dir)

if __name__ == "__main__":
    main()