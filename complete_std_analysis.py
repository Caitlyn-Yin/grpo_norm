#!/usr/bin/env python
# complete_std_analysis.py - Analyze and plot two key metrics from checkpoints
# Analyze and plot two key metrics from checkpoints

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import glob
import os
from typing import List, Dict, Tuple
import json
from tqdm import tqdm

class PolicyStdAnalyzer:
    """Complete policy standard deviation analyzer"""
    
    def __init__(self, base_model_name="Qwen/Qwen2.5-Math-1.5B"):
        self.base_model_name = base_model_name
        self.tokenizer = None
        
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")):
            print(f"Loading LoRA adapter from {checkpoint_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            print(f"Loading full model from {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return model
    
    def get_test_questions(self, difficulty="easy", num_questions=10):
        """Get fixed test question set"""
        questions = {
            "easy": [
                "What is 5 + 3?",
                "If John has 10 apples and gives 3 away, how many are left?",
                "A store has 15 items and sells 7. How many remain?",
                "What is 4 × 3?",
                "Sarah reads 5 pages per day. How many in 6 days?",
                "A pizza has 8 slices. Tom eats 3. How many left?",
                "There are 20 students. 8 are boys. How many girls?",
                "A book costs $12. How much for 3 books?",
                "Jim walks 2 miles daily. Distance in a week?",
                "25 cookies in a jar. Kids eat 10. How many remain?"
            ],
            "medium": [
                "A train travels 60 mph for 2.5 hours. Distance?",
                "30% of 200 students play sports. How many?",
                "Rectangle: length 12, width 8. Area?",
                "Save $50 weekly. Total in 12 weeks?",
                "Recipe: 2.5 cups flour for 6 cookies. For 15?",
                "Shirt costs $40 after 20% discount. Original price?",
                "Car: 240 miles on 8 gallons. MPG?",
                "5 workers finish in 12 days. Time for 3 workers?",
                "Pool fills at 10 gal/min. Time for 450 gallons?",
                "$1000 at 5% annual interest for 2 years. Final amount?"
            ],
            "hard": [
                "Factory: 240 widgets/day. After +25% then -10%, output?",
                "Sum of integers from 1 to 100?",
                "Ball drops from 100ft, bounces to 60% height. Height after 3 bounces?",
                "Rain probability 0.3 daily. Chance of rain in 3 days?",
                "Circular track radius 50m. Laps for 2km?",
                "f(x)=2x+3, g(x)=x². Find f(g(2))?",
                "20% annual growth. Years to double?",
                "3 pumps fill tank in 4 hours. Time for 2 pumps?",
                "Ways to arrange 5 different books?",
                "log(x)=2, log(y)=3. Find log(xy)?"
            ]
        }
        
        selected = questions.get(difficulty, questions["easy"])[:num_questions]
        
        formatted = []
        for q in selected:
            prompt = f"Question: {q}\nSolution: Let's think step by step."
            formatted.append(prompt)
        
        return formatted
    
    def compute_policy_std(self, model, test_prompts, max_new_tokens=50):
        """
        Compute policy standard deviation - core function
        Returns: (overall_mean_std, per_question_stds)
        """
        model.eval()
        all_stds = []
        per_question_stds = []
        
        with torch.no_grad():
            for prompt in test_prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False 
                )
                
                step_stds = []
                if outputs.scores:
                    for score in outputs.scores:        
                        probs = torch.softmax(score, dim=-1)
                        top_probs, _ = torch.max(probs, dim=-1)
                        std = torch.sqrt(top_probs * (1 - top_probs))
                        step_stds.append(std.mean().item())
                
                if step_stds:
                    question_avg_std = np.mean(step_stds)
                else:
                    question_avg_std = 0.5  
                
                per_question_stds.append(question_avg_std)
                all_stds.extend(step_stds)
        
        overall_mean = np.mean(all_stds) if all_stds else 0.5
        
        return overall_mean, per_question_stds
    
    def analyze_all_checkpoints(self, output_dir, difficulty="easy", num_questions=10):
    
        test_prompts = self.get_test_questions(difficulty, num_questions)
        
        checkpoint_dirs = []
        
        pattern = os.path.join(output_dir, "checkpoint-*")
        checkpoint_dirs.extend(glob.glob(pattern))
        
        if os.path.exists(os.path.join(output_dir, "adapter_model.bin")) or \
           os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
            checkpoint_dirs.append(output_dir)
        
        checkpoint_dirs = sorted(set(checkpoint_dirs))
        
        if not checkpoint_dirs:
            print(f"No checkpoints found in {output_dir}")
            return None
        
        print(f"Found {len(checkpoint_dirs)} checkpoints to analyze")
        
        results = []
        
        for checkpoint_path in tqdm(checkpoint_dirs, desc="Analyzing checkpoints"):
            if "checkpoint-" in checkpoint_path:
                step = int(checkpoint_path.split("-")[-1])
            else:
                step = 9999  
            
            try:
                model = self.load_checkpoint(checkpoint_path)
                
                overall_std, question_stds = self.compute_policy_std(model, test_prompts)
                
                results.append({
                    'step': step,
                    'overall_std': overall_std,
                    'question_stds': question_stds,
                    'checkpoint': checkpoint_path
                })
                
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {checkpoint_path}: {e}")
                continue
        
        return results

def plot_metrics(results, difficulty, normalization, save_dir="plots"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    steps = [r['step'] for r in results]
    overall_stds = [r['overall_std'] for r in results]
    
    sorted_indices = np.argsort(steps)
    steps = [steps[i] for i in sorted_indices]
    overall_stds = [overall_stds[i] for i in sorted_indices]
    results = [results[i] for i in sorted_indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    ax = axes[0, 0]
    ax.plot(steps, overall_stds, 'b-', linewidth=2, marker='o', markersize=5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Maximum (0.5)')
    ax.axhline(y=0.25, color='g', linestyle='--', alpha=0.5, label='Target (<<0.5)')
    
    cumulative_avg = np.mean(overall_stds)
    ax.axhline(y=cumulative_avg, color='purple', linestyle=':', 
               label=f'Average: {cumulative_avg:.4f}')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Policy Std: $\\sqrt{\\pi(1)(1-\\pi(1))}$')
    ax.set_title(f'Metric 1: Overall Policy Std\n{difficulty}-{normalization}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    condition_met = "✓ Satisfied" if cumulative_avg < 0.25 else "✗ Not Satisfied"
    ax.text(0.02, 0.98, f'Condition (<<0.5): {condition_met}',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax = axes[0, 1]
    
    stats_text = f"""Statistics for $\\frac{{1}}{{T}}\\sum_{{t=0}}^{{T-1}} \\sqrt{{\\pi_t(1)(1-\\pi_t(1))}}$:

    Mean: {np.mean(overall_stds):.4f}
    Std: {np.std(overall_stds):.4f}
    Min: {np.min(overall_stds):.4f}
    Max: {np.max(overall_stds):.4f}
    Final: {overall_stds[-1]:.4f}
    
    Condition: {"✓ << 0.5" if np.mean(overall_stds) < 0.25 else "✗ Not << 0.5"}
    Ratio to 0.5: {np.mean(overall_stds)/0.5:.1%}
    """
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.axis('off')
    
    ax = axes[1, 0]
    
    num_questions = len(results[0]['question_stds'])
    for q_idx in range(num_questions):
        q_stds = [r['question_stds'][q_idx] for r in results]
        ax.plot(steps, q_stds, marker='o', markersize=3, 
                alpha=0.7, label=f'Q{q_idx+1}')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Per-Question Policy Std')
    ax.set_title(f'Metric 2: Question-Level Std Evolution\n{difficulty}-{normalization}')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    ax = axes[1, 1]
    
    key_steps = [0, len(results)//2, -1]  # Start, middle, end
    colors = ['red', 'orange', 'green']
    
    for idx, step_idx in enumerate(key_steps):
        if step_idx < len(results):
            step = steps[step_idx]
            q_stds = results[step_idx]['question_stds']
            
            ax.hist(q_stds, bins=15, alpha=0.5, color=colors[idx], 
                   label=f'Step {step}', density=True)
    
    ax.set_xlabel('Question Std Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Question Stds Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Policy Standard Deviation Analysis\n{difficulty.upper()} - {normalization.upper()}', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = f"{save_dir}/std_analysis_{difficulty}_{normalization}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Analyze policy std from checkpoints")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory containing checkpoints")
    parser.add_argument("--difficulty", type=str, default="easy",
                       choices=["easy", "medium", "hard"])
    parser.add_argument("--normalization", type=str, default="standard",
                       choices=["standard", "no_std"])
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="std_analysis_plots")
    
    args = parser.parse_args()
    
    analyzer = PolicyStdAnalyzer()
    
    print(f"Analyzing {args.difficulty}-{args.normalization} experiment...")
    results = analyzer.analyze_all_checkpoints(
        args.output_dir, 
        args.difficulty, 
        args.num_questions
    )
    
    if results:
        plot_metrics(results, args.difficulty, args.normalization, args.save_dir)
        
        df_data = []
        for r in results:
            for q_idx, q_std in enumerate(r['question_stds']):
                df_data.append({
                    'step': r['step'],
                    'overall_std': r['overall_std'],
                    'question_id': q_idx,
                    'question_std': q_std
                })
        
        df = pd.DataFrame(df_data)
        csv_file = f"{args.save_dir}/std_data_{args.difficulty}_{args.normalization}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Data saved to {csv_file}")
        
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        final_overall = results[-1]['overall_std']
        final_avg = np.mean(results[-1]['question_stds'])
        cumulative = np.mean([r['overall_std'] for r in results])
        
        print(f"Final Overall Std: {final_overall:.4f}")
        print(f"Final Question Avg: {final_avg:.4f}")
        print(f"Cumulative Average: {cumulative:.4f}")
        print(f"Condition (<<0.5): {'✓ Satisfied' if cumulative < 0.25 else '✗ Not Satisfied'}")
        print("="*60)

if __name__ == "__main__":
    main()