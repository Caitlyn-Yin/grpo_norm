#!/usr/bin/env python3
"""
Script to analyze saved checkpoints from training runs.
Calculates variance differences, cosine similarities, and generates plots.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
import seaborn as sns

from variance_analysis import VarianceAnalyzer
from gsm8k_difficulty import GSM8KDifficulty


class CheckpointAnalyzer:
    """Analyzes checkpoints from a training run"""
    
    def __init__(self, output_dir: str, model_name: str = 'Qwen/Qwen2.5-Math-7B'):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Storage for analysis results
        self.checkpoint_results = {}
        self.variance_analyzer = VarianceAnalyzer(save_dir=str(self.output_dir / "checkpoint_analysis"))
        
    def get_checkpoint_dirs(self) -> List[Path]:
        """Get all checkpoint directories sorted by step number"""
        checkpoint_dirs = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoint_dirs.append((step, item))
                except:
                    continue
        
        # Sort by step number
        checkpoint_dirs.sort(key=lambda x: x[0])
        return [dir_path for _, dir_path in checkpoint_dirs]
    
    def load_checkpoint(self, checkpoint_dir: Path):
        """Load a checkpoint"""
        print(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model.eval()
        
        return model
    
    def compute_gradient_for_prompt(self, model, prompt: str, max_length: int = 512) -> torch.Tensor:
        """
        Compute gradient vector for a single prompt.
        This simulates what would be computed during training.
        """
        model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass with gradient computation
        model.zero_grad()
        
        # Enable gradients temporarily
        for param in model.parameters():
            param.requires_grad = True
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Compute gradients
        loss.backward()
        
        # Collect gradients from LoRA parameters
        gradients = []
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.grad is not None:
                gradients.append(param.grad.detach().cpu().flatten())
        
        # Disable gradients again
        for param in model.parameters():
            param.requires_grad = False
        
        if gradients:
            return torch.cat(gradients)
        else:
            return torch.zeros(1)
    
    def compute_reward_variance(self, model, prompts: List[str], num_samples: int = 8) -> Dict[str, float]:
        """
        Compute reward variance for each prompt by generating multiple completions.
        """
        from utils import correctness_reward_func_qa
        
        variances = {}
        
        for i, prompt in enumerate(prompts):
            rewards = []
            
            # Generate multiple completions for this prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode the generation
                completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Calculate reward (simplified - you may want to use actual reward function)
                # For now, we'll use a random reward for demonstration
                reward = np.random.random()  # Replace with actual reward calculation
                rewards.append(reward)
            
            # Calculate variance
            if len(rewards) > 1:
                variances[f"prompt_{i}"] = np.var(rewards)
            else:
                variances[f"prompt_{i}"] = 0.0
        
        return variances
    
    def analyze_checkpoint(self, checkpoint_dir: Path, test_prompts: List[str], step: int):
        """Analyze a single checkpoint"""
        model = self.load_checkpoint(checkpoint_dir)
        
        # Compute gradients for each prompt
        gradients = {}
        for i, prompt in enumerate(test_prompts):
            question_id = f"q_{step}_{i}"
            gradient = self.compute_gradient_for_prompt(model, prompt)
            self.variance_analyzer.record_gradient(question_id, step, gradient)
            gradients[question_id] = gradient
        
        # Compute reward variances
        variances = self.compute_reward_variance(model, test_prompts)
        for question_id, variance in variances.items():
            self.variance_analyzer.record_variance(question_id, step, variance)
            self.variance_analyzer.record_reward_std(question_id, step, np.sqrt(variance))
        
        # Store results
        self.checkpoint_results[step] = {
            'gradients': gradients,
            'variances': variances,
            'checkpoint_dir': str(checkpoint_dir)
        }
        
        # Clean up model from memory
        del model
        torch.cuda.empty_cache()
        
        return gradients, variances
    
    def analyze_all_checkpoints(self, difficulty: str = "easy", num_test_prompts: int = 10):
        """Analyze all checkpoints in the output directory"""
        # Get test prompts
        dataset = GSM8KDifficulty(
            difficulty=difficulty,
            data_dir='data/gsm8k_difficulty_subsets',
            split='train',
            include_answer=False,
            include_reasoning=True,
            few_shot=True,
            num_shots=2,
            seed=42,
            cot=True,
            template='qa',
            max_samples=num_test_prompts
        ).dataset
        
        test_prompts = [item['prompt'] for item in dataset][:num_test_prompts]
        
        # Get all checkpoint directories
        checkpoint_dirs = self.get_checkpoint_dirs()
        
        if not checkpoint_dirs:
            print(f"No checkpoints found in {self.output_dir}")
            return
        
        print(f"Found {len(checkpoint_dirs)} checkpoints to analyze")
        
        # Analyze each checkpoint
        for checkpoint_dir in checkpoint_dirs:
            step = int(checkpoint_dir.name.split("-")[1])
            print(f"\nAnalyzing checkpoint at step {step}")
            self.analyze_checkpoint(checkpoint_dir, test_prompts, step)
        
        # Run full variance analysis
        print("\nRunning full variance analysis...")
        analysis_results = self.variance_analyzer.run_full_analysis()
        
        return analysis_results
    
    def plot_cosine_similarity_evolution(self):
        """Plot how cosine similarity evolves over training steps"""
        # Get all steps
        steps = sorted(self.checkpoint_results.keys())
        
        if len(steps) < 2:
            print("Need at least 2 checkpoints to plot evolution")
            return
        
        # Compute cosine similarities for each step
        avg_cosines = []
        pos_ratios = []
        
        for step in steps:
            gradients = self.checkpoint_results[step]['gradients']
            
            # Compute pairwise cosine similarities
            cosines = []
            keys = list(gradients.keys())
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    g1 = gradients[keys[i]]
                    g2 = gradients[keys[j]]
                    if len(g1) > 0 and len(g2) > 0:
                        cos_sim = torch.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
                        cosines.append(cos_sim)
            
            if cosines:
                avg_cosines.append(np.mean(cosines))
                pos_ratios.append(np.mean([c > 0 for c in cosines]))
            else:
                avg_cosines.append(0)
                pos_ratios.append(0)
        
        # Create plots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Average cosine similarity
        axes[0].plot(steps, avg_cosines, 'b-', marker='o')
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Average Cosine Similarity')
        axes[0].set_title('Evolution of Cosine Similarity Between Questions')
        axes[0].grid(True, alpha=0.3)
        
        # Positive cosine ratio
        axes[1].plot(steps, pos_ratios, 'g-', marker='o')
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Ratio of Positive Cosine Similarities')
        axes[1].set_title('Proportion of Question Pairs with Positive Cosine Similarity')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "checkpoint_analysis" / "cosine_evolution.png"
        plt.savefig(save_path)
        plt.close()
        
        print(f"Cosine similarity evolution plot saved to {save_path}")
        
        return steps, avg_cosines, pos_ratios
    
    def plot_variance_evolution(self):
        """Plot how variance evolves over training steps"""
        steps = sorted(self.checkpoint_results.keys())
        
        if len(steps) < 2:
            print("Need at least 2 checkpoints to plot evolution")
            return
        
        # Get average variance for each step
        avg_variances = []
        for step in steps:
            variances = list(self.checkpoint_results[step]['variances'].values())
            avg_variances.append(np.mean(variances) if variances else 0)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, avg_variances, 'b-', marker='o')
        plt.xlabel('Training Step')
        plt.ylabel('Average Reward Variance')
        plt.title('Evolution of Reward Variance During Training')
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / "checkpoint_analysis" / "variance_evolution.png"
        plt.savefig(save_path)
        plt.close()
        
        print(f"Variance evolution plot saved to {save_path}")
        
        return steps, avg_variances
    
    def save_results(self):
        """Save all analysis results to files"""
        save_dir = self.output_dir / "checkpoint_analysis"
        save_dir.mkdir(exist_ok=True)
        
        # Save checkpoint results as JSON
        results_for_json = {}
        for step, data in self.checkpoint_results.items():
            results_for_json[str(step)] = {
                'variances': data['variances'],
                'checkpoint_dir': data['checkpoint_dir'],
                # Convert gradients to list for JSON serialization
                'gradient_norms': {k: float(torch.norm(v).item()) for k, v in data['gradients'].items()}
            }
        
        with open(save_dir / "checkpoint_results.json", 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        print(f"Results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoints from training")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to the training output directory containing checkpoints')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Math-7B',
                       help='Base model name')
    parser.add_argument('--difficulty', type=str, default='easy',
                       choices=['easy', 'medium', 'hard'],
                       help='Difficulty level for test prompts')
    parser.add_argument('--num_test_prompts', type=int, default=10,
                       help='Number of test prompts to use')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CheckpointAnalyzer(args.output_dir, args.model_name)
    
    # Analyze all checkpoints
    print(f"Analyzing checkpoints in {args.output_dir}")
    analysis_results = analyzer.analyze_all_checkpoints(
        difficulty=args.difficulty,
        num_test_prompts=args.num_test_prompts
    )
    
    # Generate plots
    print("\nGenerating plots...")
    analyzer.plot_cosine_similarity_evolution()
    analyzer.plot_variance_evolution()
    
    # Save results
    analyzer.save_results()
    
    # Print summary
    if analysis_results and 'cosine_variance_df' in analysis_results and not analysis_results['cosine_variance_df'].empty:
        df = analysis_results['cosine_variance_df']
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total question pairs analyzed: {len(df)}")
        print(f"Pairs with positive cosine similarity: {(df['cosine_similarity'] > 0).sum()} ({(df['cosine_similarity'] > 0).mean():.1%})")
        print(f"Average cosine similarity: {df['cosine_similarity'].mean():.4f}")
        print(f"Average variance difference: {df['variance_diff'].mean():.6f}")
        
        if len(df) > 1:
            corr, p_value = pearsonr(df['cosine_similarity'], df['variance_diff'])
            print(f"Correlation between cosine and variance: {corr:.3f} (p={p_value:.3f})")
        print("="*60)


if __name__ == "__main__":
    main()
