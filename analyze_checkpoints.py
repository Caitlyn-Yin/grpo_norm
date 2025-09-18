import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import logging
from typing import Dict, List, Optional, Union, Any
from tqdm import tqdm

from variance_analysis import VarianceAnalyzer
from gsm8k_difficulty import GSM8KDifficulty
from gsm8k import GSM8K
from complete_std_analysis import PolicyStdAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_from_checkpoint(checkpoint_path):
    """Load a model from a checkpoint directory"""
    try:
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        base_model_name = peft_config.base_model_name_or_path
        
        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load adapter
        logger.info(f"Loading adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def calculate_policy_gradients(model, tokenizer, prompts, device="cuda"):
    """Calculate policy gradients for a batch of prompts"""
    model.train()  # Set to train mode to enable gradient computation
    
    all_gradients = {}
    all_token_counts = {}
    all_fisher_info = {}
    
    for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        # Generate a unique ID for this prompt
        prompt_id = f"prompt_{idx}"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass with gradient computation
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Store token count
        token_count = inputs["attention_mask"].sum().item()
        all_token_counts[prompt_id] = token_count
        
        # Capture gradients
        score_functions = {}
        hooks = []
        
        # Only track LoRA parameters to keep computational cost manageable
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                # Register hook to capture gradients
                def hook_factory(name):
                    def hook(grad):
                        if grad is not None:
                            score_functions[name] = grad.detach().clone()
                    return hook
                
                hooks.append(param.register_hook(hook_factory(name)))
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Collect gradients
        param_grads = []
        for name, grad in score_functions.items():
            param_grads.append(grad.flatten())
        
        if param_grads:
            # Concatenate all parameter gradients
            all_grads = torch.cat(param_grads)
            
            # Store gradients
            all_gradients[prompt_id] = all_grads.cpu().numpy()
            
            # Calculate Fisher information (squared norm of score function)
            fisher_info = torch.norm(all_grads) ** 2
            
            # Normalize by sequence length
            if token_count > 0:
                fisher_info = fisher_info / token_count
            
            # Store Fisher information
            all_fisher_info[prompt_id] = fisher_info.item()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Zero gradients
        model.zero_grad()
    
    return all_gradients, all_token_counts, all_fisher_info

def calculate_reward_variance(model, tokenizer, prompts, num_samples=8, max_new_tokens=100):
    """Calculate reward variance by sampling multiple completions for each prompt"""
    model.eval()
    
    all_rewards = {}
    all_reward_stds = {}
    
    for idx, prompt in enumerate(tqdm(prompts, desc="Calculating reward variance")):
        prompt_id = f"prompt_{idx}"
        
        # Generate multiple samples
        completions = []
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
                completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                completions.append(completion)
        
        # Simple reward function: length of completion (for demonstration)
        # In a real scenario, you would use a proper reward function
        rewards = [len(c) / 100.0 for c in completions]
        all_rewards[prompt_id] = rewards
        
        # Calculate reward standard deviation
        if len(rewards) > 1:
            reward_std = np.std(rewards)
            all_reward_stds[prompt_id] = reward_std
    
    return all_rewards, all_reward_stds

def run_variance_analysis(args):
    """Run variance analysis on model checkpoints"""
    # Create variance analyzer
    analyzer = VarianceAnalyzer(save_dir=args.output_dir)
    
    # Load model
    model, tokenizer = load_model_from_checkpoint(args.checkpoint_path)
    
    # Load dataset
    if args.difficulty:
        logger.info(f"Loading {args.difficulty} difficulty dataset")
        dataset = GSM8KDifficulty(
            difficulty=args.difficulty,
            data_dir=args.data_dir,
            split='test',
            include_answer=True,
            include_reasoning=True,
            few_shot=True,
            num_shots=2,
            seed=42,
            cot=True,
            template='qa',
            max_samples=args.max_samples
        ).dataset
    else:
        logger.info("Loading standard GSM8K dataset")
        dataset = GSM8K(
            split='test',
            include_answer=True,
            include_reasoning=True,
            few_shot=True,
            num_shots=2,
            seed=42,
            cot=True,
            template='qa',
            max_samples=args.max_samples
        ).dataset
    
    # Extract prompts
    prompts = [item["prompt"] for item in dataset]
    answers = [item["answer"] if "answer" in item else None for item in dataset]
    
    if args.max_samples:
        prompts = prompts[:args.max_samples]
        answers = answers[:args.max_samples]
    
    logger.info(f"Loaded {len(prompts)} prompts for analysis")
    
    # Calculate policy gradients
    logger.info("Calculating policy gradients...")
    gradients, token_counts, fisher_info = calculate_policy_gradients(model, tokenizer, prompts)
    
    # Calculate reward variance
    logger.info("Calculating reward variance...")
    rewards, reward_stds = calculate_reward_variance(model, tokenizer, prompts, num_samples=args.num_samples)
    
    # Record data in analyzer
    iteration = 0  # Use 0 for single checkpoint analysis
    
    for prompt_id in gradients.keys():
        # Record gradient
        analyzer.record_gradient(prompt_id, iteration, gradients[prompt_id])
        
        # Record Fisher information
        if prompt_id in fisher_info:
            analyzer.record_fisher_info(prompt_id, iteration, fisher_info[prompt_id])
        
        # Record reward std
        if prompt_id in reward_stds:
            reward_std = reward_stds[prompt_id]
            analyzer.record_reward_std(prompt_id, iteration, reward_std)
            
            # Record variance for variance difference analysis
            variance = reward_std ** 2
            analyzer.record_variance(prompt_id, iteration, variance)
    
    # Run analysis
    logger.info("Running variance analysis...")
    analysis_results = analyzer.run_full_analysis()
    
    # Calculate policy std using PolicyStdAnalyzer
    if args.calculate_policy_std:
        logger.info("Calculating policy standard deviation...")
        std_analyzer = PolicyStdAnalyzer(tokenizer)
        overall_std, per_question_stds = std_analyzer.compute_policy_std(model, prompts)
        
        logger.info(f"Overall policy std: {overall_std:.4f}")
        
        # Save policy std results
        with open(os.path.join(args.output_dir, "policy_std_results.txt"), "w") as f:
            f.write(f"Overall policy std: {overall_std:.4f}\n\n")
            f.write("Per-question policy std:\n")
            for i, std in enumerate(per_question_stds):
                f.write(f"Question {i}: {std:.4f}\n")
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}/")
    
    return analysis_results

def analyze_multiple_checkpoints(args):
    """Analyze multiple checkpoints to track changes over iterations"""
    # Create variance analyzer
    analyzer = VarianceAnalyzer(save_dir=args.output_dir)
    
    # Get list of checkpoint directories
    checkpoint_dirs = []
    for root, dirs, files in os.walk(args.checkpoint_path):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint-"):
                checkpoint_dirs.append(os.path.join(root, dir_name))
    
    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    
    if args.max_checkpoints and len(checkpoint_dirs) > args.max_checkpoints:
        # Take evenly spaced checkpoints
        indices = np.linspace(0, len(checkpoint_dirs) - 1, args.max_checkpoints, dtype=int)
        checkpoint_dirs = [checkpoint_dirs[i] for i in indices]
    
    logger.info(f"Found {len(checkpoint_dirs)} checkpoints to analyze")
    
    # Load dataset
    if args.difficulty:
        logger.info(f"Loading {args.difficulty} difficulty dataset")
        dataset = GSM8KDifficulty(
            difficulty=args.difficulty,
            data_dir=args.data_dir,
            split='test',
            include_answer=True,
            include_reasoning=True,
            few_shot=True,
            num_shots=2,
            seed=42,
            cot=True,
            template='qa',
            max_samples=args.max_samples
        ).dataset
    else:
        logger.info("Loading standard GSM8K dataset")
        dataset = GSM8K(
            split='test',
            include_answer=True,
            include_reasoning=True,
            few_shot=True,
            num_shots=2,
            seed=42,
            cot=True,
            template='qa',
            max_samples=args.max_samples
        ).dataset
    
    # Extract prompts
    prompts = [item["prompt"] for item in dataset]
    answers = [item["answer"] if "answer" in item else None for item in dataset]
    
    if args.max_samples:
        prompts = prompts[:args.max_samples]
        answers = answers[:args.max_samples]
    
    logger.info(f"Loaded {len(prompts)} prompts for analysis")
    
    # Analyze each checkpoint
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        logger.info(f"Analyzing checkpoint {i+1}/{len(checkpoint_dirs)}: {checkpoint_dir}")
        
        try:
            # Extract iteration number from checkpoint name
            iteration = int(checkpoint_dir.split("-")[-1])
            
            # Load model
            model, tokenizer = load_model_from_checkpoint(checkpoint_dir)
            
            # Calculate policy gradients
            gradients, token_counts, fisher_info = calculate_policy_gradients(model, tokenizer, prompts)
            
            # Calculate reward variance
            rewards, reward_stds = calculate_reward_variance(model, tokenizer, prompts, num_samples=args.num_samples)
            
            # Record data in analyzer
            for prompt_id in gradients.keys():
                # Record gradient
                analyzer.record_gradient(prompt_id, iteration, gradients[prompt_id])
                
                # Record Fisher information
                if prompt_id in fisher_info:
                    analyzer.record_fisher_info(prompt_id, iteration, fisher_info[prompt_id])
                
                # Record reward std
                if prompt_id in reward_stds:
                    reward_std = reward_stds[prompt_id]
                    analyzer.record_reward_std(prompt_id, iteration, reward_std)
                    
                    # Record variance for variance difference analysis
                    variance = reward_std ** 2
                    analyzer.record_variance(prompt_id, iteration, variance)
            
            # Calculate policy std using PolicyStdAnalyzer if requested
            if args.calculate_policy_std:
                std_analyzer = PolicyStdAnalyzer(tokenizer)
                overall_std, per_question_stds = std_analyzer.compute_policy_std(model, prompts)
                
                # Save to a CSV for later analysis
                std_data = {
                    'iteration': iteration,
                    'overall_std': overall_std,
                    'question_stds': per_question_stds
                }
                
                # Append to CSV
                std_df = pd.DataFrame({
                    'iteration': [iteration] * len(per_question_stds),
                    'question_id': [f"prompt_{i}" for i in range(len(per_question_stds))],
                    'policy_std': per_question_stds
                })
                
                std_csv_path = os.path.join(args.output_dir, "policy_std_data.csv")
                if i == 0 and not os.path.exists(std_csv_path):
                    std_df.to_csv(std_csv_path, index=False)
                else:
                    std_df.to_csv(std_csv_path, mode='a', header=False, index=False)
            
            # Free up memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error analyzing checkpoint {checkpoint_dir}: {e}")
            continue
    
    # Run final analysis
    logger.info("Running final variance analysis...")
    analysis_results = analyzer.run_full_analysis()
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}/")
    
    return analysis_results

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze variance and Fisher information in model checkpoints")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint or directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, default="variance_analysis_results",
                       help="Directory to save analysis results")
    parser.add_argument("--difficulty", type=str, default=None, choices=["easy", "medium", "hard"],
                       help="Dataset difficulty level (if using GSM8KDifficulty)")
    parser.add_argument("--data_dir", type=str, default="data/gsm8k_difficulty_subsets",
                       help="Directory containing difficulty-graded datasets")
    parser.add_argument("--max_samples", type=int, default=10,
                       help="Maximum number of samples to analyze")
    parser.add_argument("--num_samples", type=int, default=8,
                       help="Number of samples to generate for reward variance calculation")
    parser.add_argument("--multiple_checkpoints", action="store_true",
                       help="Analyze multiple checkpoints to track changes over iterations")
    parser.add_argument("--max_checkpoints", type=int, default=5,
                       help="Maximum number of checkpoints to analyze if multiple_checkpoints is True")
    parser.add_argument("--calculate_policy_std", action="store_true",
                       help="Calculate policy standard deviation using PolicyStdAnalyzer")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Starting variance analysis with args: {args}")
    
    if args.multiple_checkpoints:
        analysis_results = analyze_multiple_checkpoints(args)
    else:
        analysis_results = run_variance_analysis(args)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
