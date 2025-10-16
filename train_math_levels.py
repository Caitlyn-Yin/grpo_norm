#!/usr/bin/env python3
"""
train_math_levels.py - Train on MATH dataset with different difficulty levels
Tests how problem difficulty affects variance and gradient patterns
"""

import sys
import os
import argparse
import torch
import numpy as np
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig
import logging
from typing import Dict, Optional

# Import dataset loader
from math_datasets import MATHDataset

# Import utilities
from utils import (
    format_reward_func_qa, correctness_reward_func_qa,
    print_trainable_parameters,
    print_final_stats, get_pass_at_k_stats
)

# Import the trainer
from train_difficulty import DifficultyGRPOTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define difficulty level configurations
DIFFICULTY_CONFIGS = {
    'level1': {
        'name': 'Level 1 (Elementary)',
        'level': 'Level 1',
        'description': 'Elementary mathematics',
        'expected_pass_rate': 0.8  # These are typically easier
    },
    'level2': {
        'name': 'Level 2 (Middle School)',
        'level': 'Level 2',
        'description': 'Middle school mathematics',
        'expected_pass_rate': 0.6
    },
    'level3': {
        'name': 'Level 3 (High School)',
        'level': 'Level 3',
        'description': 'High school mathematics',
        'expected_pass_rate': 0.4
    },
    'level4': {
        'name': 'Level 4 (Competition)',
        'level': 'Level 4',
        'description': 'Competition mathematics',
        'expected_pass_rate': 0.2
    },
    'level5': {
        'name': 'Level 5 (Olympiad)',
        'level': 'Level 5',
        'description': 'Olympiad-level mathematics',
        'expected_pass_rate': 0.1
    },
    'mixed_easy': {
        'name': 'Mixed Easy (Levels 1-2)',
        'levels': ['Level 1', 'Level 2'],
        'description': 'Mix of elementary and middle school',
        'expected_pass_rate': 0.7
    },
    'mixed_hard': {
        'name': 'Mixed Hard (Levels 4-5)',
        'levels': ['Level 4', 'Level 5'],
        'description': 'Mix of competition and olympiad',
        'expected_pass_rate': 0.15
    },
    'all': {
        'name': 'All Levels (1-5)',
        'levels': None,  # Will load all
        'description': 'Complete MATH dataset',
        'expected_pass_rate': 0.4
    }
}

def load_math_by_difficulty(
    difficulty_key: str, 
    subjects: Optional[str] = None,
    args = None
) -> tuple:
    """Load MATH dataset filtered by difficulty level"""
    
    config = DIFFICULTY_CONFIGS[difficulty_key]
    
    # Parse subjects if provided
    subject_list = None
    if subjects:
        subject_list = [s.strip() for s in subjects.split(',')]
    else:
        # Use all subjects
        subject_list = MATHDataset.MATH_SUBJECTS    
    
    # Common parameters
    common_params = {
        'split': 'train',
        'include_answer': False,
        'include_reasoning': True,
        'few_shot': True,
        'num_shots': args.num_shots if args else 2,
        'seed': 42,
        'cot': True,
        'template': args.format if args else 'qa',
        'subjects': subject_list
    }
    
    # Handle single level vs multiple levels
    if 'level' in config:
        # Single level
        dataset_obj = MATHDataset(
            **common_params,
            difficulty_level=config['level']
        )
    elif 'levels' in config and config['levels']:
        # Multiple specific levels - load and combine
        all_data = []
        for level in config['levels']:
            level_data = MATHDataset(
                **common_params,
                difficulty_level=level
            )
            all_data.append(level_data.dataset)
        
        # Combine datasets
        from datasets import concatenate_datasets
        combined = concatenate_datasets(all_data)
        
        # Create a wrapper object
        class CombinedDataset:
            def __init__(self, dataset):
                self.dataset = dataset
        
        dataset_obj = CombinedDataset(combined)
    else:
        # All levels
        dataset_obj = MATHDataset(**common_params)
    
    dataset = dataset_obj.dataset
    
    # Show statistics
    if 'level' in dataset[0]:
        levels = {}
        for item in dataset:
            level = item.get('level', 'unknown')
            levels[level] = levels.get(level, 0) + 1
        
        for level in sorted(levels.keys()):
            print(f"  {level}: {levels[level]} problems")
    
    if 'subject' in dataset[0]:
        subjects = {}
        for item in dataset:
            subject = item.get('subject', 'unknown')
            subjects[subject] = subjects.get(subject, 0) + 1
        
        for subject in sorted(subjects.keys()):
            print(f"  {subject}: {subjects[subject]} problems")
    
    return dataset, config

def parse_args():
    parser = argparse.ArgumentParser(description="Train GRPO on MATH dataset by difficulty levels")
    
    # Difficulty selection
    parser.add_argument('--difficulty', type=str, required=True,
                       choices=list(DIFFICULTY_CONFIGS.keys()),
                       help='Difficulty configuration to use')
    
    # Subject filtering
    parser.add_argument('--subjects', type=str, default=None,
                       help='Comma-separated subjects (e.g., "algebra,geometry")')
    
    # Dataset size
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of training samples')
    
    # Training parameters
    parser.add_argument('--format', type=str, default='qa', choices=['qa', 'code'])
    parser.add_argument('--num_shots', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Math-1.5B')
    parser.add_argument('--num_generations', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--kl_beta', type=float, default=0.02)
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--normalization', type=str, default='standard',
                       choices=['standard', 'no_std', 'batch_std'])
    
    # WandB settings
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='grpo-math-levels')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    
    # 1) add CLI flags 
    parser.add_argument('--analyze_variance', action='store_true',
                        help='Track per-prompt reward variance over time')
    parser.add_argument('--log_curvature', action='store_true',
                        help='Estimate per-prompt empirical Fisher (method 2)')
    parser.add_argument('--curv_mode', type=str, default='trace',
                        choices=['trace', 'blockwise'],
                        help='Reduce diag-Fisher to scalar per prompt')
    parser.add_argument('--lora_only_fisher', action='store_true',
                        help='Restrict Fisher to LoRA params')
    parser.add_argument('--log_cosine', action='store_true',
                        help='Log gradient cosine similarities across prompts')
    parser.add_argument('--cosine_pairs', type=int, default=300,
                        help='Random pairs per step for cosine stats')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set experiment name based on difficulty if not provided
    if not args.exp_name:
        args.exp_name = f"math_{args.difficulty}_{args.normalization}"
    
    config = DIFFICULTY_CONFIGS[args.difficulty]
    
    
    # Load dataset
    try:
        dataset, config = load_math_by_difficulty(
            args.difficulty,
            args.subjects,
            args
        )
        
        # Apply sample limit if specified
        if args.max_samples and len(dataset) > args.max_samples:
            dataset = dataset.select(range(args.max_samples))
            print(f"Limited to {args.max_samples} samples")
        
        # Shuffle dataset
        dataset = dataset.shuffle(seed=42)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup WandB
    if args.use_wandb:
        run_name = args.wandb_run_name or f"math-{args.difficulty}-{args.normalization}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                'difficulty_name': config['name'],
                'expected_pass_rate': config['expected_pass_rate']
            },
            tags=[f"math-{args.difficulty}", args.normalization, config['name']]
        )
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = None
    
    # Set paths
    model_name = args.model_name
    output_dir = f'outputs/math_levels/{args.difficulty}/{args.normalization}/{args.exp_name}'
    
    # Set scale_rewards based on normalization
    scale_rewards_map = {
        'standard': True,
        'no_std': False,
        'batch_std': 'batch'
    }
    scale_rewards = scale_rewards_map[args.normalization]
    
    # Training configuration
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f'math-{args.difficulty}-{args.normalization}',
        learning_rate=args.learning_rate,
        beta=args.kl_beta,
        scale_rewards=scale_rewards,
        logging_steps=5,
        bf16=torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=args.num_generations,
        max_prompt_length=768,
        max_completion_length=1024,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=200,
        max_grad_norm=0.1,
        report_to=report_to,
        seed=42,
    )
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        bias='none',
        lora_dropout=0.05,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='auto'
    )
    
    model = get_peft_model(model, peft_config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Select reward functions
    reward_funcs = [format_reward_func_qa, correctness_reward_func_qa]
    
    # Create trainer with difficulty tag
    trainer = DifficultyGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        difficulty=f"math_{args.difficulty}",
        #analyze_variance=True,
        
        # NEW:
        analyze_variance=args.analyze_variance,
        log_curvature=args.log_curvature,
        curv_mode=args.curv_mode,
        lora_only_fisher=args.lora_only_fisher,
        log_cosine=args.log_cosine,
        cosine_pairs=args.cosine_pairs,
        tokenizer=tokenizer,
    )
    
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    
    # Print final statistics
    print_final_stats()
    # Log expected vs actual performance
    if hasattr(trainer, 'pass_at_k_history') and trainer.pass_at_k_history:
        actual_pass_rate = np.mean(trainer.pass_at_k_history)
        
        if args.use_wandb:
            wandb.summary['expected_pass_rate'] = config['expected_pass_rate']
            wandb.summary['actual_pass_rate'] = actual_pass_rate
            wandb.summary['performance_ratio'] = actual_pass_rate / config['expected_pass_rate']
    if args.use_wandb:
        stats = get_pass_at_k_stats()
        if stats:
            wandb.summary.update(stats)
        wandb.summary['difficulty_level'] = args.difficulty
        wandb.summary['difficulty_name'] = config['name']
        wandb.finish()

if __name__ == "__main__":
    main()