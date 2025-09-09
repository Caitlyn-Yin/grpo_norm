import sys
import types
import importlib.util

# vLLM hard-import workaround
if "vllm" not in sys.modules:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.__spec__ = importlib.util.spec_from_loader("vllm", loader=None)
    vllm_stub.__path__ = []
    sys.modules["vllm"] = vllm_stub
    
    vllm_distributed_stub = types.ModuleType("vllm.distributed")
    vllm_distributed_stub.__spec__ = importlib.util.spec_from_loader("vllm.distributed", loader=None)
    vllm_distributed_stub.__path__ = []
    sys.modules["vllm.distributed"] = vllm_distributed_stub
    vllm_stub.distributed = vllm_distributed_stub
    
    vllm_dc_stub = types.ModuleType("vllm.distributed.device_communicators")
    vllm_dc_stub.__spec__ = importlib.util.spec_from_loader("vllm.distributed.device_communicators", loader=None)
    vllm_dc_stub.__path__ = []
    sys.modules["vllm.distributed.device_communicators"] = vllm_dc_stub
    vllm_distributed_stub.device_communicators = vllm_dc_stub
    
    pynccl_stub = types.ModuleType("vllm.distributed.device_communicators.pynccl")
    sys.modules["vllm.distributed.device_communicators.pynccl"] = pynccl_stub
    vllm_dc_stub.pynccl = pynccl_stub
    
    class PyNcclCommunicator:
        def __init__(self, *args, **kwargs):
            pass
    pynccl_stub.PyNcclCommunicator = PyNcclCommunicator
    
    vllm_distributed_utils_stub = types.ModuleType("vllm.distributed.utils")
    sys.modules["vllm.distributed.utils"] = vllm_distributed_utils_stub
    vllm_distributed_stub.utils = vllm_distributed_utils_stub
    
    class StatelessProcessGroup:
        def __init__(self, *args, **kwargs):
            pass
    vllm_distributed_utils_stub.StatelessProcessGroup = StatelessProcessGroup
    
    vllm_engine_stub = types.ModuleType("vllm.engine")
    sys.modules["vllm.engine"] = vllm_engine_stub
    vllm_stub.engine = vllm_engine_stub
    
    vllm_llm_stub = types.ModuleType("vllm.llm")
    sys.modules["vllm.llm"] = vllm_llm_stub
    vllm_stub.llm = vllm_llm_stub
    
    class LLM:
        def __init__(self, *args, **kwargs):
            pass
    vllm_llm_stub.LLM = LLM
    vllm_stub.LLM = LLM
    
    class AsyncLLMEngine:
        def __init__(self, *args, **kwargs):
            pass
    vllm_engine_stub.AsyncLLMEngine = AsyncLLMEngine
    
    class SamplingParams:
        def __init__(self, *args, **kwargs):
            pass
    vllm_stub.SamplingParams = SamplingParams
    
    vllm_sampling_params_stub = types.ModuleType("vllm.sampling_params")
    sys.modules["vllm.sampling_params"] = vllm_sampling_params_stub
    vllm_stub.sampling_params = vllm_sampling_params_stub
    
    class GuidedDecodingParams:
        def __init__(self, *args, **kwargs):
            pass
    vllm_sampling_params_stub.GuidedDecodingParams = GuidedDecodingParams

import os
import argparse
import torch
import numpy as np
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from typing import Dict, List, Union
import logging

from gsm8k_difficulty import GSM8KDifficulty
from utils import (
    format_reward_func_qa, correctness_reward_func_qa,
    format_reward_func_code, correctness_reward_func_code,
    print_trainable_parameters,
    print_final_stats, get_pass_at_k_stats
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DifficultyGRPOTrainer(GRPOTrainer):
    
    def __init__(self, *args, difficulty="easy", **kwargs):
        super().__init__(*args, **kwargs)
        self.difficulty = difficulty
        self.step_count = 0
    
    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        self.step_count += 1
        
        logs["difficulty"] = self.difficulty

        if hasattr(self.args, 'scale_rewards'):
            if self.args.scale_rewards == True or self.args.scale_rewards == "group":
                logs["normalization"] = "standard"
            elif self.args.scale_rewards == False:
                logs["normalization"] = "no_std"
            elif self.args.scale_rewards == "batch":
                logs["normalization"] = "batch_std"
        
        super().log(logs, start_time)

def parse_args():
    parser = argparse.ArgumentParser(description="Train GRPO on difficulty-graded GSM8K")
    
    parser.add_argument('--difficulty', type=str, default='easy',
                       choices=['easy', 'medium', 'hard'],
                       help='Dataset difficulty level')
    parser.add_argument('--data_dir', type=str, default='data/gsm8k_difficulty_subsets',
                       help='Directory containing difficulty-graded datasets')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (for debugging)')
    
    parser.add_argument('--format', type=str, default='qa', choices=['qa', 'code'])
    parser.add_argument('--num_shots', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Math-1.5B')
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--max_steps", type=int, default=400)
    
    parser.add_argument('--normalization', type=str, default='standard',
                       choices=['standard', 'no_std', 'batch_std'])
    
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='grpo-final3-difficulty')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='default')
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Arguments: {args}")
    print(f"Training on {args.difficulty} difficulty dataset")
    
    # Setup WandB
    if args.use_wandb:
        run_name = args.wandb_run_name or f"grpo-final3-{args.normalization}-{args.difficulty}-{args.exp_name}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=[args.normalization, args.difficulty, args.exp_name]
        )
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = None
    
    try:
        dataset = GSM8KDifficulty(
            difficulty=args.difficulty,
            data_dir=args.data_dir,
            split='train',
            include_answer=False,
            include_reasoning=True,
            few_shot=True,
            num_shots=args.num_shots,
            seed=42,
            cot=True,
            template=args.format,
            max_samples=args.max_samples
        ).dataset.shuffle(seed=42)
        
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the dataset files exist in {args.data_dir}")
        print("Expected files: train_easy.parquet, train_medium.parquet, train_hard.parquet")
        return
    
    model_name = args.model_name
    output_dir = f'outputs/GRPO_final3_difficulty/{args.difficulty}/{args.normalization}/{model_name.split("/")[-1]}/{args.exp_name}'
    
    if args.normalization == 'standard':
        scale_rewards = True
    elif args.normalization == 'no_std':
        scale_rewards = False
    elif args.normalization == 'batch_std':
        scale_rewards = "batch"
    else:
        raise ValueError(f"Unknown normalization: {args.normalization}")
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f'GRPO_final3_difficulty-{args.normalization}-{args.difficulty}-{args.exp_name}',
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
        save_steps=10,
        save_total_limit=200,
        max_grad_norm=0.1,
        report_to=report_to,
        log_on_each_node=False,
        seed=42,
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        bias='none',
        lora_dropout=0.05,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='auto'
    )
    
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if args.format == 'qa':
        reward_funcs = [format_reward_func_qa, correctness_reward_func_qa]
    elif args.format == 'code':
        reward_funcs = [format_reward_func_code, correctness_reward_func_code]
    
    trainer = DifficultyGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        difficulty=args.difficulty,
    )
    
    print(f"\nStarting training:")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Normalization: {args.normalization}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  scale_rewards setting: {scale_rewards}")
    
    trainer.train()
    
    trainer.save_model(output_dir)
    print(f"\nModel saved to {output_dir}")
    
    print_final_stats()
    
    if args.use_wandb:
        stats = get_pass_at_k_stats()
        if stats:
            wandb.summary.update(stats)
            wandb.summary["difficulty"] = args.difficulty
        wandb.finish()

if __name__ == "__main__":
    main()