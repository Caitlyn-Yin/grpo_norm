import sys
import types
import importlib.util
import os
os.environ["WANDB_DISABLE_SERVICE"] = "true"

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
    
    # Create vllm.distributed.device_communicators module
    vllm_dc_stub = types.ModuleType("vllm.distributed.device_communicators")
    vllm_dc_stub.__spec__ = importlib.util.spec_from_loader("vllm.distributed.device_communicators", loader=None)
    vllm_dc_stub.__path__ = []
    sys.modules["vllm.distributed.device_communicators"] = vllm_dc_stub
    vllm_distributed_stub.device_communicators = vllm_dc_stub
    
    # Create pynccl module
    pynccl_stub = types.ModuleType("vllm.distributed.device_communicators.pynccl")
    sys.modules["vllm.distributed.device_communicators.pynccl"] = pynccl_stub
    vllm_dc_stub.pynccl = pynccl_stub
    
    # Add dummy PyNcclCommunicator class
    class PyNcclCommunicator:
        def __init__(self, *args, **kwargs):
            pass
    pynccl_stub.PyNcclCommunicator = PyNcclCommunicator
    
    # Create vllm.distributed.utils module
    vllm_distributed_utils_stub = types.ModuleType("vllm.distributed.utils")
    sys.modules["vllm.distributed.utils"] = vllm_distributed_utils_stub
    vllm_distributed_stub.utils = vllm_distributed_utils_stub
    
    # Add StatelessProcessGroup class
    class StatelessProcessGroup:
        def __init__(self, *args, **kwargs):
            pass
    vllm_distributed_utils_stub.StatelessProcessGroup = StatelessProcessGroup
    
    # Add LLM and SamplingParams classes
    class LLM:
        def __init__(self, *args, **kwargs):
            pass
    vllm_stub.LLM = LLM
    
    class SamplingParams:
        def __init__(self, *args, **kwargs):
            pass
    vllm_stub.SamplingParams = SamplingParams
    
    # Create vllm.sampling_params module
    vllm_sampling_params_stub = types.ModuleType("vllm.sampling_params")
    sys.modules["vllm.sampling_params"] = vllm_sampling_params_stub
    vllm_stub.sampling_params = vllm_sampling_params_stub
    
    # Add GuidedDecodingParams class
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
from typing import Dict, List, Optional, Union, Any
import logging
import uuid

from gsm8k import GSM8K
from utils import (
    format_reward_func_qa, correctness_reward_func_qa,
    format_reward_func_code, correctness_reward_func_code,
    print_trainable_parameters,
)
from variance_analysis import VarianceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomGRPOTrainer(GRPOTrainer):
    """
    Extended GRPO Trainer for monitoring Pass@K metrics
    Does not override advantage calculation, lets TRL handle normalization
    """
    
    def __init__(self, *args, track_metrics=True, k=8, analyze_variance=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_metrics = track_metrics
        self.k = k
        self.pass_at_k_history = []
        self.accuracy_history = []
        self.step_count = 0
        
        # Variance analysis
        self.analyze_variance = analyze_variance
        if analyze_variance:
            self.variance_analyzer = VarianceAnalyzer(save_dir="variance_analysis_results")
        
        # Keep track of gradients and score functions for Fisher information
        self.current_score_functions = {}  # {question_id: [score_functions]}
        self.current_question_ids = {}  # Maps batch indices to question IDs
        self.current_token_counts = {}  # {question_id: token_count}
        
    def compute_rewards(self, completions: List[str], prompts: List[str], **kwargs):
        """
        Compute rewards and track Pass@K metrics
        """
        # Call parent method to get rewards
        rewards = super().compute_rewards(completions, prompts, **kwargs)
        
        # Track Pass@K and accuracy
        if self.track_metrics and 'final_answer' in kwargs:
            self._track_pass_at_k(completions, kwargs['final_answer'])
        
        # Calculate reward standard deviation for each question
        if self.analyze_variance:
            self._calculate_reward_std(rewards, prompts)
        
        return rewards
    
    def _track_pass_at_k(self, completions: List[str], final_answers: Union[List, str]):
        """
        Calculate and record Pass@K metrics
        Pass@K: For each question's K generations, 1 if at least one is correct, 0 otherwise
        """
        # Handle final_answers format
        if not isinstance(final_answers, list):
            final_answers = [final_answers] * len(completions)
        elif len(final_answers) == 1:
            final_answers = final_answers * len(completions)
        
        # Calculate Pass@K for each question
        num_questions = len(completions) // self.k
        
        # Generate question IDs if needed
        if self.analyze_variance:
            self.current_question_ids = {}
        
        for q_idx in range(num_questions):
            # Get K generations for this question
            start_idx = q_idx * self.k
            end_idx = start_idx + self.k
            q_completions = completions[start_idx:end_idx]
            q_answers = final_answers[start_idx:end_idx]
            
            # Check if each generation is correct
            correct_rewards = correctness_reward_func_qa(q_completions, final_answer=q_answers)
            
            # Pass@K: at least one is correct
            pass_at_k = 1.0 if any(r > 0 for r in correct_rewards) else 0.0
            self.pass_at_k_history.append(pass_at_k)
            
            # Individual sample accuracy
            accuracy = sum(correct_rewards) / len(correct_rewards)
            self.accuracy_history.append(accuracy)
            
            # Generate a unique ID for this question for variance analysis
            if self.analyze_variance:
                question_id = f"q_{self.step_count}_{q_idx}"
                for i in range(start_idx, end_idx):
                    self.current_question_ids[i] = question_id
        
        # Log to wandb
        if wandb.run is not None:
            current_pass_at_k = np.mean(self.pass_at_k_history[-num_questions:])
            current_accuracy = np.mean(self.accuracy_history[-num_questions:])
            cumulative_pass_at_k = np.mean(self.pass_at_k_history)
            cumulative_accuracy = np.mean(self.accuracy_history)
            
            wandb.log({
                "train/pass_at_k": current_pass_at_k,
                "train/sample_accuracy": current_accuracy,
                "train/cumulative_pass_at_k": cumulative_pass_at_k,
                "train/cumulative_accuracy": cumulative_accuracy,
                "train/total_questions_seen": len(self.pass_at_k_history),
            }, step=self.step_count)
        if num_questions > 0:
            current_pass_at_k = np.mean(self.pass_at_k_history[-num_questions:])
            print(f"Step {self.step_count} - Pass@{self.k}: {current_pass_at_k:.2%}")
    
    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """
        Extended logging
        """
        self.step_count += 1
        
        # Add additional monitoring metrics
        if self.pass_at_k_history:
            logs["train/running_pass_at_k"] = np.mean(self.pass_at_k_history[-100:])  # Last 100 questions
            logs["train/running_accuracy"] = np.mean(self.accuracy_history[-100:])
        
        # Record normalization type
        if hasattr(self.args, 'scale_rewards'):
            if self.args.scale_rewards == True or self.args.scale_rewards == "group":
                logs["normalization"] = "standard"
            elif self.args.scale_rewards == False:
                logs["normalization"] = "no_std"
        
        # Run variance analysis every 20 steps and only after step 50
        # This gives time for data to accumulate and reduces computational overhead
        if self.analyze_variance and self.step_count > 50 and self.step_count % 20 == 0:
            try:
                logger.info(f"Running variance analysis at step {self.step_count}")
                analysis_results = self.variance_analyzer.run_full_analysis()
                
                # Log cosine similarity analysis if available
                if 'cosine_variance_df' in analysis_results and not analysis_results['cosine_variance_df'].empty:
                    df_cos = analysis_results['cosine_variance_df']
                    
                    # Log cosine similarity statistics
                    if wandb.run is not None:
                        logs["analysis/avg_cosine_similarity"] = df_cos['cosine_similarity'].mean()
                        logs["analysis/positive_cosine_ratio"] = (df_cos['cosine_similarity'] > 0).mean()
                        logs["analysis/avg_variance_diff"] = df_cos['variance_diff'].mean()
                        
                        # Correlation between cosine similarity and variance difference
                        if len(df_cos) > 1:
                            from scipy.stats import pearsonr
                            corr, _ = pearsonr(df_cos['cosine_similarity'], df_cos['variance_diff'])
                            logs["analysis/cosine_variance_corr"] = corr
                    
                    logger.info(f"Cosine analysis: {len(df_cos)} pairs, {(df_cos['cosine_similarity'] > 0).mean():.1%} positive")
                
                # Log variance differences
                if 'question_variance_diff' in analysis_results:
                    q_var_diff = analysis_results['question_variance_diff']
                    if wandb.run is not None and q_var_diff:
                        logs["analysis/avg_question_variance_diff"] = np.mean(list(q_var_diff.values()))
                
                if 'iteration_variance_diff' in analysis_results:
                    iter_var_diff = analysis_results['iteration_variance_diff']
                    if wandb.run is not None and iter_var_diff:
                        logs["analysis/avg_iteration_variance_diff"] = np.mean(list(iter_var_diff.values()))
                    
            except Exception as e:
                logger.warning(f"Variance analysis failed: {e}")
        
        super().log(logs, start_time)

def get_memory_optimized_config(args):
    """
    Return memory-optimized training configuration
    """
    return GRPOConfig(
        output_dir=args.output_dir,
        run_name=f'GRPO-{args.normalization}-{args.exp_name}',
        learning_rate=args.learning_rate,
        beta=args.kl_beta,
        scale_rewards=args.scale_rewards,
        
        logging_steps=10,  
        bf16=torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,  
        
        num_generations=args.num_generations,
        max_prompt_length=512,
        max_completion_length=512, 
        
        
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=20,  
        save_total_limit=2,  
        
        
        max_grad_norm=0.1,
        warmup_ratio=0.1, 
        optim="adamw_8bit", 
        
        report_to="wandb" if args.use_wandb else None,
        log_on_each_node=False,
        seed=42,

        remove_unused_columns=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

def load_model_memory_optimized(model_name):
    """
    Memory-optimized model loading
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",  
        load_in_8bit=False,  
        max_memory={0: "48GB"},  
        offload_folder="offload",  
    )
    
    model.gradient_checkpointing_enable()
    
    return model
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with GRPO on GSM8K")
    parser.add_argument('--format', type=str, default='qa', choices=['qa', 'code'])
    parser.add_argument('--num_shots', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Math-7B')
    parser.add_argument("--num_generations", type=int, default=8, help="K: candidates per prompt")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--kl_beta", type=float, default=0.02, help="KL coefficient")
    parser.add_argument("--max_steps", type=int, default=400)
    
    # Normalization methods
    parser.add_argument('--normalization', type=str, default='standard',
                       choices=['standard', 'no_std'],
                       help='Advantage normalization: standard=(r-mean)/std, no_std=(r-mean)')
    
    # WandB configuration
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='iclr01_easy', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--exp_name', type=str, default='default', help='Experiment name')
    
    return parser.parse_args()

    def _calculate_reward_std(self, rewards: List[float], prompts: List[str]):
        """
        Calculate and record reward standard deviation for each question
        """
        if not self.analyze_variance or not self.current_question_ids:
            return
        
        # Group rewards by question
        question_rewards = {}
        for i, reward in enumerate(rewards):
            if i in self.current_question_ids:
                question_id = self.current_question_ids[i]
                if question_id not in question_rewards:
                    question_rewards[question_id] = []
                question_rewards[question_id].append(reward)
        
        # Calculate standard deviation for each question
        for question_id, q_rewards in question_rewards.items():
            if len(q_rewards) > 1:  # Need at least 2 samples for std
                reward_std = np.std(q_rewards)
                self.variance_analyzer.record_reward_std(question_id, self.step_count, reward_std)
                
                # Record variance for variance difference analysis
                variance = reward_std ** 2
                self.variance_analyzer.record_variance(question_id, self.step_count, variance)
    
    def compute_loss(self, model, inputs: Dict[str, Any], return_outputs=False, **kwargs):
        """
        Override compute_loss to capture gradients for Fisher information calculation
        """
        # Call parent method to compute loss
        # GRPOTrainer doesn't support return_outputs, so we just get the loss
        loss = super().compute_loss(model, inputs, **kwargs)
        
        # For Fisher information calculation, we would need the model outputs
        # Since GRPO doesn't return outputs, we'll skip Fisher information calculation for now
        # The variance analysis will still work with reward variance tracking
        
        if return_outputs:
            # If outputs are requested, we need to return a simple object with loss
            # This is for compatibility with the Trainer class
            from types import SimpleNamespace
            return SimpleNamespace(loss=loss)
        return loss
    
    def _capture_score_functions_disabled(self, model_inputs, outputs):
        """
        Capture score functions (gradients of log probabilities) for Fisher information calculation
        """
        try:
            # Skip if loss is None or not a tensor
            if not hasattr(outputs, "loss") or outputs.loss is None or not isinstance(outputs.loss, torch.Tensor):
                logger.warning("Skipping Fisher information calculation: loss is None or not a tensor")
                return
                
            self.current_score_functions = {}
            self.current_token_counts = {}
            
            hooks = []
            score_functions = {}
            
            lora_params = []
            for name, param in self.model.named_parameters():
                if "lora" in name.lower() and param.requires_grad:
                    lora_params.append(param)
                    
                    score_functions[name] = []
                    
                    def hook_factory(name):
                        def hook(grad):
                            if grad is not None:
                                score_functions[name].append(grad.detach().clone())
                        return hook
                    
                    hooks.append(param.register_hook(hook_factory(name)))
            
            if not lora_params:
                logger.warning("No LoRA parameters found for Fisher information calculation")
                return
                
            try:
                outputs.loss.backward(retain_graph=True)
            except RuntimeError as e:
                logger.warning(f"Error in backward pass: {e}")
                for hook in hooks:
                    hook.remove()
                return
            
            batch_size = model_inputs["input_ids"].size(0) if "input_ids" in model_inputs else 1
            for batch_idx in range(batch_size):
                if batch_idx in self.current_question_ids:
                    question_id = self.current_question_ids[batch_idx]
                    
                    if "attention_mask" in model_inputs:
                        token_count = model_inputs["attention_mask"][batch_idx].sum().item()
                        self.current_token_counts[question_id] = token_count
                    
                    param_grads = []
                    for name in score_functions:
                        if score_functions[name] and len(score_functions[name]) > 0:
                            if batch_idx < score_functions[name][0].size(0):
                                param_grads.append(score_functions[name][0][batch_idx].flatten())
                    
                    if param_grads:
                        try:
                            all_grads = torch.cat(param_grads)
                            
                            self.variance_analyzer.record_gradient(question_id, self.step_count, all_grads)
                            
                            fisher_info = torch.norm(all_grads) ** 2
                            
                            if question_id in self.current_token_counts and self.current_token_counts[question_id] > 0:
                                fisher_info = fisher_info / self.current_token_counts[question_id]
                            
                            self.variance_analyzer.record_fisher_info(
                                question_id, 
                                self.step_count, 
                                fisher_info.item()
                            )
                        except Exception as e:
                            logger.warning(f"Error processing gradients: {e}")
            
            for hook in hooks:
                hook.remove()
                
            self.model.zero_grad()
            
        except Exception as e:
            logger.warning(f"Error capturing score functions: {e}")
            self.model.zero_grad()


def main():
    args = parse_args()
    print(f"Arguments: {args}")
    
    # Setup WandB   
    if args.use_wandb:
        run_name = args.wandb_run_name or f"grpo-{args.normalization}-{args.format}-{args.exp_name}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=[args.normalization, args.format, args.exp_name]
        )
        report_to = "wandb"
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = None
    
    dataset = GSM8K(
        split='train',
        include_answer=False,
        include_reasoning=True,
        few_shot=True,
        num_shots=args.num_shots,
        seed=42,
        cot=True,
        template=args.format
    ).dataset.shuffle(seed=42)
    
    model_name = args.model_name
    output_dir = f'outputs/GRPO_final2_difficulty/{args.normalization}/{args.format}/{model_name.split("/")[-1]}/{args.exp_name}'
    
    if args.normalization == 'standard':
        scale_rewards = True  
    elif args.normalization == 'no_std':
        scale_rewards = False
    else:
        raise ValueError(f"Unknown normalization: {args.normalization}")
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f'GRPO_final2_difficulty-{args.normalization}-{args.exp_name}',
        learning_rate=args.learning_rate,
        beta=args.kl_beta,  
        scale_rewards=scale_rewards,  
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_generations=args.num_generations,  
        max_prompt_length=1024,
        max_completion_length=1536,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=20,
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
    
    trainer = CustomGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        track_metrics=True,
        k=args.num_generations,
        analyze_variance=True,
    )
    
    
    print(f"\nStarting training with {args.normalization} normalization...")
    print(f"scale_rewards setting: {scale_rewards}")
    trainer.train()
    
    
    trainer.save_model(output_dir)
    from utils import print_final_stats, get_pass_at_k_stats
    print_final_stats()
    print(f"\nModel saved to {output_dir}")
    

    if args.use_wandb:
        stats = get_pass_at_k_stats()
        if stats:
            wandb.summary.update(stats)
    if trainer.pass_at_k_history:
        final_pass_at_k = np.mean(trainer.pass_at_k_history)
        final_accuracy = np.mean(trainer.accuracy_history)
        print(f"\nFinal Pass@{args.num_generations}: {final_pass_at_k:.4f}")
        print(f"Final Sample Accuracy: {final_accuracy:.4f}")
        
        if args.use_wandb:
            wandb.summary["final_pass_at_k"] = final_pass_at_k
            wandb.summary["final_accuracy"] = final_accuracy
            wandb.finish()

if __name__ == "__main__":
    main()