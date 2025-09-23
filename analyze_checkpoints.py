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

from utils import correctness_reward_func_qa

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

def calculate_policy_gradients(model, tokenizer, prompts, prompt_ids: Optional[List[str]] = None,
                               device: Optional[Union[str, torch.device]] = None):
    """Calculate policy gradients for a batch of prompts"""
    model.train()  # Set to train mode to enable gradient computation

    if device is None:
        device = next(model.parameters()).device

    all_gradients: Dict[str, np.ndarray] = {}
    all_token_counts: Dict[str, int] = {}
    all_fisher_info: Dict[str, float] = {}

    for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        prompt_id = prompt_ids[idx] if prompt_ids is not None else f"prompt_{idx}"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        token_count = inputs["attention_mask"].sum().item()
        all_token_counts[prompt_id] = token_count

        score_functions: Dict[str, torch.Tensor] = {}
        hooks = []

        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                def hook_factory(param_name):
                    def hook(grad):
                        if grad is not None:
                            score_functions[param_name] = grad.detach().clone()
                    return hook

                hooks.append(param.register_hook(hook_factory(name)))

        loss.backward()

        param_grads = [grad.flatten() for grad in score_functions.values()]

        if param_grads:
            all_grads = torch.cat(param_grads)
            all_gradients[prompt_id] = all_grads.cpu().numpy()

            fisher_info = torch.norm(all_grads) ** 2
            if token_count > 0:
                fisher_info = fisher_info / token_count
            all_fisher_info[prompt_id] = fisher_info.item()

        for hook in hooks:
            hook.remove()

        model.zero_grad(set_to_none=True)

    return all_gradients, all_token_counts, all_fisher_info

def calculate_reward_variance(model, tokenizer, prompts, answers: Optional[List[Optional[str]]] = None,
                              prompt_ids: Optional[List[str]] = None, num_samples: int = 8,
                              max_new_tokens: int = 100,
                              device: Optional[Union[str, torch.device]] = None):
    """Calculate reward variance and accuracy by sampling multiple completions for each prompt"""
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    all_rewards: Dict[str, List[float]] = {}
    all_reward_stds: Dict[str, float] = {}
    all_accuracy: Dict[str, float] = {}

    for idx, prompt in enumerate(tqdm(prompts, desc="Calculating reward variance")):
        prompt_id = prompt_ids[idx] if prompt_ids is not None else f"prompt_{idx}"

        target_answer = None
        if answers is not None and idx < len(answers):
            target_answer = answers[idx]

        has_answer = target_answer is not None and str(target_answer).strip() != ""

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        completions: List[str] = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            completion = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            completions.append(completion)

        if has_answer:
            rewards = correctness_reward_func_qa(completions, final_answer=target_answer)
        else:
            rewards = [len(c) / 100.0 for c in completions]

        rewards = [float(r) for r in rewards]

        if rewards:
            all_rewards[prompt_id] = rewards
            reward_std = float(np.std(rewards))
            all_reward_stds[prompt_id] = reward_std

            if has_answer:
                accuracy = float(np.mean(rewards))
                all_accuracy[prompt_id] = accuracy

    return all_rewards, all_reward_stds, all_accuracy

def run_variance_analysis(args):
    """Run variance analysis on model checkpoints"""
    analyzer = VarianceAnalyzer(save_dir=args.output_dir)

    model, tokenizer = load_model_from_checkpoint(args.checkpoint_path)

    if args.difficulty:
        logger.info(f"Loading {args.difficulty} difficulty dataset")
        dataset = GSM8KDifficulty(
            difficulty=args.difficulty,
            data_dir=args.data_dir,
            split='test',
            include_answer=False,
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
            include_answer=False,
            include_reasoning=True,
            few_shot=True,
            num_shots=2,
            seed=42,
            cot=True,
            template='qa',
            max_samples=args.max_samples
        ).dataset

    prompts: List[str] = []
    answers: List[Optional[str]] = []
    question_texts: List[Optional[str]] = []
    dataset_indices: List[int] = []

    for idx, item in enumerate(dataset):
        prompts.append(item["prompt"])
        answers.append(item.get("final_answer"))
        question_texts.append(item.get("question"))
        dataset_indices.append(idx)

    if args.max_samples:
        prompts = prompts[:args.max_samples]
        answers = answers[:args.max_samples]
        question_texts = question_texts[:args.max_samples]
        dataset_indices = dataset_indices[:args.max_samples]

    if not prompts:
        raise SystemExit("No prompts available for analysis.")

    if args.focus_question_idx is not None:
        if args.focus_question_idx < 0 or args.focus_question_idx >= len(prompts):
            raise SystemExit(
                f"focus_question_idx={args.focus_question_idx} is out of range for {len(prompts)} prompts"
            )

        prompts = [prompts[args.focus_question_idx]]
        answers = [answers[args.focus_question_idx]]
        question_texts = [question_texts[args.focus_question_idx]]
        dataset_indices = [dataset_indices[args.focus_question_idx]]

        logger.info(
            f"Focusing analysis on prompt index {args.focus_question_idx} (dataset idx {dataset_indices[0]})"
        )
        if question_texts[0]:
            logger.info(f"Selected question text: {question_texts[0]}")

    prompt_ids = [f"prompt_{idx}" for idx in dataset_indices]

    logger.info(f"Loaded {len(prompts)} prompts for analysis")

    metadata_df = pd.DataFrame({
        'prompt_id': prompt_ids,
        'dataset_index': dataset_indices,
        'question': question_texts,
        'final_answer': answers
    })
    metadata_df.to_csv(os.path.join(args.output_dir, "prompt_metadata.csv"), index=False)

    logger.info("Calculating policy gradients...")
    gradients, token_counts, fisher_info = calculate_policy_gradients(
        model, tokenizer, prompts, prompt_ids=prompt_ids
    )

    logger.info("Calculating reward variance and accuracy...")
    _reward_samples, reward_stds, accuracies = calculate_reward_variance(
        model,
        tokenizer,
        prompts,
        answers=answers,
        prompt_ids=prompt_ids,
        num_samples=args.num_samples
    )

    if accuracies:
        avg_accuracy = float(np.mean(list(accuracies.values())))
        logger.info(f"Average empirical accuracy across prompts: {avg_accuracy:.3f}")

    iteration = 0
    for prompt_id in prompt_ids:
        if prompt_id in gradients:
            analyzer.record_gradient(prompt_id, iteration, gradients[prompt_id])

        if prompt_id in fisher_info:
            analyzer.record_fisher_info(prompt_id, iteration, fisher_info[prompt_id])

        if prompt_id in reward_stds:
            reward_std = reward_stds[prompt_id]
            analyzer.record_reward_std(prompt_id, iteration, reward_std)
            analyzer.record_variance(prompt_id, iteration, reward_std ** 2)

        if prompt_id in accuracies:
            analyzer.record_accuracy(prompt_id, iteration, accuracies[prompt_id])

    logger.info("Running variance analysis...")
    analysis_results = analyzer.run_full_analysis()

    if args.calculate_policy_std:
        logger.info("Calculating policy standard deviation...")
        std_analyzer = PolicyStdAnalyzer(tokenizer)
        overall_std, per_question_stds = std_analyzer.compute_policy_std(model, prompts)

        logger.info(f"Overall policy std: {overall_std:.4f}")

        with open(os.path.join(args.output_dir, "policy_std_results.txt"), "w") as f:
            f.write(f"Overall policy std: {overall_std:.4f}\n\n")
            f.write("Per-question policy std:\n")
            for pid, std in zip(prompt_ids, per_question_stds):
                f.write(f"{pid}: {std:.4f}\n")

    logger.info(f"Analysis complete. Results saved to {args.output_dir}/")

    return analysis_results

def analyze_multiple_checkpoints(args):
    """Analyze multiple checkpoints to track changes over iterations"""
    analyzer = VarianceAnalyzer(save_dir=args.output_dir)

    checkpoint_dirs: List[str] = []
    for root, dirs, files in os.walk(args.checkpoint_path):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint-"):
                checkpoint_dirs.append(os.path.join(root, dir_name))

    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))

    if args.max_checkpoints and len(checkpoint_dirs) > args.max_checkpoints:
        indices = np.linspace(0, len(checkpoint_dirs) - 1, args.max_checkpoints, dtype=int)
        checkpoint_dirs = [checkpoint_dirs[i] for i in indices]

    logger.info(f"Found {len(checkpoint_dirs)} checkpoints to analyze")

    if args.difficulty:
        logger.info(f"Loading {args.difficulty} difficulty dataset")
        dataset = GSM8KDifficulty(
            difficulty=args.difficulty,
            data_dir=args.data_dir,
            split='test',
            include_answer=False,
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
            include_answer=False,
            include_reasoning=True,
            few_shot=True,
            num_shots=2,
            seed=42,
            cot=True,
            template='qa',
            max_samples=args.max_samples
        ).dataset

    prompts: List[str] = []
    answers: List[Optional[str]] = []
    question_texts: List[Optional[str]] = []
    dataset_indices: List[int] = []

    for idx, item in enumerate(dataset):
        prompts.append(item["prompt"])
        answers.append(item.get("final_answer"))
        question_texts.append(item.get("question"))
        dataset_indices.append(idx)

    if args.max_samples:
        prompts = prompts[:args.max_samples]
        answers = answers[:args.max_samples]
        question_texts = question_texts[:args.max_samples]
        dataset_indices = dataset_indices[:args.max_samples]

    if not prompts:
        raise SystemExit("No prompts available for analysis.")

    if args.focus_question_idx is not None:
        if args.focus_question_idx < 0 or args.focus_question_idx >= len(prompts):
            raise SystemExit(
                f"focus_question_idx={args.focus_question_idx} is out of range for {len(prompts)} prompts"
            )

        prompts = [prompts[args.focus_question_idx]]
        answers = [answers[args.focus_question_idx]]
        question_texts = [question_texts[args.focus_question_idx]]
        dataset_indices = [dataset_indices[args.focus_question_idx]]

        logger.info(
            f"Focusing analysis on prompt index {args.focus_question_idx} (dataset idx {dataset_indices[0]})"
        )
        if question_texts[0]:
            logger.info(f"Selected question text: {question_texts[0]}")

    prompt_ids = [f"prompt_{idx}" for idx in dataset_indices]

    logger.info(f"Loaded {len(prompts)} prompts for analysis")

    metadata_df = pd.DataFrame({
        'prompt_id': prompt_ids,
        'dataset_index': dataset_indices,
        'question': question_texts,
        'final_answer': answers
    })
    metadata_df.to_csv(os.path.join(args.output_dir, "prompt_metadata.csv"), index=False)

    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        logger.info(f"Analyzing checkpoint {i + 1}/{len(checkpoint_dirs)}: {checkpoint_dir}")

        model = None
        try:
            iteration = int(checkpoint_dir.split("-")[-1])

            model, tokenizer = load_model_from_checkpoint(checkpoint_dir)

            gradients, token_counts, fisher_info = calculate_policy_gradients(
                model, tokenizer, prompts, prompt_ids=prompt_ids
            )

            _reward_samples, reward_stds, accuracies = calculate_reward_variance(
                model,
                tokenizer,
                prompts,
                answers=answers,
                prompt_ids=prompt_ids,
                num_samples=args.num_samples
            )

            if accuracies:
                avg_accuracy = float(np.mean(list(accuracies.values())))
                logger.info(f"Iteration {iteration}: mean empirical accuracy {avg_accuracy:.3f}")

            for prompt_id in prompt_ids:
                if prompt_id in gradients:
                    analyzer.record_gradient(prompt_id, iteration, gradients[prompt_id])

                if prompt_id in fisher_info:
                    analyzer.record_fisher_info(prompt_id, iteration, fisher_info[prompt_id])

                if prompt_id in reward_stds:
                    reward_std = reward_stds[prompt_id]
                    analyzer.record_reward_std(prompt_id, iteration, reward_std)
                    analyzer.record_variance(prompt_id, iteration, reward_std ** 2)

                if prompt_id in accuracies:
                    analyzer.record_accuracy(prompt_id, iteration, accuracies[prompt_id])

            if args.calculate_policy_std:
                std_analyzer = PolicyStdAnalyzer(tokenizer)
                overall_std, per_question_stds = std_analyzer.compute_policy_std(model, prompts)

                std_csv_path = os.path.join(args.output_dir, "policy_std_data.csv")
                std_df = pd.DataFrame({
                    'iteration': [iteration] * len(per_question_stds),
                    'question_id': prompt_ids,
                    'policy_std': per_question_stds
                })

                if i == 0 and not os.path.exists(std_csv_path):
                    std_df.to_csv(std_csv_path, index=False)
                else:
                    std_df.to_csv(std_csv_path, mode='a', header=False, index=False)

        except Exception as e:
            logger.error(f"Error analyzing checkpoint {checkpoint_dir}: {e}")
            continue
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

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
    parser.add_argument("--focus_question_idx", type=int, default=None,
                       help="If provided, analyze only this question index after filtering to max_samples")
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
