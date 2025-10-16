# train_difficulty.py
import sys
import types
import importlib.util

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
import random
import logging
from typing import Dict, List, Union, Any, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from gsm8k_difficulty import GSM8KDifficulty
from utils import (
    format_reward_func_qa, correctness_reward_func_qa,
    format_reward_func_code, correctness_reward_func_code,
    print_trainable_parameters,
    print_final_stats, get_pass_at_k_stats
)
from variance_analysis import VarianceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# Custom Trainer with curvature, variance, and cosine logs
# =========================================================
class DifficultyGRPOTrainer(GRPOTrainer):

    def __init__(
        self,
        *args,
        difficulty: str = "easy",
        analyze_variance: bool = True,
        log_curvature: bool = False,
        curv_mode: str = "trace",            # 'trace' or 'blockwise'
        lora_only_fisher: bool = True,
        log_cosine: bool = False,
        cosine_pairs: int = 200,
        tokenizer: Optional[AutoTokenizer] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.difficulty = difficulty
        self.step_count = 0

        # tokenizer for building per-prompt LM losses
        self.tokenizer = tokenizer

        # Variance analysis
        self.analyze_variance = analyze_variance
        if analyze_variance:
            self.variance_analyzer = VarianceAnalyzer(save_dir=f"variance_analysis_results_{difficulty}")

        # curvature / cosine flags
        self.log_curvature = log_curvature
        self.curv_mode = curv_mode
        self.lora_only_fisher = lora_only_fisher
        self.log_cosine = log_cosine
        self.cosine_pairs = cosine_pairs

        # storage for this step
        self._per_prompt_records: List[Dict[str, Any]] = []

    # ---------- utilities ----------
    def _iter_trainable_params(self, lora_only: bool = False):
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if lora_only and "lora" not in n.lower():
                continue
            yield n, p

    def _flatten_grads(self, lora_only: bool = False) -> torch.Tensor:
        flats = []
        for _, p in self._iter_trainable_params(lora_only):
            if p.grad is not None:
                flats.append(p.grad.detach().float().reshape(-1))
        if flats:
            return torch.cat(flats)
        device = next(self.model.parameters()).device
        return torch.zeros(0, device=device)

    def _zero_param_grads(self, lora_only: bool = False):
        for _, p in self._iter_trainable_params(lora_only):
            if p.grad is not None:
                p.grad.zero_()

    def _prompt_id(self, prompt_text: str) -> str:
        return f"q::{hash(prompt_text)}"

    # ---------- logging hook ----------
    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        self.step_count += 1
        logs["difficulty"] = self.difficulty

        # Map per-step correctness mean to step accuracy for W&B charts
        if "rewards/correctness_reward_func_qa/mean" in logs:
            try:
                logs["train/step_accuracy"] = float(logs["rewards/correctness_reward_func_qa/mean"])
                # Also log with explicit local step to control W&B x-axis
                if wandb.run is not None:
                    wandb.log({"train/step_accuracy": logs["train/step_accuracy"]}, step=self.step_count)
            except Exception:
                pass

        if hasattr(self.args, 'scale_rewards'):
            if self.args.scale_rewards is True or self.args.scale_rewards == "group":
                logs["normalization"] = "standard"
            elif self.args.scale_rewards is False:
                logs["normalization"] = "no_std"

        # Periodic offline summary from VarianceAnalyzer
        if self.analyze_variance and self.step_count > 50 and self.step_count % 20 == 0:
            try:
                logger.info(f"[step {self.step_count}] Running variance analysis")
                analysis_results = self.variance_analyzer.run_full_analysis()

                if 'cosine_variance_df' in analysis_results and not analysis_results['cosine_variance_df'].empty:
                    df_cos = analysis_results['cosine_variance_df']
                    if wandb.run is not None:
                        logs["analysis/avg_cosine_similarity"] = df_cos['cosine_similarity'].mean()
                        logs["analysis/positive_cosine_ratio"] = (df_cos['cosine_similarity'] > 0).mean()
                        logs["analysis/avg_variance_diff"] = df_cos['variance_diff'].mean()
                        if len(df_cos) > 1:
                            from scipy.stats import pearsonr
                            corr, _ = pearsonr(df_cos['cosine_similarity'], df_cos['variance_diff'])
                            logs["analysis/cosine_variance_corr"] = corr

                if 'question_variance_diff' in analysis_results:
                    qvd = analysis_results['question_variance_diff']
                    if wandb.run is not None and qvd:
                        logs["analysis/avg_question_variance_diff"] = float(np.mean(list(qvd.values())))

                if 'iteration_variance_diff' in analysis_results:
                    ivd = analysis_results['iteration_variance_diff']
                    if wandb.run is not None and ivd:
                        logs["analysis/avg_iteration_variance_diff"] = float(np.mean(list(ivd.values())))

                logs["analysis/difficulty"] = self.difficulty

            except Exception as e:
                logger.warning(f"Variance analysis failed: {e}")

        # Flush collected per-step curvature/cosine correlations to W&B
        if self._per_prompt_records and wandb.run is not None:
            # sigma^2 vs curvature correlation
            sig2 = [r["sigma2"] for r in self._per_prompt_records if r.get("kappa") is not None]
            kapp = [r["kappa"] for r in self._per_prompt_records if r.get("kappa") is not None]
            if len(sig2) >= 3:
                try:
                    import scipy.stats as ss
                    spear = ss.spearmanr(sig2, kapp).correlation
                    kend = ss.kendalltau(sig2, kapp).correlation
                except Exception:
                    spear = kend = None
            else:
                spear = kend = None

            # cosine stats
            cos_vals = [c for r in self._per_prompt_records for c in r.get("cosines", [])]
            prop_neg = float(np.mean([c < 0.0 for c in cos_vals])) if cos_vals else None
            pos_var_diffs = [d for r in self._per_prompt_records for d in r.get("pos_var_diffs", [])]
            med_abs_diff_poscos = float(np.median(pos_var_diffs)) if pos_var_diffs else None

            wandb.log({
                'curvature/spearman_sigma2_vs_kappa': spear,
                'curvature/kendall_sigma2_vs_kappa': kend,
                'curvature/mean_sigma2': float(np.mean(sig2)) if sig2 else None,
                'curvature/mean_kappa': float(np.mean(kapp)) if kapp else None,
                'cosine/prop_negative': prop_neg,
                'cosine/median_abs_var_diff_given_cos_pos': med_abs_diff_poscos,
                'cosine/count_pairs': len(cos_vals) if cos_vals else 0,
            }, step=self.step_count)

            # clear the cache
            self._per_prompt_records = []

        super().log(logs, start_time)

    # ---------- reward computation ----------
    def compute_rewards(self, completions: List[str], prompts: List[str], **kwargs):
        """
        Compute rewards and record per-prompt reward std / variance.
        """
        rewards = super().compute_rewards(completions, prompts, **kwargs)

        if self.analyze_variance:
            # estimate per-prompt pi_hat from K generations in this group
            # super().compute_rewards returns flat list aligned with completions
            # Here we regroup by identical prompt text.
            from collections import defaultdict
            by_prompt = defaultdict(list)
            for r, p in zip(rewards, prompts):
                by_prompt[p].append(float(r))

            per_step_records = []
            for ptxt, rs in by_prompt.items():
                qid = self._prompt_id(ptxt)
                if len(rs) >= 2:
                    mu = float(np.mean(rs))
                    std = float(np.std(rs))
                    var = std * std
                    # track in analyzer (for across-iteration stats)
                    self.variance_analyzer.record_reward_std(qid, self.step_count, std)
                    self.variance_analyzer.record_variance(qid, self.step_count, var)
                    per_step_records.append((qid, var))
                else:
                    per_step_records.append((qid, 0.0))

            # stash sigma2 for later correlations (curvature, cosine)
            # initial record with sigma2 only; fill curvature/cos later
            for qid, var in per_step_records:
                self._per_prompt_records.append({"qid": qid, "sigma2": var})

        return rewards

    # ---------- curvature (empirical Fisher, method 2) + cosine ----------
    def _build_teacher_forcing_batch(self, prompt_text: str, completion_text: str):
        """
        Build LM training batch from (prompt, generated completion):
        input = prompt + completion[:-1]
        labels = [-100]*len(prompt) + completion_tokens   (standard causal LM)
        """
        assert self.tokenizer is not None, "tokenizer is required for curvature logging"
        device = next(self.model.parameters()).device

        # tokenize prompt and completion separately for clarity
        pt = self.tokenizer(prompt_text, add_special_tokens=False)
        ct = self.tokenizer(completion_text + self.tokenizer.eos_token, add_special_tokens=False)

        input_ids = torch.tensor([pt["input_ids"] + ct["input_ids"][:-1]], device=device)
        labels    = torch.tensor([[-100]*len(pt["input_ids"]) + ct["input_ids"]], device=device)

        attention_mask = torch.ones_like(input_ids, device=device)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _empirical_fisher_for_prompt(
        self,
        prompt_text: str,
        completion_text: str,
    ) -> Dict[str, Any]:
        """
        Compute empirical diagonal Fisher for one prompt using model's own completion as pseudo-target.
        Returns: {'kappa': scalar curvature proxy, 'grad_vec': flattened grad vector (optional), 'Ti': token count}
        """
        if not self.log_curvature and not self.log_cosine:
            return {"kappa": None, "grad_vec": None, "Ti": 0}

        try:
            batch = self._build_teacher_forcing_batch(prompt_text, completion_text)
            # Forward: standard causal LM loss (averaged over non -100 labels)
            outputs = self.model(**batch)
            loss = outputs.loss  # already mean over tokens used

            # Zero grads (LoRA-only if requested)
            self._zero_param_grads(lora_only=self.lora_only_fisher)

            # Backward to get ∇θ L_i
            loss.backward(retain_graph=False)

            # token count used in the loss (labels != -100)
            Ti = int((batch["labels"] != -100).sum().item())

            # diag Fisher per param ≈ Ti * (grad^2), reduce to scalar (trace or blockwise)
            trace_sum = 0.0
            blockwise = {}
            for name, p in self._iter_trainable_params(self.lora_only_fisher):
                if p.grad is None:
                    continue
                g = p.grad.detach()
                contrib = (Ti * g * g).sum().item()   # scalar
                trace_sum += contrib
                if self.curv_mode == "blockwise":
                    # group by top-level module name as a simple block (customize if needed)
                    key = name.split('.')[0]
                    blockwise[key] = blockwise.get(key, 0.0) + contrib

            if self.curv_mode == "trace":
                kappa = trace_sum
            else:
                kappa = float(np.mean(list(blockwise.values()))) if blockwise else 0.0

            grad_vec = None
            if self.log_cosine:
                grad_vec = self._flatten_grads(lora_only=self.lora_only_fisher)

            # Clear grads for next prompt
            self._zero_param_grads(lora_only=self.lora_only_fisher)

            return {"kappa": kappa, "grad_vec": grad_vec, "Ti": Ti}

        except Exception as e:
            logger.warning(f"Empirical Fisher failed for a prompt: {e}")
            # try to clear grads to keep training safe
            self._zero_param_grads(lora_only=self.lora_only_fisher)
            return {"kappa": None, "grad_vec": None, "Ti": 0}

    def on_after_rewards(self, prompts: List[str], completions: List[str]):
        """
        OPTIONAL helper: call this after rewards are computed in your training loop.
        If your GRPOTrainer version doesn't expose such a hook, you can invoke this
        manually from wherever you have both prompts and completions available.
        """
        if not (self.log_curvature or self.log_cosine or self.analyze_variance):
            return

        # Group completions by prompt (per-prompt we’ll pick one completion for curvature;
        # you can also average over multiple completions if you want smoother estimates)
        from collections import defaultdict
        by_prompt_c = defaultdict(list)
        for p, c in zip(prompts, completions):
            by_prompt_c[p].append(c)

        # Compute curvature (empirical Fisher) and prepare cosine inputs
        grad_vectors = []
        tmp_records = []
        for ptxt, clist in by_prompt_c.items():
            qid = self._prompt_id(ptxt)
            # pick the first completion to define pseudo-targets (or use best-rewarded one)
            completion = clist[0]
            res = self._empirical_fisher_for_prompt(ptxt, completion)
            # match with the sigma2 we recorded earlier in compute_rewards
            sigma2 = None
            for rec in self._per_prompt_records:
                if rec.get("qid") == qid and "sigma2" in rec:
                    sigma2 = rec["sigma2"]
                    break
            tmp_records.append({
                "qid": qid,
                "sigma2": sigma2,
                "kappa": res["kappa"],
                "grad_vec": res["grad_vec"]
            })
            if res["grad_vec"] is not None and res["grad_vec"].numel() > 0:
                grad_vectors.append((qid, res["grad_vec"], sigma2))

        # Cosine stats among random pairs within this batch
        if self.log_cosine and len(grad_vectors) >= 2:
            import random
            cosines, pos_var_diffs = [], []
            indices = list(range(len(grad_vectors)))
            pair_budget = min(self.cosine_pairs, len(indices) * (len(indices)-1) // 2)
            seen = set()
            for _ in range(pair_budget):
                i, j = sorted(random.sample(indices, 2))
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                qid_i, gi, vi = grad_vectors[i]
                qid_j, gj, vj = grad_vectors[j]
                denom = (gi.norm() + 1e-12) * (gj.norm() + 1e-12)
                cos = torch.dot(gi, gj) / denom
                cos = float(cos.item())
                cosines.append(cos)
                if cos > 0 and (vi is not None and vj is not None):
                    pos_var_diffs.append(abs(vi - vj))

            # write back into records for wandb logging in .log()
            for rec in tmp_records:
                rec["cosines"] = cosines
                rec["pos_var_diffs"] = pos_var_diffs

        # append to step cache (consumed in .log())
        self._per_prompt_records.extend(tmp_records)

    # ---------- override compute_loss to be safe with grads ----------
    def compute_loss(self, model, inputs: Dict[str, Any], return_outputs=False, **kwargs):
        """
        Use parent compute_loss for GRPO. We leave curvature to the explicit method
        (_empirical_fisher_for_prompt) to avoid interfering with RL grads.
        """
        loss = super().compute_loss(model, inputs, **kwargs)
        if return_outputs:
            from types import SimpleNamespace
            return SimpleNamespace(loss=loss)
        return loss


# -------------------------------
# CLI
# -------------------------------
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
                        choices=['standard', 'no_std'])

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='iclr01_easy')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='default')

    # NEW: analysis/curvature/cosine flags
    parser.add_argument('--analyze_variance', action='store_true', help='Track per-prompt reward variance over time')
    parser.add_argument('--log_curvature', action='store_true', help='Estimate per-prompt empirical Fisher (method 2)')
    parser.add_argument('--curv_mode', type=str, default='trace', choices=['trace', 'blockwise'],
                        help='Reduce diag-Fisher to scalar per prompt')
    parser.add_argument('--lora_only_fisher', action='store_true', help='Restrict Fisher to LoRA params')
    parser.add_argument('--log_cosine', action='store_true', help='Log gradient cosine similarities across prompts')
    parser.add_argument('--cosine_pairs', type=int, default=200, help='Random pairs per step for cosine stats')

    return parser.parse_args()


def _set_deterministic_seeds(seed: int = 42):
    """Set Python/NumPy/PyTorch seeds and enable deterministic behavior."""
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic kernels where possible
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _get_first_available(ex: Dict[str, Any], candidates: List[str]) -> Any:
    for key in candidates:
        if key in ex:
            return ex[key]
    raise KeyError(f"None of the keys {candidates} found in example: {list(ex.keys())}")


def log_step0_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    reward_func,
    k: int,
    max_new_tokens: int,
    n_eval: int = 64,
    seed: int = 42,
):
    """
    Compute and log a deterministic step-0 baseline accuracy/pass@k so that
    runs with different normalization settings share the same initial point.

    Uses greedy decoding (do_sample=False) for determinism.
    """
    _set_deterministic_seeds(seed)
    model.eval()

    num_items = min(n_eval, len(dataset))
    prompts: List[str] = []
    final_answers: List[Any] = []

    for i in range(num_items):
        ex = dataset[i]
        # Robustly fetch prompt and final answer
        prompt_text = _get_first_available(ex, ["prompt", "input", "question", "query"])
        answer_val = _get_first_available(ex, ["final_answer", "answer", "label"]) if "final_answer" in ex or "answer" in ex or "label" in ex else None
        prompts.append(prompt_text)
        final_answers.append(answer_val)

    completions: List[str] = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        for _ in range(k):
            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Keep only the continuation, not the prompt
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(text)

    # If no answers are available, skip logging
    if all(a is None for a in final_answers):
        print("Step 0 baseline skipped: no ground-truth answers present in dataset items.")
        return

    rewards = reward_func(completions, final_answer=final_answers * k)

    # Compute pass@k by grouping per prompt
    per_q_pass = []
    for i in range(num_items):
        start = i * k
        end = start + k
        per_q_pass.append(1.0 if any(r > 0 for r in rewards[start:end]) else 0.0)
    pass_at_k = float(np.mean(per_q_pass)) if per_q_pass else 0.0

    # Sample-level accuracy across all generated samples
    accuracy = float(np.mean(rewards)) if rewards else 0.0

    if wandb.run is not None:
        wandb.log({
            "baseline/accuracy": accuracy,
            "baseline/pass_at_k": pass_at_k,
            "train/step_accuracy": accuracy,
        }, step=0)

    print(f"Step 0 baseline - Acc: {accuracy:.4f}, Pass@{k}: {pass_at_k:.4f}")


def main():
    args = parse_args()
    print(f"Arguments: {args}")
    print(f"Training on {args.difficulty} difficulty dataset")

    # Ensure identical initialization across runs (normalization vs no normalization)
    _set_deterministic_seeds(42)

    # Setup WandB
    if args.use_wandb:
        run_name = args.wandb_run_name or f"grpo-efr_difficulty-{args.normalization}-{args.difficulty}-{args.exp_name}"
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

    # Load dataset
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
    output_dir = f'outputs/efr_difficulty/{args.difficulty}/{args.normalization}/{model_name.split("/")[-1]}/{args.exp_name}'

    if args.normalization == 'standard':
        scale_rewards = True
    elif args.normalization == 'no_std':
        scale_rewards = False
    else:
        raise ValueError(f"Unknown normalization: {args.normalization}")

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f'GRPO_efr_difficulty-{args.normalization}-{args.difficulty}-{args.exp_name}',
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
        log_on_each_node=False,
        seed=42,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        bias='none',
        lora_dropout=0.05,
    )

    # Model & tokenizer
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

    # Rewards
    if args.format == 'qa':
        reward_funcs = [format_reward_func_qa, correctness_reward_func_qa]
    elif args.format == 'code':
        reward_funcs = [format_reward_func_code, correctness_reward_func_code]
    else:
        raise ValueError("Unknown format")

    # Log a deterministic step-0 baseline before any training updates
    try:
        baseline_reward_func = correctness_reward_func_qa if args.format == 'qa' else correctness_reward_func_code
        log_step0_baseline(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            reward_func=baseline_reward_func,
            k=args.num_generations,
            max_new_tokens=training_args.max_completion_length,
            n_eval=min(64, len(dataset)),
            seed=42,
        )
    except Exception as e:
        print(f"Warning: step-0 baseline logging failed: {e}")

    trainer = DifficultyGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        difficulty=args.difficulty,

        # NEW toggles
        analyze_variance=args.analyze_variance,
        log_curvature=args.log_curvature,
        curv_mode=args.curv_mode,
        lora_only_fisher=args.lora_only_fisher,
        log_cosine=args.log_cosine,
        cosine_pairs=args.cosine_pairs,

        tokenizer=tokenizer,
    )

    print(f"\nStarting training:")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Normalization: {args.normalization}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  scale_rewards setting: {scale_rewards}")

    trainer.train()

    # Save model
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
