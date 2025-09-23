#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, math
import os, glob, json, math, random, argparse, csv
from typing import List, Dict, Any, Optional, Tuple
from peft import AutoPeftModelForCausalLM
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import (
    format_reward_func_qa, correctness_reward_func_qa,
    format_reward_func_code, correctness_reward_func_code,
)

# Fallback to a plain text/JSONL file.
def load_prompts(args) -> List[Dict[str, Any]]:
    if args.prompts_file:
        ext = os.path.splitext(args.prompts_file)[1].lower()
        rows = []
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            if ext in ['.jsonl', '.jl']:
                for line in f:
                    o = json.loads(line)
                    rows.append(o)
            else:
                # plain text: one prompt per line
                for line in f:
                    rows.append({"prompt": line.strip()})
        return rows[:args.eval_prompts]
    # Try dataset classes if available
    try:
        if args.dataset == "math":
            from math_datasets import MATHDataset
            ds = MATHDataset(
                split='train',
                include_answer=True,
                include_reasoning=True,
                few_shot=True,
                num_shots=args.num_shots,
                seed=42,
                cot=True,
                template=args.format,
                subjects=None,  # all subjects
            ).dataset
            # The dataset items usually have 'question' or the formatted 'prompt'.
            items = []
            for x in ds.select(range(min(args.eval_prompts, len(ds)))):
                # prefer the already-formatted input if present
                if 'prompt' in x:
                    items.append({'prompt': x['prompt']})
                elif 'question' in x:
                    items.append({'prompt': x['question']})
                else:
                    # best effort merge common fields
                    text = x.get('question', '') + "\n" + x.get('context', '')
                    items.append({'prompt': text.strip()})
            return items
        elif args.dataset == "gsm8k":
            from gsm8k_difficulty import GSM8KDifficulty
            ds = GSM8KDifficulty(
                difficulty=args.difficulty, data_dir=args.data_dir,
                split='train', include_answer=False, include_reasoning=True,
                few_shot=True, num_shots=args.num_shots, seed=42, cot=True,
                template=args.format, max_samples=args.eval_prompts
            ).dataset
            return [{'prompt': x.get('prompt', x.get('question', ''))} for x in ds]
    except Exception as e:
        print(f"[WARN] Falling back to a simple prompts file. Error: {e}")

    raise ValueError("No prompts source. Provide --prompts_file or install dataset loaders.")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_name: str, ckpt: Optional[str], bf16: bool):
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float16

    # If the checkpoint is a PEFT adapter dir (has adapter_model.safetensors),
    # AutoPeftModelForCausalLM will:
    #  - read adapter_config.json
    #  - load the base model it references
    #  - attach LoRA modules
    model = AutoPeftModelForCausalLM.from_pretrained(
        ckpt or model_name,
        torch_dtype=dtype,
        device_map=None,                 # keep manual control
        low_cpu_mem_usage=True,
    )

    # Ensure LoRA params require grad; freeze the rest to save memory
    for n, p in model.named_parameters():
        if "lora" in n.lower():
            p.requires_grad = True
        else:
            p.requires_grad = False

    tok = AutoTokenizer.from_pretrained(ckpt or model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id

    # We'll switch to train() just for the Fisher backward.
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    return model, tok

@torch.no_grad()
def generate_k(model, tok, prompt_text: str, k: int, gen_kwargs: Dict[str, Any]) -> List[str]:
    """Sample K completions for one prompt with fixed decoding cfg."""
    inputs = tok([prompt_text], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        num_return_sequences=k,
        **gen_kwargs
    )
    texts = tok.batch_decode(out, skip_special_tokens=True)
    # strip the prompt prefix if needed
    # (best effort: keep only the suffix after the original prompt length)
    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for seq in out.view(k, -1):
        comp_ids = seq[prompt_len:]
        comps = tok.decode(comp_ids, skip_special_tokens=True)
        completions.append(comps)
    return completions

def _canon(s: str) -> str:
    s = s.strip()
    # extract \boxed{...} if present
    m = re.search(r'\\boxed\{([^{}]+)\}', s)
    if m: s = m.group(1)
    # remove commas and spaces
    s = s.replace(',', '').strip()
    # LaTeX common wraps
    s = s.replace(r'\,', '').replace(r'\!', '')
    return s

def _try_float(s: str):
    try:
        return float(s)
    except Exception:
        # try simple fractions like a/b
        m = re.fullmatch(r'\s*(-?\d+)\s*/\s*(-?\d+)\s*', s)
        if m:
            b = float(m.group(1)); c = float(m.group(2))
            if c != 0: return b / c
        return None

def _match_math(pred: str, gt: str, tol=1e-4) -> bool:
    p = _canon(pred)
    g = _canon(gt)
    if not p or not g: return False
    # numeric compare if both parse
    pf, gf = _try_float(p), _try_float(g)
    if pf is not None and gf is not None:
        return math.isfinite(pf) and math.isfinite(gf) and abs(pf - gf) <= tol*max(1.0, abs(gf))
    # fallback: case-insensitive exact after canonicalization
    return p.lower() == g.lower()

def compute_rewards_for_completions(completions, prompt_text, fmt, gt_answer=None):
    rewards = []
    for c in completions:
        ok = _match_math(c, gt_answer or "")
        rewards.append(1.0 if ok else 0.0)
    return rewards

def teacher_forcing_batch(tok, prompt_text: str, completion_text: str, device):
    """
    Build a standard causal-LM batch:
      input_ids = prompt + completion
      labels    = [-100]*len(prompt) + completion
    HF will do the 1-token shift internally, so lengths must match.
    """
    # tokenize separately
    pt = tok(prompt_text, add_special_tokens=False)
    # include eos so the model is asked to predict it
    ct = tok(completion_text + (tok.eos_token or ""), add_special_tokens=False)

    prompt_ids = pt["input_ids"]
    comp_ids   = ct["input_ids"]

    input_ids = torch.tensor([prompt_ids + comp_ids], device=device)
    labels    = torch.tensor([[-100]*len(prompt_ids) + comp_ids], device=device)
    attn_mask = torch.ones_like(input_ids, device=device)

    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

def empirical_fisher_trace(model, tok, prompt_text, completion_text, lora_only=True):
    device = next(model.parameters()).device
    batch = teacher_forcing_batch(tok, prompt_text, completion_text, device)

    was_training = model.training
    model.train()                      # ensure grad graph is built

    outputs = model(**batch)
    loss = outputs.loss                # mean over supervised tokens

    # zero only the params we’ll use
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if lora_only and "lora" not in n.lower():
            continue
        if p.grad is not None:
            p.grad.zero_()

    loss.backward()                    # <- now grads exist

    Ti = int((batch["labels"] != -100).sum().item())
    trace_sum = 0.0
    flats = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if lora_only and "lora" not in n.lower():
            continue
        if p.grad is None:
            continue
        g = p.grad.detach()
        trace_sum += (Ti * g * g).sum().item()
        flats.append(g.float().reshape(-1))
    grad_vec = torch.cat(flats) if flats else torch.zeros(0, device=device)

    # clean up + restore mode
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if lora_only and "lora" not in n.lower():
            continue
        if p.grad is not None:
            p.grad.zero_()

    if not was_training:
        model.eval()

    return float(trace_sum), grad_vec, Ti


def pairwise_cosine_stats(grad_vecs: List[torch.Tensor], sig2: List[float], max_pairs: int = 200):
    idxs = list(range(len(grad_vecs)))
    pairs = []
    seen = set()
    while len(pairs) < min(max_pairs, len(idxs)*(len(idxs)-1)//2):
        i, j = sorted(random.sample(idxs, 2))
        if (i, j) in seen: continue
        seen.add((i, j)); pairs.append((i, j))
    cos_vals = []; pos_var_diffs = []
    for i, j in pairs:
        gi, gj = grad_vecs[i], grad_vecs[j]
        if gi.numel() == 0 or gj.numel() == 0: continue
        cos = torch.dot(gi, gj) / ((gi.norm()+1e-12)*(gj.norm()+1e-12))
        cv = float(cos.item())
        cos_vals.append(cv)
        if cv > 0.0:
            pos_var_diffs.append(abs(sig2[i]-sig2[j]))
    prop_neg = float(np.mean([c < 0.0 for c in cos_vals])) if cos_vals else None
    med_abs_diff_poscos = float(np.median(pos_var_diffs)) if pos_var_diffs else None
    return cos_vals, prop_neg, med_abs_diff_poscos

def spearman_kendall(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[float]]:
    try:
        import scipy.stats as ss
        s = ss.spearmanr(x, y).correlation
        k = ss.kendalltau(x, y).correlation
        return float(s), float(k)
    except Exception:
        return None, None

def analyze_checkpoint(
    ckpt_path: str,
    model_name: str,
    bf16: bool,
    prompts: List[Dict[str, Any]],
    fmt: str,
    K: int,
    gen_kwargs: Dict[str, Any],
    lora_only: bool,
    cosine_pairs: int,
    out_dir: str,
    seed: int
):
    print(f"\n=== Analyzing checkpoint: {ckpt_path} ===")
    set_seed(seed)
    model, tok = load_model_and_tokenizer(model_name, ckpt_path, bf16)

    # Select a fixed completion to define pseudo-targets for curvature (use first sample)
    per_prompt_rows = []
    grad_vecs = []
    sig2s = []
    kappas = []

    for idx, row in enumerate(prompts):
        prompt_text = row['prompt']
        gt_answer = row.get('answer', '')
        # 1) Generate K completions & rewards
        completions = generate_k(model, tok, prompt_text, K, gen_kwargs)
        rewards = compute_rewards_for_completions(completions, prompt_text, fmt)
        pi_hat = float(np.mean(rewards))
        sigma2 = float(np.var(rewards))
        sig2s.append(sigma2)

        # 2) empirical Fisher (method 2) using the first completion as pseudo-target
        completion_text = completions[0]
        kappa, grad_vec, Ti = empirical_fisher_trace(model, tok, prompt_text, completion_text, lora_only=lora_only)
        grad_vecs.append(grad_vec.detach().cpu())
        kappas.append(kappa)

        per_prompt_rows.append({
            "checkpoint": ckpt_path,
            "prompt_index": idx,
            "prompt_hash": hash(prompt_text),
            "pi_hat": pi_hat,
            "sigma2": sigma2,
            "kappa": kappa,
            "Ti": Ti,
        })

    # 3) Correlations
    spear, kend = spearman_kendall(sig2s, kappas)

    # 4) Cosine stats
    cos_vals, prop_neg, med_abs_diff_poscos = pairwise_cosine_stats(grad_vecs, sig2s, max_pairs=cosine_pairs)

    # 5) bar_sigma (ingredient of C(n,T))
    bar_sigma = float(np.mean([math.sqrt(max(0.0, s)) for s in sig2s])) if sig2s else None

    # 6) Save CSVs
    os.makedirs(out_dir, exist_ok=True)
    # per-prompt file
    pp_csv = os.path.join(out_dir, f"per_prompt_{os.path.basename(ckpt_path).replace('/', '_')}.csv")
    with open(pp_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_prompt_rows[0].keys()))
        w.writeheader(); w.writerows(per_prompt_rows)

    # summary row
    summary_row = {
        "checkpoint": ckpt_path,
        "num_prompts": len(prompts),
        "K": K,
        "spearman_sigma2_kappa": spear,
        "kendall_sigma2_kappa": kend,
        "bar_sigma": bar_sigma,
        "cosine_prop_negative": prop_neg,
        "cosine_median_abs_var_diff_given_pos": med_abs_diff_poscos,
        "cosine_count_pairs": len(cos_vals) if cos_vals else 0,
    }
    sum_csv = os.path.join(out_dir, "summary.csv")
    write_header = not os.path.exists(sum_csv)
    with open(sum_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_header: w.writeheader()
        w.writerow(summary_row)

    print(f"Saved per-prompt CSV  -> {pp_csv}")
    print(f"Appended summary row  -> {sum_csv}")
    return pp_csv, sum_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_glob", type=str, required=True,
                    help="Glob for checkpoints, e.g. 'outputs/.../checkpoint-*'")
    ap.add_argument("--model_name", type=str, default=None,
                    help="(Optional) base model name; if None, we load from checkpoint path")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--dataset", type=str, default="math", choices=["math","gsm8k","file"])
    ap.add_argument("--prompts_file", type=str, default=None,
                    help="If provided, read prompts from this txt|jsonl file (one per line or jsonl with 'prompt')")
    ap.add_argument("--format", type=str, default="qa", choices=["qa","code"])
    ap.add_argument("--difficulty", type=str, default="easy",
                    help="used only if dataset=gsm8k")
    ap.add_argument("--data_dir", type=str, default="data/gsm8k_difficulty_subsets")
    ap.add_argument("--num_shots", type=int, default=2)
    ap.add_argument("--eval_prompts", type=int, default=256, help="number of prompts to analyze per checkpoint")
    ap.add_argument("--K", type=int, default=8, help="samples per prompt")
    # decoding config
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--lora_only_fisher", action="store_true")
    ap.add_argument("--cosine_pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", type=str, default="analysis_outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    prompts = load_prompts(args)
    gen_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=0,
    )

    ckpts = sorted(glob.glob(args.checkpoints_glob))
    if not ckpts:
        raise SystemExit(f"No checkpoints matched: {args.checkpoints_glob}")

    for ck in ckpts:
        analyze_checkpoint(
            ckpt_path=ck,
            model_name=args.model_name or ck,
            bf16=args.bf16,
            prompts=prompts,
            fmt=args.format,
            K=args.K,
            gen_kwargs=gen_kwargs,
            lora_only=args.lora_only_fisher,
            cosine_pairs=args.cosine_pairs,
            out_dir=args.out_dir,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
