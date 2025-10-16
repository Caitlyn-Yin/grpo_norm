# utils.py 
import re
import numpy as np
import logging

pass_at_k_history = []
accuracy_history = []

def print_trainable_parameters(model):
    """Print model trainable parameters information"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")
    print(f"Percentage trainable: {100 * trainable_params / all_params:.2f}%")

def format_reward_func_qa(completions, **kwargs):
    """Check if completion text contains correct answer format"""
    patterns = [
        r"####\s*The final answer is\s*[\-]?\d+",
        r"####\s*[\-]?\d+",
    ]
    
    rewards = []
    for completion in completions:
        reward = 0.0
        for pattern in patterns:
            if completion and re.search(pattern, completion):
                reward = 0.5
                break
        rewards.append(reward)
    
    return rewards

def correctness_reward_func_qa(completions, final_answer=None, **kwargs):
    """
    Check if answers are correct and track Pass@K
    """
    global pass_at_k_history, accuracy_history
    
    if final_answer is None:
        logging.warning("No final_answer provided")
        return [0.0] * len(completions)
    
    if not isinstance(final_answer, list):
        final_answers = [final_answer] * len(completions)
    elif len(final_answer) == 1:
        final_answers = final_answer * len(completions)
    else:
        final_answers = list(final_answer)
    
    rewards = []
    patterns = [
        r'####\s*([\-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'The final answer is[:\s]*([\-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'answer[is:：:\s]*([\-]?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    
    for completion, ground_truth in zip(completions, final_answers):
        reward = 0.0
        
        if completion:
            for pattern in patterns:
                match = re.search(pattern, completion, re.IGNORECASE)
                if match:
                    try:
                        answer_str = match.group(1).replace(',', '').replace('$', '')
                        predicted = float(answer_str)
                        
                        if ground_truth is not None:
                            expected = float(ground_truth)
                            if abs(predicted - expected) < 1e-6:
                                reward = 1.0
                                break
                    except:
                        continue
        
        rewards.append(reward)
    k = 8
    if len(completions) == k:
        pass_at_k = 1.0 if any(r > 0 for r in rewards) else 0.0
        accuracy = sum(rewards) / len(rewards)
        
        pass_at_k_history.append(pass_at_k)
        accuracy_history.append(accuracy)
        
        print(f"\n Current batch - Pass@{k}: {pass_at_k:.2f}, Accuracy: {accuracy:.2%}")
        
        if len(pass_at_k_history) > 0:
            cumulative_pass = np.mean(pass_at_k_history)
            cumulative_acc = np.mean(accuracy_history)
            print(f"Cumulative - Pass@{k}: {cumulative_pass:.4f}, Accuracy: {cumulative_acc:.4f}")
            print(f"   Total questions seen: {len(pass_at_k_history)}")
        
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "train/step_accuracy": accuracy,
                })
        except:
            pass
    
    correct_count = sum(1 for r in rewards if r > 0)
    if correct_count > 0 or len(rewards) > 0:
        logging.info(f"Batch accuracy: {correct_count}/{len(rewards)} = {100*correct_count/len(rewards):.1f}%")
    
    return rewards

def format_reward_func_code(completions, **kwargs):
    """Check if code contains return statement"""
    pattern = r"\n\s*return\s+.+"
    rewards = []
    for completion in completions:
        if completion and re.search(pattern, completion):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def correctness_reward_func_code(completions, final_answer=None, **kwargs):
    """Code correctness check (simplified version)"""
    if final_answer is None:
        return [0.0] * len(completions)
    
    if not isinstance(final_answer, list):
        final_answers = [final_answer] * len(completions)
    elif len(final_answer) == 1:
        final_answers = final_answer * len(completions)
    else:
        final_answers = list(final_answer)
    
    rewards = []
    for completion, ground_truth in zip(completions, final_answers):
        reward = 0.0
        try:
            match = re.search(r'return\s+([\-]?\d+(?:\.\d+)?)', completion)
            if match:
                predicted = float(match.group(1))
                expected = float(ground_truth)
                if abs(predicted - expected) < 1e-6:
                    reward = 1.0
        except:
            pass
        rewards.append(reward)
    
    return rewards

def get_pass_at_k_stats():
    """Get Pass@K statistics"""
    global pass_at_k_history, accuracy_history
    
    if not pass_at_k_history:
        return None
    
    return {
        'cumulative_pass_at_k': np.mean(pass_at_k_history),
        'cumulative_accuracy': np.mean(accuracy_history),
        'recent_pass_at_k': np.mean(pass_at_k_history[-10:]),  # Last 10 questions
        'recent_accuracy': np.mean(accuracy_history[-10:]),
        'total_questions': len(pass_at_k_history),
    }

def reset_metrics():
    """Reset metrics"""
    global pass_at_k_history, accuracy_history
    pass_at_k_history = []
    accuracy_history = []
    print("Metrics reset.")

def print_final_stats():
    """Print final statistics"""
    stats = get_pass_at_k_stats()
    if stats:
        print("\n" + "="*60)
        print("FINAL TRAINING STATISTICS")
        print("="*60)
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Final Pass@K: {stats['cumulative_pass_at_k']:.4f}")
        print(f"Final Accuracy: {stats['cumulative_accuracy']:.4f}")
        print(f"Recent Pass@K (last 10): {stats['recent_pass_at_k']:.4f}")
        print(f"Recent Accuracy (last 10): {stats['recent_accuracy']:.4f}")
        print("="*60)
