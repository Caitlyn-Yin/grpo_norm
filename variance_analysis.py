import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional, Union

class VarianceAnalyzer:
    """
    Analyzes variance differences between questions and iterations,
    as well as cosine similarity between policy gradients.
    """
    
    def __init__(self, save_dir="variance_analysis"):
        """Initialize the analyzer with a directory to save results"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Storage for variance data
        self.question_variances = {}  # {question_id: [variance_iter1, variance_iter2, ...]}
        self.iteration_variances = {}  # {iteration: [variance_q1, variance_q2, ...]}
        
        # Storage for cosine similarity data
        self.question_gradients = {}  # {question_id: {iteration: gradient_vector}}
        self.cosine_similarities = {}  # {(q1, q2): [cos_sim_iter1, cos_sim_iter2, ...]}
        
        # Storage for Fisher information
        self.question_fisher = {}  # {question_id: {iteration: fisher_value}}
        self.question_reward_std = {}  # {question_id: {iteration: reward_std}}
        
    def record_variance(self, question_id: str, iteration: int, variance: float):
        """Record variance for a specific question and iteration"""
        if question_id not in self.question_variances:
            self.question_variances[question_id] = {}
        self.question_variances[question_id][iteration] = variance
        
        if iteration not in self.iteration_variances:
            self.iteration_variances[iteration] = {}
        self.iteration_variances[iteration][question_id] = variance
    
    def record_gradient(self, question_id: str, iteration: int, gradient_vector: torch.Tensor):
        """Record gradient vector for a specific question and iteration"""
        if question_id not in self.question_gradients:
            self.question_gradients[question_id] = {}
        
        # Store as numpy array to save memory
        self.question_gradients[question_id][iteration] = gradient_vector.detach().cpu().numpy()
    
    def record_fisher_info(self, question_id: str, iteration: int, fisher_value: float):
        """Record Fisher information for a specific question and iteration"""
        if question_id not in self.question_fisher:
            self.question_fisher[question_id] = {}
        self.question_fisher[question_id][iteration] = fisher_value
    
    def record_reward_std(self, question_id: str, iteration: int, reward_std: float):
        """Record reward standard deviation for a specific question and iteration"""
        if question_id not in self.question_reward_std:
            self.question_reward_std[question_id] = {}
        self.question_reward_std[question_id][iteration] = reward_std
    
    def compute_cosine_similarities(self):
        """Compute cosine similarities between all pairs of questions for each iteration"""
        question_ids = list(self.question_gradients.keys())
        iterations = set()
        for q_id in question_ids:
            iterations.update(self.question_gradients[q_id].keys())
        iterations = sorted(list(iterations))
        
        for i, q1 in enumerate(question_ids):
            for j in range(i+1, len(question_ids)):
                q2 = question_ids[j]
                pair_key = (q1, q2)
                self.cosine_similarities[pair_key] = {}
                
                for iteration in iterations:
                    if iteration in self.question_gradients[q1] and iteration in self.question_gradients[q2]:
                        g1 = self.question_gradients[q1][iteration]
                        g2 = self.question_gradients[q2][iteration]
                        
                        # Compute cosine similarity
                        cos_sim = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-8)
                        self.cosine_similarities[pair_key][iteration] = cos_sim
    
    def analyze_variance_differences(self):
        """
        Analyze variance differences between questions and iterations
        Returns: (question_variance_diff, iteration_variance_diff)
        """
        # Calculate variance differences between questions
        question_variance_diff = {}
        for iteration, q_vars in self.iteration_variances.items():
            if len(q_vars) >= 2:  # Need at least 2 questions to compare
                q_values = list(q_vars.values())
                q_var = np.var(q_values)
                question_variance_diff[iteration] = q_var
        
        # Calculate variance differences between iterations for each question
        iteration_variance_diff = {}
        for question, iter_vars in self.question_variances.items():
            if len(iter_vars) >= 2:  # Need at least 2 iterations to compare
                iter_values = list(iter_vars.values())
                iter_var = np.var(iter_values)
                iteration_variance_diff[question] = iter_var
        
        return question_variance_diff, iteration_variance_diff
    
    def analyze_cosine_similarity_vs_variance(self):
        """
        Analyze relationship between cosine similarity and variance similarity
        Returns: DataFrame with cosine similarity and variance difference data
        """
        data = []
        
        for (q1, q2), cos_sims in self.cosine_similarities.items():
            for iteration, cos_sim in cos_sims.items():
                if (iteration in self.iteration_variances and 
                    q1 in self.iteration_variances[iteration] and 
                    q2 in self.iteration_variances[iteration]):
                    
                    var1 = self.iteration_variances[iteration][q1]
                    var2 = self.iteration_variances[iteration][q2]
                    var_diff = abs(var1 - var2)
                    var_ratio = max(var1, var2) / (min(var1, var2) + 1e-8)
                    
                    data.append({
                        'question1': q1,
                        'question2': q2,
                        'iteration': iteration,
                        'cosine_similarity': cos_sim,
                        'variance1': var1,
                        'variance2': var2,
                        'variance_diff': var_diff,
                        'variance_ratio': var_ratio
                    })
        
        return pd.DataFrame(data)
    
    def analyze_fisher_vs_reward_std(self):
        """
        Analyze correlation between Fisher information and reward standard deviation
        Returns: DataFrame with Fisher and reward std data
        """
        data = []
        
        for question_id in set(self.question_fisher.keys()) & set(self.question_reward_std.keys()):
            for iteration in set(self.question_fisher[question_id].keys()) & set(self.question_reward_std[question_id].keys()):
                fisher = self.question_fisher[question_id][iteration]
                reward_std = self.question_reward_std[question_id][iteration]
                
                data.append({
                    'question_id': question_id,
                    'iteration': iteration,
                    'fisher_info': fisher,
                    'reward_std': reward_std,
                    'reward_var': reward_std**2
                })
        
        return pd.DataFrame(data)
    
    def plot_cosine_similarity_over_iterations(self):
        """Plot how cosine similarity changes over iterations"""
        plt.figure(figsize=(12, 8))
        
        # Get all iterations
        all_iterations = set()
        for _, cos_sims in self.cosine_similarities.items():
            all_iterations.update(cos_sims.keys())
        iterations = sorted(list(all_iterations))
        
        # Calculate average cosine similarity per iteration
        avg_cos_sims = []
        pos_cos_sims_ratio = []
        
        for iteration in iterations:
            iteration_cos_sims = []
            for _, cos_sims in self.cosine_similarities.items():
                if iteration in cos_sims:
                    iteration_cos_sims.append(cos_sims[iteration])
            
            if iteration_cos_sims:
                avg_cos_sim = np.mean(iteration_cos_sims)
                pos_ratio = np.mean([cs > 0 for cs in iteration_cos_sims])
                
                avg_cos_sims.append(avg_cos_sim)
                pos_cos_sims_ratio.append(pos_ratio)
        
        # Plot average cosine similarity
        plt.subplot(2, 1, 1)
        plt.plot(iterations, avg_cos_sims, 'b-', marker='o')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Average Cosine Similarity')
        plt.title('Average Cosine Similarity Between Questions Over Iterations')
        plt.grid(True, alpha=0.3)
        
        # Plot ratio of positive cosine similarities
        plt.subplot(2, 1, 2)
        plt.plot(iterations, pos_cos_sims_ratio, 'g-', marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Ratio of Positive Cosine Similarities')
        plt.title('Proportion of Question Pairs with Positive Cosine Similarity')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'cosine_similarity_over_iterations.png'))
        plt.close()
    
    def plot_variance_vs_cosine(self, df):
        """Plot variance difference vs cosine similarity"""
        plt.figure(figsize=(10, 8))
        
        # Filter for cosine > 0 and cosine < 0
        df_pos = df[df['cosine_similarity'] > 0]
        df_neg = df[df['cosine_similarity'] <= 0]
        
        plt.scatter(df_pos['cosine_similarity'], df_pos['variance_diff'], 
                   alpha=0.5, label='Cosine > 0', color='blue')
        plt.scatter(df_neg['cosine_similarity'], df_neg['variance_diff'], 
                   alpha=0.5, label='Cosine <= 0', color='red')
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['cosine_similarity'], df['variance_diff'], 1)
            p = np.poly1d(z)
            plt.plot(sorted(df['cosine_similarity']), p(sorted(df['cosine_similarity'])), 
                    "r--", alpha=0.8)
            
            # Calculate correlation
            corr, p_value = pearsonr(df['cosine_similarity'], df['variance_diff'])
            plt.title(f'Variance Difference vs Cosine Similarity\nCorrelation: {corr:.3f} (p={p_value:.3f})')
        
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Variance Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        pos_count = len(df_pos)
        neg_count = len(df_neg)
        total = pos_count + neg_count
        pos_ratio = pos_count / total if total > 0 else 0
        
        stats_text = f"Positive Cosine: {pos_count}/{total} ({pos_ratio:.1%})\n"
        stats_text += f"Negative Cosine: {neg_count}/{total} ({1-pos_ratio:.1%})"
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'variance_vs_cosine.png'))
        plt.close()
    
    def plot_fisher_vs_reward_std(self, df):
        """Plot Fisher information vs reward standard deviation squared"""
        plt.figure(figsize=(10, 8))
        
        plt.scatter(df['fisher_info'], df['reward_var'], alpha=0.5)
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['fisher_info'], df['reward_var'], 1)
            p = np.poly1d(z)
            plt.plot(sorted(df['fisher_info']), p(sorted(df['fisher_info'])), 
                    "r--", alpha=0.8)
            
            # Calculate correlation
            corr, p_value = pearsonr(df['fisher_info'], df['reward_var'])
            plt.title(f'Reward Variance vs Fisher Information\nCorrelation: {corr:.3f} (p={p_value:.3f})')
        
        plt.xlabel('Fisher Information (κ)')
        plt.ylabel('Reward Variance (σ²)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'fisher_vs_reward_var.png'))
        plt.close()
    
    def run_full_analysis(self):
        """Run all analyses and generate plots"""
        print("Running variance difference analysis...")
        q_var_diff, iter_var_diff = self.analyze_variance_differences()
        
        print("Computing cosine similarities...")
        self.compute_cosine_similarities()
        
        print("Analyzing cosine similarity vs variance...")
        cos_var_df = self.analyze_cosine_similarity_vs_variance()
        
        print("Analyzing Fisher information vs reward std...")
        fisher_reward_df = self.analyze_fisher_vs_reward_std()
        
        print("Generating plots...")
        self.plot_cosine_similarity_over_iterations()
        
        if not cos_var_df.empty:
            self.plot_variance_vs_cosine(cos_var_df)
            cos_var_df.to_csv(os.path.join(self.save_dir, 'cosine_variance_data.csv'), index=False)
        
        if not fisher_reward_df.empty:
            self.plot_fisher_vs_reward_std(fisher_reward_df)
            fisher_reward_df.to_csv(os.path.join(self.save_dir, 'fisher_reward_data.csv'), index=False)
        
        # Save summary statistics
        with open(os.path.join(self.save_dir, 'variance_analysis_summary.txt'), 'w') as f:
            f.write("VARIANCE ANALYSIS SUMMARY\n")
            f.write("========================\n\n")
            
            f.write("Question Variance Differences (across iterations):\n")
            for q, var in iter_var_diff.items():
                f.write(f"  Question {q}: {var:.6f}\n")
            f.write(f"  Average: {np.mean(list(iter_var_diff.values())):.6f}\n\n")
            
            f.write("Iteration Variance Differences (across questions):\n")
            for iter_num, var in q_var_diff.items():
                f.write(f"  Iteration {iter_num}: {var:.6f}\n")
            f.write(f"  Average: {np.mean(list(q_var_diff.values())):.6f}\n\n")
            
            if not cos_var_df.empty:
                pos_cos = cos_var_df[cos_var_df['cosine_similarity'] > 0]
                neg_cos = cos_var_df[cos_var_df['cosine_similarity'] <= 0]
                
                f.write("Cosine Similarity Analysis:\n")
                f.write(f"  Total pairs analyzed: {len(cos_var_df)}\n")
                f.write(f"  Pairs with positive cosine: {len(pos_cos)} ({len(pos_cos)/len(cos_var_df):.1%})\n")
                f.write(f"  Pairs with negative cosine: {len(neg_cos)} ({len(neg_cos)/len(cos_var_df):.1%})\n")
                
                if len(pos_cos) > 0:
                    f.write(f"  Avg variance diff for positive cosine: {pos_cos['variance_diff'].mean():.6f}\n")
                if len(neg_cos) > 0:
                    f.write(f"  Avg variance diff for negative cosine: {neg_cos['variance_diff'].mean():.6f}\n")
                
                if len(cos_var_df) > 1:
                    corr, p_value = pearsonr(cos_var_df['cosine_similarity'], cos_var_df['variance_diff'])
                    f.write(f"  Correlation between cosine and variance diff: {corr:.3f} (p={p_value:.3f})\n\n")
            
            if not fisher_reward_df.empty:
                f.write("Fisher Information vs Reward Variance Analysis:\n")
                f.write(f"  Total data points: {len(fisher_reward_df)}\n")
                
                if len(fisher_reward_df) > 1:
                    corr, p_value = pearsonr(fisher_reward_df['fisher_info'], fisher_reward_df['reward_var'])
                    f.write(f"  Correlation between Fisher and reward variance: {corr:.3f} (p={p_value:.3f})\n")
                    
                    # Linear regression coefficients
                    z = np.polyfit(fisher_reward_df['fisher_info'], fisher_reward_df['reward_var'], 1)
                    f.write(f"  Linear fit: σ² = {z[0]:.4f} * κ + {z[1]:.4f}\n")
        
        print(f"Analysis complete. Results saved to {self.save_dir}/")
        return {
            'question_variance_diff': iter_var_diff,
            'iteration_variance_diff': q_var_diff,
            'cosine_variance_df': cos_var_df,
            'fisher_reward_df': fisher_reward_df
        }
