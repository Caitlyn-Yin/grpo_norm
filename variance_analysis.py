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
        self.question_variances = {}  # {question_id: [variance_iter1, variance_iter2, ...]}
        self.iteration_variances = {}  # {iteration: [variance_q1, variance_q2, ...]}
        self.question_gradients = {}  # {question_id: {iteration: gradient_vector}}
        self.cosine_similarities = {}  # {(q1, q2): [cos_sim_iter1, cos_sim_iter2, ...]}
        self.question_fisher = {}  # {question_id: {iteration: fisher_value}}
        self.question_reward_std = {}  # {question_id: {iteration: reward_std}}
        self.question_accuracy = {}  # {question_id: {iteration: accuracy}}

    def record_variance(self, question_id: str, iteration: int, variance: float):
        """Record variance for a specific question and iteration"""
        if question_id not in self.question_variances:
            self.question_variances[question_id] = {}
        self.question_variances[question_id][iteration] = variance

        if iteration not in self.iteration_variances:
            self.iteration_variances[iteration] = {}
        self.iteration_variances[iteration][question_id] = variance

    def record_gradient(self, question_id: str, iteration: int, gradient_vector):
        """Record gradient vector for a specific question and iteration.
        Accepts either a torch.Tensor or a numpy.ndarray."""
        if question_id not in self.question_gradients:
            self.question_gradients[question_id] = {}

        try:
            import torch as _torch  # local import to avoid hard dependency at import time
            if _torch.is_tensor(gradient_vector):
                gradient_array = gradient_vector.detach().cpu().numpy()
            else:
                gradient_array = np.asarray(gradient_vector)
        except Exception:
            # Fallback: best-effort numpy conversion
            gradient_array = np.asarray(gradient_vector)

        self.question_gradients[question_id][iteration] = gradient_array

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

    def record_accuracy(self, question_id: str, iteration: int, accuracy: float):
        """Record accuracy for a specific question and iteration"""
        if question_id not in self.question_accuracy:
            self.question_accuracy[question_id] = {}
        self.question_accuracy[question_id][iteration] = accuracy

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
        Returns: DataFrame with Fisher and reward std data aggregated across all iterations
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

    def analyze_fisher_vs_reward_std_by_iteration(self):
        """
        Analyze Fisher information vs reward std at specific test times (iterations).
        Returns: DataFrame with per-iteration correlations and summary statistics
        """
        # Get all iterations that have both Fisher info and reward std data
        all_iterations = set()
        for question_id in set(self.question_fisher.keys()) & set(self.question_reward_std.keys()):
            fisher_iterations = set(self.question_fisher[question_id].keys())
            reward_iterations = set(self.question_reward_std[question_id].keys())
            all_iterations.update(fisher_iterations & reward_iterations)
        
        iterations = sorted(list(all_iterations))
        
        iteration_data = []
        correlation_data = []
        
        for iteration in iterations:
            fisher_values = []
            reward_std_values = []
            question_ids = []
            
            # Collect Fisher info and reward std for this iteration
            for question_id in set(self.question_fisher.keys()) & set(self.question_reward_std.keys()):
                if iteration in self.question_fisher[question_id] and iteration in self.question_reward_std[question_id]:
                    fisher = self.question_fisher[question_id][iteration]
                    reward_std = self.question_reward_std[question_id][iteration]
                    
                    fisher_values.append(fisher)
                    reward_std_values.append(reward_std)
                    question_ids.append(question_id)
                    
                    iteration_data.append({
                        'iteration': iteration,
                        'question_id': question_id,
                        'fisher_info': fisher,
                        'reward_std': reward_std,
                        'reward_var': reward_std**2
                    })
            
            # Calculate correlation for this iteration
            if len(fisher_values) > 1:
                corr, p_value = pearsonr(fisher_values, reward_std_values)
                correlation_data.append({
                    'iteration': iteration,
                    'correlation': corr,
                    'p_value': p_value,
                    'n_samples': len(fisher_values),
                    'mean_fisher': np.mean(fisher_values),
                    'std_fisher': np.std(fisher_values),
                    'mean_reward_std': np.mean(reward_std_values),
                    'std_reward_std': np.std(reward_std_values)
                })
        
        return pd.DataFrame(iteration_data), pd.DataFrame(correlation_data)

    def analyze_cross_time_correlations(self):
        """
        Analyze correlations between Fisher information at time step i and reward variance at time step j
        where i ≠ j, to show they are uncorrelated (as expected theoretically).
        Returns: DataFrame with cross-time correlations
        """
        # Get all iterations that have both Fisher info and reward std data
        all_iterations = set()
        for question_id in set(self.question_fisher.keys()) & set(self.question_reward_std.keys()):
            fisher_iterations = set(self.question_fisher[question_id].keys())
            reward_iterations = set(self.question_reward_std[question_id].keys())
            all_iterations.update(fisher_iterations & reward_iterations)
        
        iterations = sorted(list(all_iterations))
        
        if len(iterations) < 2:
            print("Need at least 2 iterations for cross-time correlation analysis")
            return pd.DataFrame()
        
        cross_correlations = []
        
        # For each pair of different time steps (i, j) where i ≠ j
        for i, iter_i in enumerate(iterations):
            for j, iter_j in enumerate(iterations):
                if i >= j:  # Skip same time step and avoid duplicates
                    continue
                
                fisher_values_i = []
                reward_std_values_j = []
                question_ids = []
                
                # Collect Fisher info at time i and reward std at time j
                for question_id in set(self.question_fisher.keys()) & set(self.question_reward_std.keys()):
                    if (iter_i in self.question_fisher[question_id] and 
                        iter_j in self.question_reward_std[question_id]):
                        
                        fisher_i = self.question_fisher[question_id][iter_i]
                        reward_std_j = self.question_reward_std[question_id][iter_j]
                        
                        fisher_values_i.append(fisher_i)
                        reward_std_values_j.append(reward_std_j)
                        question_ids.append(question_id)
                
                # Calculate correlation between Fisher(i) and reward_std(j)
                if len(fisher_values_i) > 1:
                    corr, p_value = pearsonr(fisher_values_i, reward_std_values_j)
                    cross_correlations.append({
                        'time_i': iter_i,
                        'time_j': iter_j,
                        'time_diff': iter_j - iter_i,
                        'correlation': corr,
                        'p_value': p_value,
                        'n_samples': len(fisher_values_i),
                        'mean_fisher_i': np.mean(fisher_values_i),
                        'mean_reward_std_j': np.mean(reward_std_values_j),
                        'std_fisher_i': np.std(fisher_values_i),
                        'std_reward_std_j': np.std(reward_std_values_j)
                    })
        
        return pd.DataFrame(cross_correlations)

    def get_accuracy_dataframe(self) -> pd.DataFrame:
        """Return accuracy measurements as a DataFrame"""
        data = []

        for question_id, iter_accs in self.question_accuracy.items():
            for iteration, accuracy in iter_accs.items():
                data.append({
                    'question_id': question_id,
                    'iteration': iteration,
                    'accuracy': accuracy
                })

        return pd.DataFrame(data)

    def plot_cosine_similarity_over_iterations(self):
        """Plot how cosine similarity changes over iterations"""
        plt.figure(figsize=(12, 8))

        all_iterations = set()
        for _, cos_sims in self.cosine_similarities.items():
            all_iterations.update(cos_sims.keys())
        iterations = sorted(list(all_iterations))

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

        df_pos = df[df['cosine_similarity'] > 0]
        df_neg = df[df['cosine_similarity'] <= 0]

        plt.scatter(df_pos['cosine_similarity'], df_pos['variance_diff'],
                   alpha=0.5, label='Cosine > 0', color='blue')
        plt.scatter(df_neg['cosine_similarity'], df_neg['variance_diff'],
                   alpha=0.5, label='Cosine <= 0', color='red')

        if len(df) > 1:
            z = np.polyfit(df['cosine_similarity'], df['variance_diff'], 1)
            p = np.poly1d(z)
            plt.plot(sorted(df['cosine_similarity']), p(sorted(df['cosine_similarity'])),
                    "r--", alpha=0.8)

            corr, p_value = pearsonr(df['cosine_similarity'], df['variance_diff'])
            plt.title(f'Variance Difference vs Cosine Similarity\nCorrelation: {corr:.3f} (p={p_value:.3f})')

        plt.xlabel('Cosine Similarity')
        plt.ylabel('Variance Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)

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

        if len(df) > 1:
            z = np.polyfit(df['fisher_info'], df['reward_var'], 1)
            p = np.poly1d(z)
            plt.plot(sorted(df['fisher_info']), p(sorted(df['fisher_info'])),
                    "r--", alpha=0.8)

            corr, p_value = pearsonr(df['fisher_info'], df['reward_var'])
            plt.title(f'Reward Variance vs Fisher Information\nCorrelation: {corr:.3f} (p={p_value:.3f})')

        plt.xlabel('Fisher Information (κ)')
        plt.ylabel('Reward Variance (σ²)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'fisher_vs_reward_var.png'))
        plt.close()

    def plot_fisher_vs_reward_std_by_iteration(self, iteration_df, correlation_df):
        """Plot Fisher information vs reward std analysis by iteration"""
        if correlation_df.empty:
            print("No correlation data available for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Correlation over iterations
        axes[0, 0].plot(correlation_df['iteration'], correlation_df['correlation'], 'b-o', markersize=6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Correlation')
        axes[0, 0].set_title('Fisher Info vs Reward Std Correlation Over Iterations')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add significance markers
        sig_mask = correlation_df['p_value'] < 0.05
        if sig_mask.any():
            axes[0, 0].scatter(correlation_df.loc[sig_mask, 'iteration'], 
                             correlation_df.loc[sig_mask, 'correlation'], 
                             color='red', s=100, marker='*', label='p < 0.05')
            axes[0, 0].legend()
        
        # Plot 2: Mean Fisher information over iterations
        axes[0, 1].plot(correlation_df['iteration'], correlation_df['mean_fisher'], 'g-o', markersize=6)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Mean Fisher Information')
        axes[0, 1].set_title('Mean Fisher Information Over Iterations')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Mean reward std over iterations
        axes[1, 0].plot(correlation_df['iteration'], correlation_df['mean_reward_std'], 'orange', marker='o', markersize=6)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Mean Reward Std')
        axes[1, 0].set_title('Mean Reward Standard Deviation Over Iterations')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Sample size over iterations
        axes[1, 1].bar(correlation_df['iteration'], correlation_df['n_samples'], alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_title('Sample Size Per Iteration')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'fisher_vs_reward_std_by_iteration.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual scatter plots for each iteration
        unique_iterations = sorted(iteration_df['iteration'].unique())
        if len(unique_iterations) > 1:
            n_iterations = len(unique_iterations)
            n_cols = min(3, n_iterations)
            n_rows = (n_iterations + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_iterations == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i, iteration in enumerate(unique_iterations):
                row = i // n_cols
                col = i % n_cols
                
                if n_iterations == 1:
                    ax = axes[0]
                elif n_rows == 1:
                    ax = axes[0, col]
                elif n_cols == 1:
                    ax = axes[row, 0]
                else:
                    ax = axes[row, col]
                
                iter_data = iteration_df[iteration_df['iteration'] == iteration]
                
                ax.scatter(iter_data['fisher_info'], iter_data['reward_std'], alpha=0.6)
                ax.set_xlabel('Fisher Information')
                ax.set_ylabel('Reward Std')
                ax.set_title(f'Iteration {iteration}')
                ax.grid(True, alpha=0.3)
                
                # Add correlation info
                if len(iter_data) > 1:
                    corr_data = correlation_df[correlation_df['iteration'] == iteration]
                    if not corr_data.empty:
                        corr = corr_data.iloc[0]['correlation']
                        p_val = corr_data.iloc[0]['p_value']
                        ax.text(0.05, 0.95, f'r={corr:.3f}\np={p_val:.3f}', 
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_iterations, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                
                if n_iterations == 1:
                    continue  # No unused subplots
                elif n_rows == 1:
                    ax = axes[0, col]
                elif n_cols == 1:
                    ax = axes[row, 0]
                else:
                    ax = axes[row, col]
                ax.set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'fisher_vs_reward_std_scatter_by_iteration.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_cross_time_correlations(self, cross_corr_df):
        """Plot cross-time correlations to show Fisher(i) and reward_std(j) are uncorrelated"""
        if cross_corr_df.empty:
            print("No cross-time correlation data available for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Correlation vs time difference
        axes[0, 0].scatter(cross_corr_df['time_diff'], cross_corr_df['correlation'], alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Time Difference (j - i)')
        axes[0, 0].set_ylabel('Correlation')
        axes[0, 0].set_title('Cross-Time Correlations: Fisher(i) vs Reward_Std(j)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add significance markers
        sig_mask = cross_corr_df['p_value'] < 0.05
        if sig_mask.any():
            axes[0, 0].scatter(cross_corr_df.loc[sig_mask, 'time_diff'], 
                             cross_corr_df.loc[sig_mask, 'correlation'], 
                             color='red', s=100, marker='*', label='p < 0.05')
            axes[0, 0].legend()
        
        # Plot 2: Distribution of correlations
        axes[0, 1].hist(cross_corr_df['correlation'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Correlation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Cross-Time Correlations')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Correlation vs sample size
        axes[1, 0].scatter(cross_corr_df['n_samples'], cross_corr_df['correlation'], alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Sample Size')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_title('Correlation vs Sample Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: P-values distribution
        axes[1, 1].hist(cross_corr_df['p_value'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0.05, color='r', linestyle='--', alpha=0.7, label='p = 0.05')
        axes[1, 1].set_xlabel('P-value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of P-values')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'cross_time_correlations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a heatmap of correlations
        if len(cross_corr_df) > 1:
            unique_times = sorted(set(cross_corr_df['time_i'].tolist() + cross_corr_df['time_j'].tolist()))
            n_times = len(unique_times)
            
            # Create correlation matrix
            corr_matrix = np.full((n_times, n_times), np.nan)
            
            for _, row in cross_corr_df.iterrows():
                i_idx = unique_times.index(row['time_i'])
                j_idx = unique_times.index(row['time_j'])
                corr_matrix[i_idx, j_idx] = row['correlation']
            
            plt.figure(figsize=(10, 8))
            im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(im, label='Correlation')
            
            # Add text annotations
            for i in range(n_times):
                for j in range(n_times):
                    if not np.isnan(corr_matrix[i, j]):
                        plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
            
            plt.xlabel('Time j (Reward Std)')
            plt.ylabel('Time i (Fisher Info)')
            plt.title('Cross-Time Correlation Heatmap\nFisher(i) vs Reward_Std(j)')
            plt.xticks(range(n_times), [f't={t}' for t in unique_times], rotation=45)
            plt.yticks(range(n_times), [f't={t}' for t in unique_times])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'cross_time_correlation_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
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
        
        print("Analyzing Fisher information vs reward std by iteration...")
        fisher_reward_iter_df, fisher_reward_corr_df = self.analyze_fisher_vs_reward_std_by_iteration()
        
        print("Analyzing cross-time correlations...")
        cross_time_corr_df = self.analyze_cross_time_correlations()

        accuracy_df = self.get_accuracy_dataframe()

        print("Generating plots...")
        self.plot_cosine_similarity_over_iterations()

        if not cos_var_df.empty:
            self.plot_variance_vs_cosine(cos_var_df)
            cos_var_df.to_csv(os.path.join(self.save_dir, 'cosine_variance_data.csv'), index=False)

        if not fisher_reward_df.empty:
            self.plot_fisher_vs_reward_std(fisher_reward_df)
            fisher_reward_df.to_csv(os.path.join(self.save_dir, 'fisher_reward_data.csv'), index=False)
        
        if not fisher_reward_iter_df.empty:
            self.plot_fisher_vs_reward_std_by_iteration(fisher_reward_iter_df, fisher_reward_corr_df)
            fisher_reward_iter_df.to_csv(os.path.join(self.save_dir, 'fisher_reward_data_by_iteration.csv'), index=False)
            fisher_reward_corr_df.to_csv(os.path.join(self.save_dir, 'fisher_reward_correlations_by_iteration.csv'), index=False)
        
        if not cross_time_corr_df.empty:
            self.plot_cross_time_correlations(cross_time_corr_df)
            cross_time_corr_df.to_csv(os.path.join(self.save_dir, 'cross_time_correlations.csv'), index=False)

        if not accuracy_df.empty:
            accuracy_df.to_csv(os.path.join(self.save_dir, 'accuracy_data.csv'), index=False)

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
                f.write("Fisher Information vs Reward Variance Analysis (Overall):\n")
                f.write(f"  Total data points: {len(fisher_reward_df)}\n")

                if len(fisher_reward_df) > 1:
                    corr, p_value = pearsonr(fisher_reward_df['fisher_info'], fisher_reward_df['reward_var'])
                    f.write(f"  Correlation between Fisher and reward variance: {corr:.3f} (p={p_value:.3f})\n")

                    # Linear regression coefficients
                    z = np.polyfit(fisher_reward_df['fisher_info'], fisher_reward_df['reward_var'], 1)
                    f.write(f"  Linear fit: σ² = {z[0]:.4f} * κ + {z[1]:.4f}\n")
            
            if not fisher_reward_corr_df.empty:
                f.write("\nFisher Information vs Reward Std Analysis (By Iteration):\n")
                f.write(f"  Iterations analyzed: {len(fisher_reward_corr_df)}\n")
                
                # Summary statistics for correlations
                correlations = fisher_reward_corr_df['correlation']
                p_values = fisher_reward_corr_df['p_value']
                
                f.write(f"  Mean correlation: {correlations.mean():.3f} ± {correlations.std():.3f}\n")
                f.write(f"  Range: [{correlations.min():.3f}, {correlations.max():.3f}]\n")
                
                # Count significant correlations
                sig_corr = fisher_reward_corr_df[fisher_reward_corr_df['p_value'] < 0.05]
                f.write(f"  Significant correlations (p < 0.05): {len(sig_corr)}/{len(fisher_reward_corr_df)} ({len(sig_corr)/len(fisher_reward_corr_df):.1%})\n")
                
                if len(sig_corr) > 0:
                    f.write(f"  Mean significant correlation: {sig_corr['correlation'].mean():.3f}\n")
                
                # Trend analysis
                if len(fisher_reward_corr_df) > 1:
                    iter_corr, iter_p = pearsonr(fisher_reward_corr_df['iteration'], fisher_reward_corr_df['correlation'])
                    f.write(f"  Correlation trend over iterations: {iter_corr:.3f} (p={iter_p:.3f})\n")
                
                f.write("\n  Per-iteration details:\n")
                for _, row in fisher_reward_corr_df.iterrows():
                    sig_marker = " *" if row['p_value'] < 0.05 else ""
                    f.write(f"    Iteration {row['iteration']}: r={row['correlation']:.3f}, p={row['p_value']:.3f}, n={row['n_samples']}{sig_marker}\n")
            
            if not cross_time_corr_df.empty:
                f.write("\nCross-Time Correlation Analysis (Fisher(i) vs Reward_Std(j), i≠j):\n")
                f.write(f"  Total cross-time pairs analyzed: {len(cross_time_corr_df)}\n")
                
                # Summary statistics for cross-time correlations
                cross_correlations = cross_time_corr_df['correlation']
                cross_p_values = cross_time_corr_df['p_value']
                
                f.write(f"  Mean cross-time correlation: {cross_correlations.mean():.3f} ± {cross_correlations.std():.3f}\n")
                f.write(f"  Range: [{cross_correlations.min():.3f}, {cross_correlations.max():.3f}]\n")
                
                # Count significant cross-time correlations
                sig_cross_corr = cross_time_corr_df[cross_time_corr_df['p_value'] < 0.05]
                f.write(f"  Significant cross-time correlations (p < 0.05): {len(sig_cross_corr)}/{len(cross_time_corr_df)} ({len(sig_cross_corr)/len(cross_time_corr_df):.1%})\n")
                
                # Test if correlations are centered around zero (as expected)
                from scipy import stats
                t_stat, p_val = stats.ttest_1samp(cross_correlations, 0)
                f.write(f"  T-test against zero correlation: t={t_stat:.3f}, p={p_val:.3f}\n")
                
                if len(sig_cross_corr) > 0:
                    f.write(f"  Mean significant cross-time correlation: {sig_cross_corr['correlation'].mean():.3f}\n")
                
                f.write("\n  Cross-time correlation details:\n")
                for _, row in cross_time_corr_df.iterrows():
                    sig_marker = " *" if row['p_value'] < 0.05 else ""
                    f.write(f"    Fisher(t={row['time_i']}) vs Reward_Std(t={row['time_j']}): r={row['correlation']:.3f}, p={row['p_value']:.3f}, n={row['n_samples']}{sig_marker}\n")

            if not accuracy_df.empty:
                f.write("\nAccuracy Analysis:\n")
                avg_accuracy = accuracy_df['accuracy'].mean()
                f.write(f"  Overall average accuracy: {avg_accuracy:.3f}\n")

                iter_group = accuracy_df.groupby('iteration')['accuracy'].mean().to_dict()
                for iteration, acc in sorted(iter_group.items()):
                    f.write(f"  Iteration {iteration} average accuracy: {acc:.3f}\n")

        print(f"Analysis complete. Results saved to {self.save_dir}/")
        return {
            'question_variance_diff': iter_var_diff,
            'iteration_variance_diff': q_var_diff,
            'cosine_variance_df': cos_var_df,
            'fisher_reward_df': fisher_reward_df,
            'accuracy_df': accuracy_df
        }
