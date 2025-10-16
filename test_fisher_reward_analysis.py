#!/usr/bin/env python3
"""
Test script to demonstrate the new Fisher information vs reward std analysis by iteration.
This script creates sample data and runs the analysis to show how the new functionality works.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variance_analysis import VarianceAnalyzer

def create_sample_data():
    """Create sample Fisher information and reward std data for testing"""
    analyzer = VarianceAnalyzer(save_dir="test_fisher_reward_analysis")
    
    # Create sample data for 3 questions across 5 iterations
    questions = ['prompt_0', 'prompt_1', 'prompt_2']
    iterations = [0, 100, 200, 300, 400]
    
    # Generate realistic Fisher information and reward std values
    np.random.seed(42)
    
    for iteration in iterations:
        for i, question in enumerate(questions):
            # Fisher information: decreases over iterations (learning effect)
            base_fisher = 10.0 - (iteration / 1000.0) * 5.0
            fisher_info = base_fisher + np.random.normal(0, 1.0)
            fisher_info = max(0.1, fisher_info)  # Ensure positive
            
            # Reward std: also decreases over iterations (convergence)
            base_reward_std = 0.3 - (iteration / 1000.0) * 0.15
            reward_std = base_reward_std + np.random.normal(0, 0.05)
            reward_std = max(0.01, reward_std)  # Ensure positive
            
            # Add some correlation between Fisher info and reward std
            if iteration > 100:  # Stronger correlation in later iterations
                correlation_factor = 0.7
                reward_std = reward_std + correlation_factor * (fisher_info - base_fisher) * 0.1
            
            analyzer.record_fisher_info(question, iteration, fisher_info)
            analyzer.record_reward_std(question, iteration, reward_std)
    
    return analyzer

def main():
    """Run the test analysis"""
    print("Creating sample data...")
    analyzer = create_sample_data()
    
    print("Running Fisher information vs reward std analysis by iteration...")
    
    # Run the new per-iteration analysis
    iteration_df, correlation_df = analyzer.analyze_fisher_vs_reward_std_by_iteration()
    
    print("\nResults:")
    print("=" * 50)
    
    print("\nPer-iteration correlation data:")
    print(correlation_df.to_string(index=False))
    
    print("\nDetailed iteration data:")
    print(iteration_df.to_string(index=False))
    
    # Generate plots
    print("\nGenerating plots...")
    analyzer.plot_fisher_vs_reward_std_by_iteration(iteration_df, correlation_df)
    
    # Also run the original analysis for comparison
    print("\nRunning original Fisher vs reward std analysis...")
    original_df = analyzer.analyze_fisher_vs_reward_std()
    analyzer.plot_fisher_vs_reward_std(original_df)
    
    print("\nAnalysis complete! Check the 'test_fisher_reward_analysis' directory for results.")
    print("\nKey differences:")
    print("- Original analysis: Aggregates all data points across iterations")
    print("- New analysis: Calculates correlation separately for each iteration")
    print("- New analysis: Shows how correlation changes over training iterations")
    print("- New analysis: Provides per-iteration statistics and trend analysis")

if __name__ == "__main__":
    main()
