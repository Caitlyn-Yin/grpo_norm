#!/bin/bash

# Activate conda environment
source /home/heqiy/miniconda3/etc/profile.d/conda.sh
conda activate grpo_env

# Script to run analysis after training completes

# Default values
DIFFICULTY="easy"
NORMALIZATION="standard"
EXP_NAME="iclr1_analysis"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --difficulty)
            DIFFICULTY="$2"
            shift 2
            ;;
        --normalization)
            NORMALIZATION="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set paths
OUTPUT_DIR="outputs/GRPO_final3_difficulty/${DIFFICULTY}/${NORMALIZATION}/Qwen2.5-Math-7B/${EXP_NAME}"
PLOT_DIR="training_plots_${DIFFICULTY}_${NORMALIZATION}_${EXP_NAME}"
WANDB_DIR="wandb"

echo "Analyzing training results..."
echo "Output directory: $OUTPUT_DIR"
echo "Plot directory: $PLOT_DIR"

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory $OUTPUT_DIR does not exist!"
    echo "Make sure training has completed and checkpoints are saved."
    exit 1
fi

# Create plots from checkpoints and wandb logs
echo "Creating training plots..."
python plot_training_metrics.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_dir "$WANDB_DIR" \
    --save_dir "$PLOT_DIR"

# Run variance analysis on the final checkpoint
echo "Running variance analysis on checkpoints..."
python analyze_checkpoints.py \
    --model_dir "$OUTPUT_DIR" \
    --difficulty "$DIFFICULTY" \
    --normalization "$NORMALIZATION" \
    --save_dir "$PLOT_DIR/variance_analysis"

# Run standard deviation analysis
echo "Running standard deviation analysis..."
python complete_std_analysis.py \
    --model_dir "$OUTPUT_DIR" \
    --difficulty "$DIFFICULTY" \
    --normalization "$NORMALIZATION" \
    --save_dir "$PLOT_DIR/std_analysis"

echo "Analysis complete!"
echo "Plots saved to: $PLOT_DIR"

# Generate summary report
echo "Generating summary report..."
cat > "$PLOT_DIR/summary.md" << EOF
# Training Analysis Report

## Configuration
- Difficulty: $DIFFICULTY
- Normalization: $NORMALIZATION
- Experiment: $EXP_NAME
- Model: Qwen/Qwen2.5-Math-7B

## Output Files
- Training metrics: $PLOT_DIR/training_metrics.csv
- Training plots: $PLOT_DIR/training_metrics_overview.png
- Individual plots: $PLOT_DIR/individual_plots/
- Variance analysis: $PLOT_DIR/variance_analysis/
- Std analysis: $PLOT_DIR/std_analysis/

## Key Metrics
See training_metrics.csv for detailed metrics.

Generated on: $(date)
EOF

echo "Summary report saved to: $PLOT_DIR/summary.md"
