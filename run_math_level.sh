#!/bin/bash
set -euo pipefail

# Activate conda environment
source /home/heqiy/miniconda3/etc/profile.d/conda.sh
conda activate grpo_env

echo "=========================================="
echo "MATH Dataset Experiments with Dual Normalization"
echo "=========================================="

# ---------------- Configuration ----------------
MODEL="Qwen/Qwen2.5-Math-1.5B"
MAX_STEPS=500
WANDB_PROJECT="grpo-math-levels_01"

# Toggle: use ALL GPUs per run (DDP). If 1, runs are sequential per level.
USE_ALL_GPUS=1

# Toggle: enable theory-analysis logging in training.
ENABLE_ANALYSIS=1
ANALYSIS_FLAGS="--analyze_variance \
  --log_curvature --curv_mode trace --lora_only_fisher \
  --log_cosine --cosine_pairs 200"

LOG_DIR="logs/math_levels_01"
OUT_DIR="outputs/math_levels_01"

mkdir -p "$LOG_DIR" "$OUT_DIR"

# -------------- Helpers ----------------
gpu_count() { nvidia-smi -L | wc -l; }

run_single_experiment() {
  local LEVEL="$1"      # 1..5
  local NORM="$2"       # standard | no_std
  local GPU="$3"        # ignored when USE_ALL_GPUS=1
  local LEVEL_NAME="$4"

  local EXTRA_FLAGS=""
  if [[ "$ENABLE_ANALYSIS" -eq 1 ]]; then
    EXTRA_FLAGS="$ANALYSIS_FLAGS"
  fi

  local LOG_FILE="${LOG_DIR}/level${LEVEL}_${NORM}.log"

  if [[ "$USE_ALL_GPUS" -eq 1 ]]; then
    # DDP: use all GPUs, run in foreground
    local NGPUS
    NGPUS="$(gpu_count)"
    echo "Running Level ${LEVEL} (${LEVEL_NAME}) with ${NORM} using all ${NGPUS} GPUs..."
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node="${NGPUS}" \
      train_math_levels.py \
      --difficulty "level${LEVEL}" \
      --normalization "${NORM}" \
      --model_name "${MODEL}" \
      --max_steps "${MAX_STEPS}" \
      --use_wandb \
      --wandb_project "${WANDB_PROJECT}" \
      --exp_name "level${LEVEL}_${NORM}" \
      ${EXTRA_FLAGS} 2>&1 | tee "${LOG_FILE}"
  else
    # Single GPU pinned, backgrounded
    echo "  [GPU ${GPU}] Running Level ${LEVEL} (${LEVEL_NAME}) with ${NORM} normalization"
    CUDA_VISIBLE_DEVICES="${GPU}" python train_math_levels.py \
      --difficulty "level${LEVEL}" \
      --normalization "${NORM}" \
      --model_name "${MODEL}" \
      --max_steps "${MAX_STEPS}" \
      --use_wandb \
      --wandb_project "${WANDB_PROJECT}" \
      --exp_name "level${LEVEL}_${NORM}" \
      ${EXTRA_FLAGS} \
      > "${LOG_FILE}" 2>&1 &
    local PID="$!"
    echo "  [GPU ${GPU}] Started PID ${PID}: Level ${LEVEL} - ${NORM}"
    echo "${PID}"
  fi
}

run_level_experiments() {
  local LEVEL="$1"
  local LEVEL_NAME="$2"
  local GPU1="$3"
  local GPU2="$4"

  echo ""
  echo "=========================================="
  echo "Starting Level ${LEVEL} (${LEVEL_NAME})"
  echo "=========================================="

  if [[ "$USE_ALL_GPUS" -eq 1 ]]; then
    # Sequential (all GPUs)
    run_single_experiment "${LEVEL}" "standard" 0 "${LEVEL_NAME}"
    run_single_experiment "${LEVEL}" "no_std" 0 "${LEVEL_NAME}"
  else
    # Parallel (two GPUs)
    local PID1 PID2
    PID1="$(run_single_experiment "${LEVEL}" "standard" "${GPU1}" "${LEVEL_NAME}")"
    PID2="$(run_single_experiment "${LEVEL}" "no_std"  "${GPU2}" "${LEVEL_NAME}")"
    echo "  Waiting for both experiments to complete..."
    wait "${PID1}" "${PID2}"
  fi

  echo "  Level ${LEVEL} completed!"

  # Show results summary
  echo ""
  echo "  Results for Level ${LEVEL}:"
  echo "  -------------------------"

  local LOG_STD="${LOG_DIR}/level${LEVEL}_standard.log"
  local LOG_NOS="${LOG_DIR}/level${LEVEL}_no_std.log"

  if [[ -f "${LOG_STD}" ]]; then
    echo "  Standard normalization:"
    tail -n 50 "${LOG_STD}" | grep -E "Final Pass@K|Final Sample|FINAL" || echo "    Check log for details"
  fi
  if [[ -f "${LOG_NOS}" ]]; then
    echo "  No_std normalization:"
    tail -n 50 "${LOG_NOS}" | grep -E "Final Pass@K|Final Sample|FINAL" || echo "    Check log for details"
  fi
}

run_all_levels_parallel() {
  # Note: only meaningful when USE_ALL_GPUS=0
  echo ""
  echo "Running ALL levels in parallel using 8 GPUs"
  echo "Level 1 & 2 use GPUs 0-3, Level 3 & 4 use GPUs 4-7, Level 5 uses GPUs 0-1"
  echo ""

  run_single_experiment 1 "standard" 0 "Elementary" >/dev/null
  run_single_experiment 1 "no_std"   1 "Elementary" >/dev/null
  run_single_experiment 2 "standard" 2 "Middle School" >/dev/null
  run_single_experiment 2 "no_std"   3 "Middle School" >/dev/null
  run_single_experiment 3 "standard" 4 "High School" >/devnull 2>&1 || true
  run_single_experiment 3 "no_std"   5 "High School" >/dev/null
  run_single_experiment 4 "standard" 6 "Competition" >/dev/null
  run_single_experiment 4 "no_std"   7 "Competition" >/dev/null

  echo ""
  echo "Waiting for Levels 1-4 to complete before starting Level 5..."
  wait

  echo ""
  echo "Starting Level 5 (Olympiad) on freed GPUs..."
  run_single_experiment 5 "standard" 0 "Olympiad" >/dev/null
  run_single_experiment 5 "no_std"   1 "Olympiad" >/dev/null

  wait
  echo ""
  echo "All levels completed!"
}

show_menu() {
  echo ""
  echo "=========================================="
  echo "Select which experiment to run:"
  echo "=========================================="
  echo "1) Level 1 (Elementary)     - Uses GPUs 0,1"
  echo "2) Level 2 (Middle School)  - Uses GPUs 2,3"
  echo "3) Level 3 (High School)    - Uses GPUs 4,5"
  echo "4) Level 4 (Competition)    - Uses GPUs 6,7"
  echo "5) Level 5 (Olympiad)       - Uses GPUs 0,1"
  echo "6) ALL levels in parallel   - Uses all 8 GPUs"
  echo "7) Levels 1-3 (Easy to Med) - Uses GPUs 0-5"
  echo "8) Levels 4-5 (Hard)        - Uses GPUs 6,7,0,1"
  echo "0) Exit"
  echo "=========================================="
  echo ""
}

# ---------------- Main ----------------
show_menu
read -r -p "Enter your choice (0-8): " choice

case "$choice" in
  1) run_level_experiments 1 "Elementary"    0 1 ;;
  2) run_level_experiments 2 "Middle School" 2 3 ;;
  3) run_level_experiments 3 "High School"   4 5 ;;
  4) run_level_experiments 4 "Competition"   6 7 ;;
  5) run_level_experiments 5 "Olympiad"      0 1 ;;
  6) run_all_levels_parallel ;;
  7)
     echo "Running Levels 1-3 in parallel..."
     PID_A="$(run_single_experiment 1 'standard' 0 'Elementary')"
     PID_B="$(run_single_experiment 1 'no_std'   1 'Elementary')"
     PID_C="$(run_single_experiment 2 'standard' 2 'Middle School')"
     PID_D="$(run_single_experiment 2 'no_std'   3 'Middle School')"
     PID_E="$(run_single_experiment 3 'standard' 4 'High School')"
     PID_F="$(run_single_experiment 3 'no_std'   5 'High School')"
     wait "${PID_A}" "${PID_B}" "${PID_C}" "${PID_D}" "${PID_E}" "${PID_F}"
     echo "Levels 1-3 completed!"
     ;;
  8)
     echo "Running Levels 4-5 (Hard problems)..."
     PID_G="$(run_single_experiment 4 'standard' 6 'Competition')"
     PID_H="$(run_single_experiment 4 'no_std'   7 'Competition')"
     PID_I="$(run_single_experiment 5 'standard' 0 'Olympiad')"
     PID_J="$(run_single_experiment 5 'no_std'   1 'Olympiad')"
     wait "${PID_G}" "${PID_H}" "${PID_I}" "${PID_J}"
     echo "Levels 4-5 completed!"
     ;;
  0) echo "Exiting..."; exit 0 ;;
  *) echo "Invalid choice!"; exit 1 ;;
esac

echo ""
echo "=========================================="
echo "EXPERIMENT SUMMARY"
echo "=========================================="
echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  WandB Project: ${WANDB_PROJECT}"
echo ""
echo "Output locations:"
echo "  Logs: ${LOG_DIR}/"
echo "  Models: ${OUT_DIR}/"
echo ""
echo "Final Results:"
echo "--------------"

for level in 1 2 3 4 5; do
  for norm in standard no_std; do
    LOG_FILE="${LOG_DIR}/level${level}_${norm}.log"
    if [[ -f "${LOG_FILE}" ]]; then
      echo ""
      echo "Level ${level} - ${norm}:"
      tail -n 100 "${LOG_FILE}" | grep -E "Final Pass@K|Final Sample Accuracy|FINAL" || echo "  Running or check log"
    fi
  done
done

echo ""
echo "=========================================="
echo "To monitor progress in real-time:"
echo "  tail -f logs/math_levels_01/level<N>_<norm>.log"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
echo "=========================================="
