#!/bin/bash
set -e

# Experiment Runner for MOSAIC Scaling Analysis
# USAGE: Run this script from the project root (MOSAIC_ADV/)
# bash adv_work/final_work/run_experiment.sh

echo "Step 1: Setting up directories..."
# Ensure we are in the project root by checking for 'adv_work'
if [ ! -d "adv_work/final_work" ]; then
    echo "Error: Please run this script from the project root (MOSAIC_ADV/)."
    echo "Current directory: $(pwd)"
    ls -F
    exit 1
fi

# Add current directory AND final_work to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/adv_work/final_work

mkdir -p outputs/baseline
mkdir -p outputs/ours

echo "Step 2: Generating Scaling Test Cases..."
python3 adv_work/final_work/generate_test_cases.py

JSON_PATH="adv_work/final_work/scaling_experiment.json"
# Fallback logic
if [ ! -f "$JSON_PATH" ] && [ -f "scaling_experiment.json" ]; then
    mv scaling_experiment.json "$JSON_PATH"
fi

echo "Step 3: Running Baseline (Vanilla MOSAIC)..."
python3 adv_work/final_work/inference_baseline.py \
    --json_path "$JSON_PATH" \
    --output_dir outputs/baseline

echo "Step 4: Running Ours (Spatial-Aware Attention Masking)..."
python3 adv_work/final_work/inference_masked.py \
    --json_path "$JSON_PATH" \
    --output_dir outputs/ours \
    --use_shape_mask \
    --use_langsam_mask

echo "Step 5: Evaluating Results..."
EVAL_SCRIPT="adv_work/final_work/eval.py"
if [ ! -f "$EVAL_SCRIPT" ]; then
    EVAL_SCRIPT="evaluation/eval.py"
fi

if [ -f "$EVAL_SCRIPT" ]; then
    echo "Evaluating Baseline..."
    python3 "$EVAL_SCRIPT" --json_path "$JSON_PATH" --output_dir outputs/baseline

    echo "Evaluating Ours..."
    python3 "$EVAL_SCRIPT" --json_path "$JSON_PATH" --output_dir outputs/ours
else
    echo "Warning: eval.py not found. Skipping evaluation."
fi

echo "Experiment Complete! Results are in outputs/baseline and outputs/ours."
