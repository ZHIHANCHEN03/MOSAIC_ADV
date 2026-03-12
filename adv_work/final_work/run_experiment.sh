#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Experiment Runner for MOSAIC Scaling Analysis
# USAGE: Run this script from the project root (MOSAIC_ADV/)
# bash adv_work/final_work/run_experiment.sh

echo "Step 1: Setting up directories..."
# Ensure we are in the project root
if [ ! -d "adv_work/final_work" ]; then
    echo "Error: Please run this script from the project root (MOSAIC_ADV/)."
    exit 1
fi

mkdir -p outputs/baseline
mkdir -p outputs/ours

echo "Step 2: Generating Scaling Test Cases..."
# Run generator from root, it should handle paths correctly or we adjust it
# generate_test_cases.py writes 'scaling_experiment.json' to CWD.
# We want it in adv_work/final_work/ or root?
# Let's write to adv_work/final_work/scaling_experiment.json
python3 adv_work/final_work/generate_test_cases.py

echo "Step 3: Running Baseline (Vanilla MOSAIC)..."
# Check if json exists
JSON_PATH="adv_work/final_work/scaling_experiment.json"
if [ ! -f "$JSON_PATH" ]; then
    echo "Error: Test case JSON not found at $JSON_PATH"
    # Fallback: maybe it generated in root?
    if [ -f "scaling_experiment.json" ]; then
        mv scaling_experiment.json "$JSON_PATH"
    else
        exit 1
    fi
fi

# Run Baseline Inference
# We use inference_baseline.py which should be in adv_work/final_work/
if [ -f "adv_work/final_work/inference_baseline.py" ]; then
    python3 adv_work/final_work/inference_baseline.py \
        --json_path "$JSON_PATH" \
        --output_dir outputs/baseline
else
    # Fallback to root inference.py if baseline script missing
    python3 inference.py \
        --json_path "$JSON_PATH" \
        --output_dir outputs/baseline
fi

echo "Step 4: Running Ours (Spatial-Aware Attention Masking)..."
python3 adv_work/final_work/inference_masked.py \
    --json_path "$JSON_PATH" \
    --output_dir outputs/ours

echo "Step 5: Evaluating Results..."
echo "Evaluating Baseline..."
python3 evaluation/eval.py \
    --json_path "$JSON_PATH" \
    --output_dir outputs/baseline

echo "Evaluating Ours..."
python3 evaluation/eval.py \
    --json_path "$JSON_PATH" \
    --output_dir outputs/ours

echo "Experiment Complete! Results are in outputs/baseline and outputs/ours."
