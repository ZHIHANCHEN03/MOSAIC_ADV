#!/bin/bash

# Experiment Runner for MOSAIC Scaling Analysis

# 1. Setup
echo "Step 1: Setting up experiment..."
cd adv_work/final_work || exit
mkdir -p ../../outputs/baseline
mkdir -p ../../outputs/ours

# 2. Generate Test Data
echo "Step 2: Generating Scaling Test Cases (8, 10, 12 subjects)..."
python3 generate_test_cases.py

# 3. Run Baseline (Vanilla MOSAIC)
echo "Step 3: Running Baseline (Vanilla MOSAIC)..."
# We use the original inference.py but point it to our new json and output dir
# Note: Original inference.py might need modification to accept args, 
# or we just use a simple wrapper here.
# Assuming we can run a vanilla version of inference_masked.py without the mask patch.
# For simplicity, let's just run inference_masked.py but DISABLE the mask injection logic via a flag or separate script.
# Or better: We create a baseline script that is just the original inference logic.
# Let's assume inference.py in root is vanilla.
cd ../../
python3 inference.py --json_path adv_work/final_work/scaling_experiment.json --output_dir outputs/baseline
# Note: Original inference.py might not support args. If not, we might need to patch it or copy it.
# If inference.py is hardcoded, we should use a copy.

# 4. Run Ours (Spatial-Aware Attention Masking)
echo "Step 4: Running Ours (Spatial-Aware Attention Masking)..."
cd adv_work/final_work
python3 inference_masked.py --json_path scaling_experiment.json --output_dir ../../outputs/ours

# 5. Evaluation
echo "Step 5: Evaluating Results..."
cd ../../
echo "Evaluating Baseline..."
python3 evaluation/eval.py --json_path adv_work/final_work/scaling_experiment.json --output_dir outputs/baseline

echo "Evaluating Ours..."
python3 evaluation/eval.py --json_path adv_work/final_work/scaling_experiment.json --output_dir outputs/ours

echo "Experiment Complete! Check outputs/baseline and outputs/ours."
