#!/bin/bash
set -e

if [ ! -d "adv_work/final_work" ]; then
    echo "Error: Please run this script from the project root (MOSAIC_ADV/)."
    echo "Current directory: $(pwd)"
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/adv_work/final_work

mkdir -p outputs/ours_quick

python3 adv_work/final_work/generate_test_cases.py \
    --output_path adv_work/final_work/quick_scaling_experiment.json \
    --subject_counts 8,10,12 \
    --cases_per_count 1 \
    --interaction_ratio 0.3 \
    --seed 42

python3 adv_work/final_work/inference_masked.py \
    --json_path adv_work/final_work/quick_scaling_experiment.json \
    --output_dir outputs/ours_quick \
    --use_shape_mask

python3 adv_work/final_work/eval.py \
    --json_path adv_work/final_work/quick_scaling_experiment.json \
    --output_dir outputs/ours_quick
