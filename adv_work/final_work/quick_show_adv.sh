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
    --subject_counts 6,8,10 \
    --cases_per_count 1 \
    --interaction_ratio 0.3 \
    --seed 42 \
    --use_llm_selection \
    --llm_model gemini-3.1-flash-lite-preview \
    --candidate_pool_size 30

python3 adv_work/final_work/inference_masked.py \
    --json_path adv_work/final_work/quick_scaling_experiment.json \
    --output_dir outputs/ours_quick \
    --penalty_strength 6 \
    --kernel_size 31 \
    --num_samples 2 \
    --self_attn_penalty 1.2 \
    --text_attn_weight 0.7 \
    --disable_bbox

python3 adv_work/final_work/eval.py \
    --json_path adv_work/final_work/quick_scaling_experiment.json \
    --output_dir outputs/ours_quick
