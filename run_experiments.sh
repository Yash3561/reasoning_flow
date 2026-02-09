#!/bin/bash

# Experiment 1: Order 0 (Positions) - Semantic Clustering
echo "Running Experiment 1..."
python cot-hidden-dynamic.py --hf_model Qwen/Qwen2.5-0.5B --data_file data/all_final_data.json --pooling step_mean --accumulation cumulative --similarity_order 0 --save_dir results/exp1_order0 --sections logicA,logicB,logicC --hide_axis_text

# Experiment 2: Order 1 (Velocities) - Logic Clustering
echo "Running Experiment 2..."
python cot-hidden-dynamic.py --hf_model Qwen/Qwen2.5-0.5B --data_file data/all_final_data.json --pooling step_mean --accumulation cumulative --similarity_order 1 --save_dir results/exp2_order1 --sections logicA,logicB,logicC --hide_axis_text

# Experiment 3: Cross-Language (Logic Invariance) - Focus on LogicA
echo "Running Experiment 3..."
python cot-hidden-dynamic.py --hf_model Qwen/Qwen2.5-0.5B --data_file data/all_final_data.json --pooling step_mean --accumulation cumulative --similarity_order 1 --save_dir results/exp3_multilingual --sections logicA --hide_axis_text

echo "All experiments completed."
