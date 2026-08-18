#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd simnpo

python unlearn.py \
    --algo simnpo_gdr \
    --model_dir codellama/CodeLlama-7b-hf \
    --data_file ../data/codellama/D_forget.json \
    --out_dir ./ckpt/simnpo_gdr \
    --max_len 256 \
    --epochs 2 \
    --lr 1e-5 \
    --per_device_batch_size 32 \
    --beta 0.1 \
    --coeff 0.1 \
    --npo_coeff 0.1 \
    --gamma 0.1
