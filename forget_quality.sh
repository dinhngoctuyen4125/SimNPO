#!/bin/bash

LORA_CHECKPOINT="./simnpo/ckpt/simnpo_gdr"
BASE_MODEL="codellama/CodeLlama-7b-hf"

/home/ritsu/miniconda3/envs/simnpo/bin/python forget_quality.py \
    --model_path "${LORA_CHECKPOINT}" \
    --base_model "${BASE_MODEL}" \
    --batch_size 32 \
    --max_new_tokens 300
