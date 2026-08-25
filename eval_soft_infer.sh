#!/bin/bash

#SBATCH --job-name=ood
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

BASE_MODEL="codellama/CodeLlama-7b-hf"
OOD_SETTING="C"
for SEED in 0
do
    OUTPUT_1="./simnpo/ckpt/simnpo_gdr"

    # Eval on D_test.json
    TESTPATH_1="../Data-Collection/codellama/D_test.json"
    /home/ritsu/miniconda3/envs/simnpo/bin/python eval_soft_infer.py \
      --test_dataset ${TESTPATH_1} \
      --base_model ${BASE_MODEL} \
      --seed ${SEED} \
      --lora_weights ${OUTPUT_1} \
      --ood_type "_all" \
      --ood_setting ${OOD_SETTING} \
      --ood_weights "./ood_checkpoints_codellama_${SEED}/" \
      --ood_base_model "microsoft/codebert-base" \
      --ood_setting_name "codellama"
done