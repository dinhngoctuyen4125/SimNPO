# SimNPO + OOD: Unlearning Deprecated APIs from Code LLMs

## Setup

```bash
conda create -n simnpo python=3.10
conda activate simnpo
pip install -r requirements.txt
```

## Pipeline

### Stage 1: Train SimNPO LoRA (Unlearn Deprecated APIs)

```bash
sudo nohup bash run_script.sh > run_simnpo.log 2>&1 &
```

### Stage 2: Train OOD Detector

```bash
sudo nohup bash train_ood.sh > run_ood.log 2>&1 &
```

### Stage 3: Soft-Weighted Inference

```bash
sudo nohup bash eval_soft_infer.sh > sim_ood.log 2>&1 &
```