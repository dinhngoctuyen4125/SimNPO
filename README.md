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
sudo nohup bash run_script.sh > logs/run_simnpo.log 2>&1 &
```

*(Tùy chọn)* Đánh giá mis/rep/dep API sau khi train SimNPO (trên tập D_test):
```bash
sudo nohup bash forget_quality.sh > logs/run_forget.log 2>&1 &
```

### Stage 2: Train OOD Detector

```bash
sudo nohup bash train_ood.sh > logs/run_ood.log 2>&1 &
```

### Stage 3: Soft-Weighted Inference

```bash
sudo nohup bash eval_soft_infer.sh > logs/sim_ood.log 2>&1 &
```