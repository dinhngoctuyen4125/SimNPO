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
sudo nohup bash run_simnpo.sh > logs/run_simnpo.log 2>&1 &
```

*(Tùy chọn)* Đánh giá mis/rep/dep API sau khi train SimNPO (trên tập D_test):
```bash
sudo nohup bash forget_quality.sh > logs/forget_quality.log 2>&1 &
```

### Stage 2: Train OOD Detector

```bash
sudo nohup bash train_ood.sh > logs/train_ood.log 2>&1 &
```

### Stage 3: Soft-Weighted Inference

```bash
sudo nohup bash eval_soft_infer.sh > logs/eval_soft_infer.log 2>&1 &
```

### Stage 4: Test Model Utility (HumanEval)

```bash
sudo nohup bash model_utility.sh > logs/model_utility.log 2>&1 &
```

> [!WARNING]  
> **Lưu ý cho GPU RTX 5090:** Môi trường `prod` (hoặc `simnpo`) mặc định (CUDA 11.8) sẽ bị lỗi `no kernel image` trên card mới. Hãy cài môi trường `prod_eval` riêng biệt bằng các lệnh sau:
> ```bash
> conda create -n prod_eval python=3.10 -y
> conda activate prod_eval
> tail -n +4 requirements.txt | pip install -r /dev/stdin
> pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
> conda install "mkl<2024.0" -c conda-forge -y
> ```
> *(Đừng quên sửa `model_utility.sh` trỏ sang `prod_eval` trước khi chạy)*