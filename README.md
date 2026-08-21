# SimNPO + OOD: Unlearning Deprecated APIs from Code LLMs

A framework for removing deprecated API knowledge from Large Language Models using **SimNPO** (Simple Negative Preference Optimization) combined with **OOD-guided soft-weighted LoRA inference**.

## Overview

The system consists of **3 stages**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: SimNPO LoRA Training (unlearn deprecated APIs)        │
│                                                                 │
│  CodeLlama-7b (frozen) + LoRA (q_proj, v_proj)                 │
│       ↓  Train with SimNPO loss on D_forget.json                │
│  Output: LoRA adapter (simnpo/ckpt/simnpo_gdr/)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: OOD Detector Training                                 │
│                                                                 │
│  CodeBERT/RoBERTa + LoRA → contrastive learning on D_forget     │
│       ↓  Fit Mahalanobis stats + OCSVM + GMM                   │
│  Output: OOD artifacts (ood_checkpoints_codellama_0/)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: OOD-Guided Soft-Weighted Inference                    │
│                                                                 │
│  For each input x:                                              │
│    1. OOD Detector → w(x) ∈ {0, (0.3,0.4], 1.2}               │
│    2. LLM forward:  out = W·x·1 + ΔW·x·w(x)                   │
│       w(x)=0   → base model only (retain capability)           │
│       w(x)=1.2 → LoRA unlearn activated (forget deprecated)    │
│  Output: generated code without deprecated APIs                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── run_script.sh                 # Stage 1: SimNPO LoRA training
├── train_ood.sh                  # Stage 2: OOD detector training
├── eval_soft_infer.sh            # Stage 3: Soft-weighted inference
│
├── simnpo/                       # SimNPO unlearning module
│   ├── unlearn.py                #   CLI entry point
│   └── baselines/
│       ├── __init__.py
│       ├── dataset.py            #   Dataset classes (DepAPIDataset)
│       ├── iterative.py          #   SimNPO training loop & loss
│       └── utils.py              #   Model/tokenizer utilities
│
├── train_ood.py                  # OOD detector training script
├── eval_o3.py                    # Inference with OOD + soft-weighted LoRA
│
├── src/                          # Shared modules
│   ├── ood_model_selector.py     #   RoBERTa OOD model (train & inference)
│   ├── ood_data.py               #   OOD dataset loading
│   ├── ood_utils.py              #   OOD utilities & metrics
│   ├── ood_calculate_log.py      #   AUROC, TNR, DTACC metrics
│   ├── modeling_llama_hacked_o.py#   LlamaForCausalLM with ood_weight support
│   ├── peft_model_hacked_o.py    #   Custom PeftModel for soft-weighted LoRA
│   ├── lora_model_hacked_o.py    #   Custom LoraModel
│   ├── lora_layer_hacked_o.py    #   LoRA Linear: W·x·w_base + ΔW·x·w(x)
│   └── mapping_hacked_o.py       #   PEFT model type mapping
│
├── data/
│   └── codellama/
│       ├── D_forget.json         #   Forget set (deprecated API samples)
│       └── D_test.json           #   Test set for evaluation
│
├── requirements.txt
└── OOD_Architecture_Reference.md #   Detailed architecture documentation
```

## Setup

### 1. Create Environment

```bash
conda create -n simnpo python=3.10
conda activate simnpo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Requires CUDA-compatible GPU. Models are loaded in bfloat16.

## Pipeline

### Stage 1: Train SimNPO LoRA (Unlearn Deprecated APIs)

```bash
bash run_script.sh
```

Trains a LoRA adapter (rank=8, `q_proj` + `v_proj`) on CodeLlama-7b using SimNPO loss to unlearn deprecated API knowledge while preserving general capability via retain loss (GDR).

**Input:** `D_forget.json` — each item contains:
```json
{
    "function": "full function code with deprecated API usage",
    "probing input": "code context before the API call",
    "y_neg": "continuation using deprecated API (to unlearn)",
    "y_pos": "continuation using replacement API (correct)",
    "retain": "clean code to preserve capability"
}
```

**Output:** LoRA adapter at `simnpo/ckpt/simnpo_gdr/`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--algo` | — | Algorithm: `simnpo_gdr` |
| `--model_dir` | — | HuggingFace model path |
| `--data_file` | — | Path to data JSON file |
| `--out_dir` | — | Output directory for LoRA adapter |
| `--max_len` | `4096` | Max sequence length |
| `--epochs` | `5` | Training epochs |
| `--lr` | `1e-5` | Learning rate |
| `--per_device_batch_size` | `2` | Batch size per GPU |
| `--beta` | `0.1` | SimNPO temperature |
| `--coeff` | `0.1` | Retain loss weight |
| `--npo_coeff` | `0.1` | Forget loss weight |
| `--gamma` | `0.1` | SimNPO gamma parameter |

---

### Stage 2: Train OOD Detector

```bash
bash train_ood.sh
```

Trains a RoBERTa-based (CodeBERT) OOD detector using contrastive learning to distinguish "forget domain" inputs from general inputs. After each epoch, fits a One-Class SVM and GMM for scoring.

**Input:** `D_forget.json` — uses `"function"` field as in-distribution data and `"retain"` field as out-of-distribution data.

**Output:** `ood_checkpoints_codellama_0/` containing:
```
├── codellama_all_ood_codellama_roberta_ocsvm/     # RoBERTa LoRA weights
├── codellama_all_ood_codellama_ocsvm.pkl           # Fitted OCSVM
├── codellama_all_ood_codellama_mean_list_ocsvm.pt  # Per-layer mean vectors
├── codellama_all_ood_codellama_precision_list_ocsvm.pt
├── codellama_all_ood_codellama_fea_list_ocsvm.pt
├── codellama_all_ood_codellama_gmm_w_ocsvm.pkl     # GMM for weight computation
└── codellama_all_ood_codellama_threshold_ocsvm.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name_or_path` | `microsoft/codebert-base` | RoBERTa encoder |
| `--unlearn_dataset` | `codellama_torch` | Dataset name (ID) |
| `--ood_dataset` | `ood_codellama` | Dataset name (OOD) |
| `--base_unlearn_path` | — | Path to forget data JSON |
| `--base_ood_path` | — | Path to OOD/retain data JSON |
| `--batch_size` | `8` | Batch size |
| `--num_train_epochs` | `2` | Training epochs |
| `--seed` | `2026` | Random seed |

---

### Stage 3: Soft-Weighted Inference

```bash
bash eval_soft_infer.sh
```

Combines the trained OOD detector with the SimNPO LoRA adapter for inference. For each input, the OOD detector computes a per-sample weight `w(x)` that modulates the LoRA contribution:

```
w(x) = 0     → LoRA off, base model only (input is NOT deprecated API related)
w(x) ∈ (0,1) → Partial LoRA activation
w(x) = 1.2   → Full LoRA activation (input IS deprecated API related)
```

This is implemented via hacked HuggingFace files (`src/*_hacked_o.py`) that modify the LoRA forward pass to: `output = W·x·1 + ΔW·x·w(x)`.

**Input:**
- Base model: `codellama/CodeLlama-7b-hf`
- LoRA adapter: `simnpo/ckpt/simnpo_gdr/` (from Stage 1)
- OOD artifacts: `ood_checkpoints_codellama_0/` (from Stage 2)
- Test data: `D_test.json`

**Output:** JSON file with per-sample predictions and soft-weight summary statistics.