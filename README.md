# SimNPO: Unlearning Deprecated APIs from Code LLMs

A framework for removing deprecated API knowledge from Large Language Models using **SimNPO** (Simple Negative Preference Optimization).

## Project Structure

```
├── run_script.sh                     # Script chạy unlearning
├── requirements.txt                  # Dependencies
├── data/
│   └── codellama/
│       └── depAPI.json
└── simnpo/
    ├── unlearn.py                # CLI entry point
    └── baselines/
        ├── __init__.py
        ├── dataset.py            # Dataset classes (DepAPIDataset)
        ├── iterative.py          # SimNPO training loop & loss
        └── utils.py              # Model/tokenizer utilities
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

> **Note:** Requires CUDA-compatible GPU. Model is loaded in bfloat16 with LoRA adapters via `peft`.

## Run Unlearning

### Start

```bash
bash run_script.sh
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--algo` | — | Algorithm: `simnpo_gdr` |
| `--model_dir` | — | Path to HuggingFace model directory |
| `--tokenizer_dir` | `None` | Tokenizer path (defaults to `model_dir`) |
| `--data_file` | — | Path to data JSON file |
| `--out_dir` | — | Output directory for unlearned model |
| `--max_len` | `4096` | Max sequence length |
| `--epochs` | `5` | Number of training epochs |
| `--lr` | `1e-5` | Learning rate |
| `--per_device_batch_size` | `2` | Batch size per GPU |
| `--beta` | `0.1` | SimNPO temperature |
| `--coeff` | `0.1` | Retain loss weight |
| `--npo_coeff` | `0.1` | Forget loss weight |
| `--gamma` | `0.1` | SimNPO gamma parameter |

### Data Format

Input JSON file (`depAPI.json`) — each item contains:

```json
{
    "probing input": "code context using deprecated API...",
    "forget": "deprecated API usage to unlearn",
    "retain": "clean code to preserve capability"
}
```

### Output

The unlearned model checkpoint is saved to `--out_dir`, with one checkpoint per epoch.