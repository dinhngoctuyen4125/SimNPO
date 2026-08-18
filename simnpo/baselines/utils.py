from pathlib import Path
import os
import torch
from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re


def get_rootpath():
    return str(Path(__file__).parent.resolve())


def get_basename(file_path: str):
    return os.path.basename(os.path.normpath(file_path))


def read_text(file_path: str) -> str:
    if Path(file_path).suffix != '.txt':
        raise ValueError

    with open(file_path, 'r') as f:
        text: str = f.read()
    return text


def read_json(fpath: str):
    fpath = str(fpath)
    with open(fpath, 'r') as f:
        return json.load(f)


def load_model(model_dir: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    return model


def load_tokenizer(
    tokenizer_dir: str,
    add_pad_token: bool = True,
    use_fast: bool = True
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast) 
    if add_pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_dir: str,
    tokenizer_dir: str | None = None,
    add_pad_token: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = load_model(model_dir)
    tokenizer = load_tokenizer(tokenizer_dir or model_dir, add_pad_token)
    return model, tokenizer


def pad_or_trim_tensor(tensor, target_length, padding_value=0):
    current_length = tensor.size(0)
    
    if current_length < target_length:
        # Padding
        padding_size = target_length - current_length
        padding_tensor = torch.full((padding_size,), padding_value, dtype=tensor.dtype)
        padded_tensor = torch.cat((tensor, padding_tensor))
        return padded_tensor
    
    elif current_length > target_length:
        # Trimming
        trimmed_tensor = tensor[:target_length]
        return trimmed_tensor
    
    else:
        # No change needed
        return tensor
