import os
import argparse
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class BaseTrainer(Trainer):
    def __init__(self, beta=None, gamma=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.gamma = gamma

class SimNPO_FT(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_data = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_data[0].long(),
            "attention_mask": forget_data[1].long(),
            "labels": forget_data[2].long(),
        }

        retain_data = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_data[0].long(),
            "attention_mask": retain_data[1].long(),
            "labels": retain_data[2].long(),
        }

        outputs = model(**forget_inputs)
        current_forget_loss = outputs.loss

        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss

        forget_loss = - torch.nn.functional.logsigmoid(self.beta * current_forget_loss).mean() * 2 / self.beta

        loss = forget_loss + self.gamma * retain_loss
        
        return (loss, outputs) if return_outputs else loss

class CustomAPIDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
            
    def __len__(self):
        return len(self.data)
        
    def _format_and_tokenize(self, context, target):

        text = context + target
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        labels = input_ids.copy()
        
        context_encoded = self.tokenizer(context, add_special_tokens=True)
        context_len = len(context_encoded['input_ids'])
        
        for i in range(min(context_len, len(labels))):
            labels[i] = -100
            
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_context = item["probing input"]
        
        forget_input_ids, forget_mask, forget_labels = self._format_and_tokenize(raw_context, item["forget"])
        
        retain_input_ids, retain_mask, retain_labels = self._format_and_tokenize("", item["retain"])
        
        return {
            "forget": (forget_input_ids, forget_mask, forget_labels),
            "retain": (retain_input_ids, retain_mask, retain_labels)
        }

class UnlearnFlow:
    def __init__(self, model_name, cache_dir, file_path, beta=1.0, gamma=0.0, **kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.file_path = file_path
        self.beta = beta
        self.gamma = gamma
        self.dataset_seed = kwargs.get("dataset_seed", 0)
        self.forget_ratio = kwargs.get("forget_ratio", 200)
        self.self_retain = kwargs.get("self_retain", False)
        self.if_llama = "llama" in self.model_name.lower()
        self.batch_size = kwargs.get("batch_size", 4)
        self.lr = kwargs.get("lr", 1e-5)
        self.max_steps = kwargs.get("max_steps", 100)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.num_train_epochs = kwargs.get("num_train_epochs", 3)
        self.max_length = kwargs.get("max_length", 2048)
        self.output_dir = kwargs.get("output_dir", "outputs/simnpo_checkpoints")
        self.final_model_dir = kwargs.get("final_model_dir", "outputs/simnpo_final")

    def init_model(self):
        print(f"Loading the checkpoint from {self.model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            if self.if_llama:
                tokenizer.add_special_tokens({"pad_token": "[pad]"})
            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

        model.resize_token_embeddings(len(tokenizer))
        
        self.model = model
        self.tokenizer = tokenizer

    def init_dataset(self):
        self.unlearn_dataset = CustomAPIDataset(
            file_path=self.file_path, 
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

        def custom_collator(samples):
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            forget_ids = pad_sequence([s["forget"][0] for s in samples], batch_first=True, padding_value=pad_id)
            forget_masks = pad_sequence([s["forget"][1] for s in samples], batch_first=True, padding_value=0)
            forget_labels = pad_sequence([s["forget"][2] for s in samples], batch_first=True, padding_value=-100)
            
            retain_ids = pad_sequence([s["retain"][0] for s in samples], batch_first=True, padding_value=pad_id)
            retain_masks = pad_sequence([s["retain"][1] for s in samples], batch_first=True, padding_value=0)
            retain_labels = pad_sequence([s["retain"][2] for s in samples], batch_first=True, padding_value=-100)
            
            return {
                "forget": (forget_ids, forget_masks, forget_labels),
                "retain": (retain_ids, retain_masks, retain_labels)
            }
            
        self.unlearn_collator = custom_collator

    def init_unlearner(self):
        training_args = transformers.TrainingArguments(
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=True,
            output_dir=self.output_dir,
            optim="adamw_torch",
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            report_to=[],
            logging_steps=10,
        )

        self.unlearner = SimNPO_FT(
            model=self.model,
            train_dataset=self.unlearn_dataset,
            data_collator=self.unlearn_collator,
            args=training_args,
            beta=self.beta,
            gamma=self.gamma
        )

    def save(self):
        self.model.save_pretrained(self.final_model_dir)
        self.tokenizer.save_pretrained(self.final_model_dir)

    def run(self):
        print("1. Initializing model...")
        self.init_model()
        
        print("2. Preparing dataset...")
        self.init_dataset()
        
        print("3. Initializing SimNPO Trainer...")
        self.init_unlearner()
        
        print("4. Starting training...")
        if self.unlearner:
            self.unlearner.train()
            
        print("5. Training completed. Saving model...")
        self.save()
        print("Done!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--file_path", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="outputs/simnpo_checkpoints")
    parser.add_argument("--final_model_dir", default="outputs/simnpo_final")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    workflow = UnlearnFlow(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        file_path=args.file_path,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        beta=args.beta,
        gamma=args.gamma,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        final_model_dir=args.final_model_dir,
    )
    workflow.run()
