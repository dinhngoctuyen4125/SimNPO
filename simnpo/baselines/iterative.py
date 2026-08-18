from .utils import load_model_and_tokenizer
from .dataset import ForgetRetainDataset, DepAPIDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from torch import nn


def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,
    beta: float = 0.1,
    coeff: float = 1.0,
    npo_coeff: float = 1.0,
    gamma: float = 0.0
):
    if 'gdr' in loss_type and not data_file.endswith('.json'):
        assert retain_data_file is not None, "Retain data must be specified for SimNPO+GDR."

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ref_model = None

    if data_file.endswith('.json'):
        dataset = DepAPIDataset(data_file, tokenizer=tokenizer, max_len=max_len)
    else:
        dataset = ForgetRetainDataset(
            data_file,
            tokenizer=tokenizer,
            retain_file_path=retain_data_file,
            max_len=max_len
        )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='epoch',
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        gradient_checkpointing=True,
        report_to='none'
    )

    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        beta=beta,
        coeff=coeff,
        npo_coeff=npo_coeff,
        gamma=gamma
    )

    model.config.use_cache = False  # silence the warnings.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)



class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 loss_type: str = 'simnpo',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 coeff: float = 1.0,
                 npo_coeff: float = 1.0,
                 gamma: float = 0.0,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Beta parameter for SimNPO
        self.coeff = coeff
        self.npo_coeff = npo_coeff
        self.gamma = gamma  # Gamma parameter for SimNPO

        super().__init__(*args, **kwargs)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        
        ### 1. Run model ###
        x_f, x_r = inputs
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        )
        loss_f = outputs_f.loss

        if 'gdr' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

        ### 2. Compute Loss ###
        loss = 0

        if 'simnpo' in self.loss_type:
            # NOTE: This operates on the full logits tensor (shape [batch, seq_len, vocab_size]),
            # not per-token log-probabilities. Verify this matches the intended SimNPO formulation.
            neg_log_ratio = - outputs_f.logits - self.gamma
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        else:
            raise NotImplementedError("Only SimNPO variants are supported. Loss type not recognized.")

        if 'gdr' in self.loss_type:
            # NOTE: This print executes every training step — may produce excessive log output.
            print(f"loss_f: {loss_f}, loss_r: {loss_r}")
            loss = self.npo_coeff * loss + self.coeff * loss_r

        return (loss, outputs_f) if return_outputs else loss


    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            x_f, x_r = inputs
            input_ids = x_f['input_ids']
            labels = x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone()
            attention_mask = x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
        else:
            input_ids = inputs['input_ids']
            labels = inputs['labels'] if 'labels' in inputs else inputs['input_ids'].clone()
            attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else torch.ones_like(inputs['input_ids'], dtype=torch.bool)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
