import math
import os
import argparse
import csv
import json
import logging
import pickle
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer
from src.ood_model_selector import RobertaForSelector_inference


set_seed(42)
MAX_GENERATION_LENGTH = 300


# ---- OOD helpers (from eval_o3.py) ----

def gmm_cdf(x, gmm):
    w = gmm.weights_
    m = gmm.means_.flatten()
    s = np.sqrt(gmm.covariances_.flatten())
    return np.sum([wi * norm.cdf(x, mi, si) for wi, mi, si in zip(w, m, s)])

def obtain_weights(input_x, gmm, x0):
    cp_x = gmm_cdf(input_x, gmm)
    cp_sym = gmm_cdf(2 * x0 - input_x, gmm)
    cp_sum = (1 - max(cp_x, cp_sym) + min(cp_x, cp_sym)) * 10
    w = math.exp(cp_sum - 2) / (1 + math.exp(cp_sum - 2))
    if w > 0.9: return 1.2
    elif 0.3 < w <= 0.4: return w
    else: return 0


class DeltaWeightManager:
    """Forward-hook hệ thống: output = base_out + w(x) * (W_prod - W_base) · x"""
    def __init__(self):
        self.ood_weight = 0
        self._hooks = []

    @staticmethod
    def compute_and_register(base_model, prod_model, target_names):
        mgr = DeltaWeightManager()
        prod_mods = dict(prod_model.named_modules())
        cnt = 0
        for name, base_mod in base_model.named_modules():
            if not isinstance(base_mod, nn.Linear) or not any(t in name for t in target_names):
                continue
            if name not in prod_mods:
                continue
            dw = (prod_mods[name].weight.data.cpu() - base_mod.weight.data.cpu()).clone()
            db = None
            if base_mod.bias is not None and prod_mods[name].bias is not None:
                db = (prod_mods[name].bias.data.cpu() - base_mod.bias.data.cpu()).clone()
            dev, dt = base_mod.weight.device, base_mod.weight.dtype
            dw = dw.to(device=dev, dtype=dt)
            if db is not None: db = db.to(device=dev, dtype=dt)
            base_mod.register_forward_hook(mgr._make_hook(dw, db))
            cnt += 1
        print(f"Registered {cnt} delta-W hooks")
        return mgr

    def _make_hook(self, dw, db):
        ref = self
        def hook(mod, inp, out):
            w = ref.ood_weight
            if isinstance(w, (int, float)) and w == 0: return out
            return out + w * F.linear(inp[0], dw, db)
        return hook

    def set_weight(self, w): self.ood_weight = w



def sample_code_from_llm(args, prompt, model, tokenizer):
    completions = []

    if tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False, verbose=False) 
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, verbose=False) 
        
    input_ids = torch.tensor([input_ids]).to(model.device)
    eos_token = tokenizer.eos_token_id

    num_return_sequences = args.acctual_num_samples
    if args.temperature == 0.0:
        args.num_samples = 1
        num_return_sequences = 1

    model.eval()

    # 2. FIX LOOPS: Tính tổng số vòng lặp bằng math.ceil để không bị làm tròn xuống
    loops = math.ceil(args.num_samples / num_return_sequences)

    for _ in range(loops):
        # 3. FIX OVER-GENERATION: Tính toán số sample cần sinh cho batch hiện tại
        # Tránh trường hợp vòng lặp cuối bị sinh thừa sample
        current_batch_size = min(num_return_sequences, args.num_samples - len(completions))
        
        if current_batch_size <= 0:
            break

        try:
            if args.temperature > 0:
                tokens = model.generate(
                    input_ids,
                    do_sample=True,
                    num_return_sequences=current_batch_size,
                    max_length=input_ids.shape[1] + MAX_GENERATION_LENGTH,
                    temperature=args.temperature,
                    use_cache=True,
                    top_k=args.topk,
                    top_p=args.topp,
                    eos_token_id=eos_token,
                    pad_token_id=eos_token
                )
            else:
                tokens = model.generate(
                    input_ids,
                    num_return_sequences=1,
                    max_length=input_ids.shape[1] + MAX_GENERATION_LENGTH,
                    use_cache=True,
                    do_sample=False,
                    eos_token_id=eos_token,
                    pad_token_id=eos_token
                )

            for tok in tokens:
                # Cắt bỏ phần prompt, chỉ giải mã phần model sinh ra
                tok = tok[input_ids.shape[1]:]
                text = tokenizer.decode(tok, skip_special_tokens=True)
                text = text.replace('Ċ', '\n').replace('Ġ', ' ')
                completions.append(text)
                
        except RuntimeError as e:
            logging.error(f"Could not sample from model: {e}")

    return completions


def load_model_tokenizer(args, model_name, model_path):

    if model_path:
        model_path = model_path
    else:
        model_path = model_name
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True, 
        torch_dtype="auto", 
        device_map="auto"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generate_code_fn = lambda args, prompt: sample_code_from_llm(
        args, prompt, model, tokenizer
    )

    return generate_code_fn, tokenizer


def generate_code_for_tasks(args, except_tasks, save_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # open save file
    f = open(save_file, "a")
    csv_file = save_file.replace(".jsonl", ".csv")

    manager = None
    ood_components = None  # (ood_model, ood_tokenizer, ood_clr, ood_gmm, ood_x0)

    if getattr(args, 'ood_weights', None):
        # --- PROD+OOD mode: base on GPU, delta-W hooks ---
        print("\n=== Loading Base + PROD + OOD ===")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        prod_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
        tgt = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        manager = DeltaWeightManager.compute_and_register(base_model, prod_model, tgt)
        del prod_model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
        base_model.eval()
        generate_code_fn = lambda a, p: sample_code_from_llm(a, p, base_model, tokenizer)

        # Load OOD detector
        ood_tok = RobertaTokenizer.from_pretrained(args.ood_base_model)
        t = [x for x in args.ood_type.split("_") if x]
        t = t[0] if t else "all"
        wp = os.path.join(args.ood_weights, f"{args.ood_setting_name}_{t}_ood_{args.ood_setting_name}")
        ood_mdl = RobertaForSelector_inference(args.ood_base_model, lora_path=wp+"_roberta_ocsvm", projection_dim=100).to(device)
        with open(wp+"_ocsvm.pkl","rb") as fp: ood_clr = pickle.load(fp)
        with open(wp+"_gmm_w_ocsvm.pkl","rb") as fp: ood_gmm = pickle.load(fp)
        with open(wp+"_threshold_ocsvm.json") as fp: th = json.load(fp)
        
        # Load additional lists for get_unsup_Mah_score_s
        ood_mean = torch.load(wp+"_mean_list_ocsvm.pt", map_location=torch.device(device))
        ood_prec = torch.load(wp+"_precision_list_ocsvm.pt", map_location=torch.device(device))
        ood_fea = torch.load(wp+"_fea_list_ocsvm.pt", map_location=torch.device(device))

        ood_components = (ood_mdl, ood_tok, ood_clr, ood_gmm, th[0], ood_mean, ood_prec, ood_fea)
        print("OOD detector loaded.")
    else:
        # --- Normal mode ---
        generate_code_fn, tokenizer = load_model_tokenizer(args, args.model_name, args.model_path)

    # load dataset
    dataset = load_dataset("openai/openai_humaneval")
    # dataset = dataset['test'].select(range(10))
    dataset = dataset['test']
    
    csv_data = []
    
    if os.path.exists(save_file):
        with open(save_file, "r") as f_read:
            for line in f_read:
                csv_data.append(json.loads(line))
                
    for i in tqdm(range(len(dataset))):
        task_id = dataset[i]["task_id"]

        if (task_id in except_tasks):
            continue

        # construct prompt
        prompt = dataset[i]["prompt"]

        # Set OOD weight if in OOD mode
        if manager and ood_components:
            ood_mdl, ood_tok, ood_clr, ood_gmm, ood_x0, ood_mean, ood_prec, ood_fea = ood_components
            enc = ood_tok(prompt, padding=True, truncation=True, max_length=512, return_tensors='pt')
            
            with torch.no_grad():
                mah_score = ood_mdl.get_unsup_Mah_score_s(enc, ood_mean, ood_prec, ood_fea)[:, 1:]
            
            test_score = ood_clr.score_samples(mah_score)
            w = obtain_weights(test_score[0], ood_gmm, ood_x0)
            manager.set_weight(w)

        for completion in generate_code_fn(args, prompt):
            # Fix indentation: Tokenizer thường sinh dòng đầu thiếu 1 dấu cách.
            # Phát hiện indent chuẩn từ prompt và căn chỉnh dòng đầu completion.
            prompt_lines = prompt.split('\n')
            expected_indent = 4  # Mặc định cho HumanEval
            for line in reversed(prompt_lines):
                if line.strip():
                    expected_indent = len(line) - len(line.lstrip())
                    break

            comp_lines = completion.split('\n')
            for idx, line in enumerate(comp_lines):
                if line.strip():  # Tìm dòng không trống đầu tiên
                    actual_indent = len(line) - len(line.lstrip())
                    if actual_indent < expected_indent:
                        comp_lines[idx] = ' ' * (expected_indent - actual_indent) + line
                    break
            completion = '\n'.join(comp_lines)

            output ={
                    "task_id": task_id,
                    "prompt": prompt,
                    "completion": completion,
                }
            f.write(json.dumps(output) + "\n")
            f.flush()
            
            csv_data.append(output)
    
    f.close()
    
    if csv_data:
        with open(csv_file, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=["task_id", "prompt", "completion"])
            writer.writeheader()
            for row in csv_data:
                writer.writerow({k: row[k] for k in ["task_id", "prompt", "completion"]})
        print(f"Đã lưu kết quả Model Utility ra file CSV tại: {csv_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="CodeLlama-7b-hf")
    parser.add_argument("--model_path", default=None, help="PROD checkpoint path (full fine-tuned model).")
    parser.add_argument("--ood_weights", default=None, type=str, help="OOD checkpoint dir. If set, uses PROD+OOD mode.")
    parser.add_argument("--ood_base_model", default="microsoft/codebert-base", type=str)
    parser.add_argument("--ood_type", default="_all", type=str)
    parser.add_argument("--ood_setting_name", default="codellama", type=str)
    parser.add_argument("--dataset", default="HumanEval", type=str)    
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--acctual-num-samples", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--topp", default=None, type=float)
    parser.add_argument("--topk", default=None, type=int)
    parser.add_argument("--few-shot", default=0, type=int)
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_name = args.model_name.split("/")[-1]
    save_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_{model_name}_temp{args.temperature}_topp{args.topp}_topk{args.topk}_samples{args.num_samples}_{args.few_shot}shot_{args.output_file_suffix}.jsonl",
    )
    
    except_tasks = []
    if os.path.exists(save_file):
        print(f"File {save_file} already exists in {args.output_dir}.")
        lines = open(save_file).readlines()
        for line in lines:
            task_id = json.loads(line)["task_id"]
            if task_id not in except_tasks:
                except_tasks.append(task_id)

    generate_code_for_tasks(args, except_tasks, save_file)


if __name__ == "__main__":
    main(parse_args())