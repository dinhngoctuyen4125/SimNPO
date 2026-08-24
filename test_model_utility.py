import math
import os
import sys
import argparse
import csv
import json
import logging
import pprint
import pickle
import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import norm

from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaTokenizer
from peft import PeftModel


set_seed(42)
MAX_GENERATION_LENGTH = 300


# --- OOD Utility Functions (from eval_o3.py) ---

def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)


def cumulative_probability(x, gmm):
    return gmm_cdf(x, gmm)


def symmetric_cumulative_probability(x, x0, gmm):
    symmetric_x = 2 * x0 - x
    return gmm_cdf(symmetric_x, gmm)


def obtain_weights(input_x, gmm, x0):
    cp_x = cumulative_probability(input_x, gmm)
    cp_symmetric_x = symmetric_cumulative_probability(input_x, x0, gmm)
    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    scaling_factor = 10
    cp_sum *= scaling_factor
    range_th = 2
    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))
    if w_res > 0.9:
        w_res = 1.2
    elif w_res <= 0.4 and w_res > 0.3:
        w_res = w_res
    else:
        w_res = 0
    return w_res


def load_ood_components(args, device):
    """Load all OOD detector components."""
    from src.ood_model_selector import RobertaForSelector_inference
    from transformers import RobertaTokenizer

    ood_base_model = args.ood_base_model
    ood_tokenizer = RobertaTokenizer.from_pretrained(ood_base_model)

    types = args.ood_type.split("_")
    ood_types = [t for t in types if len(t) > 0]
    ood_type = "ocsvm"

    ood_models = []
    ood_clrs = []
    ood_x0 = []
    ood_mean_lists = []
    ood_precision_lists = []
    ood_fea_lists = []
    ood_gmm_w_cls = []

    for t in ood_types:
        prefix = os.path.join(args.ood_weights, f"{args.ood_setting_name}_{t}_ood_{args.ood_setting_name}")

        roberta_path = prefix + f"_roberta_{ood_type}"
        ocsvm_path = prefix + f"_{ood_type}.pkl"
        threshold_path = prefix + f"_threshold_{ood_type}.json"
        mean_list_path = prefix + f"_mean_list_{ood_type}.pt"
        precision_list_path = prefix + f"_precision_list_{ood_type}.pt"
        fea_list_path = prefix + f"_fea_list_{ood_type}.pt"
        gmm_w_path = prefix + f"_gmm_w_{ood_type}.pkl"

        ood_models.append(RobertaForSelector_inference(ood_base_model, lora_path=roberta_path, projection_dim=100).to(device))
        with open(ocsvm_path, "rb") as f:
            ood_clrs.append(pickle.load(f))
        with open(gmm_w_path, "rb") as f:
            ood_gmm_w_cls.append(pickle.load(f))
        with open(threshold_path) as f:
            threshold = json.load(f)
        ood_x0.append(threshold[0])
        ood_mean_lists.append(torch.load(mean_list_path, map_location=torch.device(device)))
        ood_precision_lists.append(torch.load(precision_list_path, map_location=torch.device(device)))
        ood_fea_lists.append(torch.load(fea_list_path, map_location=torch.device(device)))

    return {
        'models': ood_models,
        'clrs': ood_clrs,
        'gmm_w_cls': ood_gmm_w_cls,
        'x0': ood_x0,
        'mean_lists': ood_mean_lists,
        'precision_lists': ood_precision_lists,
        'fea_lists': ood_fea_lists,
        'tokenizer': ood_tokenizer,
    }


def compute_ood_weight(prompt, model, ood_components, device):
    """Compute OOD weight for a single prompt and set it on the model."""
    ood_tokenizer = ood_components['tokenizer']
    ood_input = ood_tokenizer(
        [prompt], padding='max_length', truncation=True, max_length=512, return_tensors="pt"
    )

    max_w = 0.0
    for j in range(len(ood_components['models'])):
        mah_score = ood_components['models'][j].get_unsup_Mah_score_s(
            ood_input,
            ood_components['mean_lists'][j],
            ood_components['precision_lists'][j],
            ood_components['fea_lists'][j]
        )[:, 1:]
        test_score = ood_components['clrs'][j].score_samples(mah_score)
        w = obtain_weights(test_score[0], ood_components['gmm_w_cls'][j], ood_components['x0'][j])
        max_w = max(max_w, w)

    ood_weight_tensor = torch.tensor([max_w], dtype=torch.bfloat16).to(device)
    model.init_oodweight(ood_weight=[1, ood_weight_tensor])
    return max_w


def sample_code_from_llm(args, prompt, model, tokenizer, ood_components=None):
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

    # OOD mode: force num_return_sequences=1 for compatibility with hacked model
    if ood_components:
        num_return_sequences = 1
        compute_ood_weight(prompt, model, ood_components, input_ids.device)

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

    ood_components = None

    if args.ood_weights:
        # OOD mode: use hacked LlamaForCausalLM_ood + hacked PeftModel
        from src.peft_model_hacked_o import PeftModel as PeftModelOOD
        from src.modeling_llama_hacked_o import LlamaForCausalLM_ood

        device = "cuda" if torch.cuda.is_available() else "cpu"

        lora_target_modules = ["q_proj", "v_proj"]
        config = AutoConfig.from_pretrained(model_path)
        config.lora_target_modules = lora_target_modules
        config.orthogonal_loss = False
        config.orthogonal_loss_weight = 0.1

        model = LlamaForCausalLM_ood.from_pretrained(
            model_path,
            config=config,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModelOOD.from_pretrained(
            model,
            args.lora_path,
            torch_dtype=torch.bfloat16,
        )
        model.init_olora(orthogonal_loss=False, olora_weights={})
        model.init_active_adapters_d(active_adapters_d=['default'])

        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side='left')
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model.half()
        model.eval()

        # Load OOD components
        print("[*] Loading OOD components...")
        ood_components = load_ood_components(args, device)
        print("[*] OOD components loaded.")

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            low_cpu_mem_usage=True, 
            torch_dtype="auto", 
            device_map="auto"
        )

        if args.lora_path:
            model = PeftModel.from_pretrained(model, args.lora_path)
            model = model.merge_and_unload()

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    generate_code_fn = lambda args, prompt: sample_code_from_llm(
        args, prompt, model, tokenizer, ood_components
    )

    return generate_code_fn, tokenizer


def generate_code_for_tasks(args, except_tasks, save_file):

    # open save file
    f = open(save_file, "a")
    csv_file = save_file.replace(".jsonl", ".csv")

    # load model
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
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter checkpoint. If provided, will merge LoRA weights into base model.")
    parser.add_argument("--model_path", default=None, help="Directory where a pre-trained LLM or fine-tuned LLM is saved. If None, will load from huggingface cache.",)
    parser.add_argument("--dataset", default="HumanEval", type=str)    
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--acctual-num-samples", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--topp", default=None, type=float)
    parser.add_argument("--topk", default=None, type=int)
    parser.add_argument("--few-shot", default=0, type=int)
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    # OOD arguments
    parser.add_argument("--ood_weights", default=None, help="Path to OOD checkpoint directory. If provided, enables soft-weighted inference.")
    parser.add_argument("--ood_base_model", default="microsoft/codebert-base", help="Base model for OOD detector.")
    parser.add_argument("--ood_type", default="_all", help="OOD type (e.g. '_all', '_torch').")
    parser.add_argument("--ood_setting_name", default="codellama", help="OOD setting name.")
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