import math
import os
import argparse
import csv
import json
import logging
import pprint
import torch
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


set_seed(42)
MAX_GENERATION_LENGTH = 300


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
        args, prompt, model, tokenizer
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