from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, load_from_disk


def truncate_back_no_signature(d):
    line = d.split('\n')
    code = []
    for l in line:
        if len(l.strip()) == 0:
            code.append(l)
            continue
        indent = len(l) - len(l.lstrip())
        if indent == 0:
            break
        else:
            code.append(l)

    return '\n'.join(code)


def load_dataset_my(dataset_name):
    if dataset_name == "MBPP":
        dataset =load_dataset("mbpp", "sanitized")
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif dataset_name == "HumanEval":
        dataset =load_dataset("openai/openai_humaneval")
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif dataset_name == "CodeForces2305":
        dataset = load_from_disk("data/CodeForces2305")            
    else:
        raise ValueError("dataset_name not found")
    return dataset


def load_dataset_map_my(dataset_name):
    if dataset_name == "MBPP":
        dataset =load_dataset("mbpp", "sanitized")
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif dataset_name == "HumanEval":
        dataset =load_dataset("openai/openai_humaneval")
        dataset = concatenate_datasets([dataset[k] for k in dataset.keys()])
    elif dataset_name == "CodeForces2305":
        dataset = load_from_disk("data/CodeForces2305")
    
    dataset_map = {}
    for i in tqdm(range(len(dataset))):
        dataset_map[dataset[i]["task_id"]] = dataset[i]
    return dataset_map