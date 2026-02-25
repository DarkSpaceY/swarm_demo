import logging
from datasets import load_dataset
from config_loader import get_config

CONFIG = get_config()

def load_math_problems(dataset_name=None, split=None, num_samples=None, streaming=None, seed=None, shuffle_buffer=None):
    """
    加载并过滤适合 Stage 0 的数学题库。
    目前以 OpenR1-Math-220k 为例，选取其中的简单问题。
    """
    dataset_cfg = CONFIG["dataset"]
    if dataset_name is None:
        dataset_name = dataset_cfg["name_default"]
    if split is None:
        split = dataset_cfg["split_default"]
    if num_samples is None:
        num_samples = dataset_cfg["num_samples_default"]
    if streaming is None:
        streaming = dataset_cfg["streaming_default"]
    if seed is None:
        seed = dataset_cfg["seed_default"]
    if shuffle_buffer is None:
        shuffle_buffer = dataset_cfg["shuffle_buffer_default"]
    logging.info(f"Loading dataset: {dataset_name} ({split})")
    try:
        if dataset_name == "openai/gsm8k" and split == "test":
            dataset = load_dataset(dataset_name, "main", split=split, streaming=streaming)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return []

    if streaming and shuffle_buffer and shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)

    problems = []
    count = 0
    
    problem_field = dataset_cfg["problem_field"]
    answer_field = dataset_cfg["answer_field"]
    id_field = dataset_cfg["id_field"]
    empty_text = dataset_cfg["empty_text"]
    data_keys = CONFIG["data_keys"]
    for sample in dataset:
        if count >= num_samples:
            break
            
        # 简单过滤：只选取问题文本较短的，且包含明确答案的
        problem_text = sample.get(problem_field, empty_text)
        answer = sample.get(answer_field, empty_text)
        
        if problem_text and answer:
            problems.append({
                data_keys["problem"]: problem_text,
                data_keys["answer"]: answer,
                data_keys["id"]: sample.get(id_field, str(count))
            })
            count += 1
            
    logging.info(f"Loaded {len(problems)} math problems.")
    return problems
