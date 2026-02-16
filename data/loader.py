import logging
from datasets import load_dataset

def load_math_problems(dataset_name="open-r1/OpenR1-Math-220k", split="train", num_samples=100, streaming=True, seed=42, shuffle_buffer=1000):
    """
    加载并过滤适合 Stage 0 的数学题库。
    目前以 OpenR1-Math-220k 为例，选取其中的简单问题。
    """
    logging.info(f"Loading dataset: {dataset_name} ({split})")
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return []

    if streaming and shuffle_buffer and shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)

    problems = []
    count = 0
    
    for sample in dataset:
        if count >= num_samples:
            break
            
        # 简单过滤：只选取问题文本较短的，且包含明确答案的
        problem_text = sample.get('problem', '')
        answer = sample.get('answer', '')
        
        if problem_text and answer:
            problems.append({
                "problem": problem_text,
                "answer": answer,
                "id": sample.get('uuid', str(count))
            })
            count += 1
            
    logging.info(f"Loaded {len(problems)} math problems.")
    return problems
