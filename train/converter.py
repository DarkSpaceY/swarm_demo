import json
import logging
import os

def build_chatml_prompt(chat_history):
    if not chat_history:
        return "<|assistant|>\n"
    chunks = []
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            chunks.append(f"<|system|>\n{content}")
        elif role == "user":
            chunks.append(f"<|user|>\n{content}")
        elif role == "assistant":
            chunks.append(f"<|assistant|>\n{content}")
    chunks.append("<|assistant|>\n")
    return "\n".join(chunks)

def convert_trajectory_to_sft(trajectory):
    """
    将单条成功轨迹转换为 SFT 训练数据。
    每个 Agent 的每一步输出都可以作为一条独立的训练样本。
    
    Args:
        trajectory: List[Dict], 包含每一步的 agent_id, input, output
    
    Returns:
        List[Dict], 适配 Unsloth/HuggingFace 的 SFT 数据格式
        Example:
        {"instruction": "系统提示 + 用户输入", "input": "", "output": "模型输出"}
    """
    sft_data = []
    
    for step in trajectory:
        agent_id = step['agent']
        chat_history = step['input'] # List[Dict]
        model_output = step['output']
        
        prompt = build_chatml_prompt(chat_history)
        sample = {
            "instruction": prompt,
            "input": "",
            "output": model_output,
            "agent_id": agent_id,
            "round": step['round']
        }
        sft_data.append(sample)
        
    return sft_data

def save_sft_data(sft_data, filepath):
    """保存为 JSONL 格式"""
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in sft_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    logging.info(f"Saved {len(sft_data)} SFT samples to {filepath}")
