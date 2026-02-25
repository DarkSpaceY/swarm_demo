import json
import logging
import os
from config_loader import get_config

CONFIG = get_config()

def normalize_chat_history(chat_history):
    if not isinstance(chat_history, list):
        return []
    normalized = []
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", CONFIG["common"]["empty_text"])
        if role not in set(CONFIG["agent"]["roles"]):
            continue
        if content is None:
            content = CONFIG["common"]["empty_text"]
        elif not isinstance(content, str):
            content = str(content)
        normalized.append({"role": role, "content": content})
    return normalized

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
    
    data_keys = CONFIG["data_keys"]
    for step in trajectory:
        agent_id = step[data_keys["agent_id"]]
        chat_history = step[data_keys["input"]] # List[Dict]
        model_output = step[data_keys["output"]]
        
        messages = normalize_chat_history(chat_history)
        messages.append({"role": "assistant", "content": model_output})
        sample = {
            data_keys["messages"]: messages,
            data_keys["agent_id"]: agent_id,
            data_keys["round"]: step[data_keys["round"]]
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

def save_jsonl_data(records, filepath):
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in records:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    logging.info(f"Saved {len(records)} records to {filepath}")

def _group_steps_by_agent_round(trajectory):
    grouped = {}
    data_keys = CONFIG["data_keys"]
    for step in trajectory:
        agent_id = step.get(data_keys["agent_id"])
        round_id = step.get(data_keys["round"])
        if agent_id is None or round_id is None:
            continue
        grouped.setdefault(agent_id, {}).setdefault(round_id, []).append(step)
    return grouped

def convert_trajectories_to_preference(candidate_trajectories, consensus_idx, max_pairs_per_agent=None, is_correct=None):
    if consensus_idx is None or consensus_idx < 0 or consensus_idx >= len(candidate_trajectories):
        return []
    
    # 如果提供了 is_correct，我们可以区分“成功案例”和“失败案例”
    # 如果没有提供，默认按照原来的逻辑（即假设共识是对的）
    is_success_case = is_correct if is_correct is not None else True
    
    chosen_traj = candidate_trajectories[consensus_idx]
    chosen_steps = chosen_traj.get("trajectory", [])
    chosen_grouped = _group_steps_by_agent_round(chosen_steps)
    if not chosen_grouped:
        return []
        
    records = []
    for agent_id, rounds in chosen_grouped.items():
        pairs_for_agent = 0
        for round_id, chosen_round_steps in rounds.items():
            for chosen_step in chosen_round_steps:
                chosen_output = chosen_step.get("output", "")
                chosen_input = chosen_step.get("input")
                if not chosen_output or chosen_input is None:
                    continue
                
                # 在成功案例中，强化共识选项，远离其他选项
                # 在失败案例中，远离共识选项，强化那些可能投给其他选项的路径
                for idx, traj in enumerate(candidate_trajectories):
                    if idx == consensus_idx:
                        continue
                    
                    rejected_grouped = _group_steps_by_agent_round(traj.get("trajectory", []))
                    rejected_round_steps = rejected_grouped.get(agent_id, {}).get(round_id, [])
                    
                    for rejected_step in rejected_round_steps:
                        rejected_output = rejected_step.get("output", "")
                        if not rejected_output:
                            continue
                        
                        data_keys = CONFIG["data_keys"]
                        
                        if is_success_case:
                            # 成功案例：强化共识 (chosen_output)，远离其他 (rejected_output)
                            records.append({
                                data_keys["messages"]: normalize_chat_history(chosen_input),
                                data_keys["chosen"]: chosen_output,
                                data_keys["rejected"]: rejected_output,
                                data_keys["agent_id"]: agent_id,
                                data_keys["round"]: round_id,
                            })
                        else:
                            # 失败案例：远离共识 (chosen_output)，强化其他 (rejected_output)
                            # 注意：这里我们反转了 chosen 和 rejected
                            records.append({
                                data_keys["messages"]: normalize_chat_history(chosen_input),
                                data_keys["chosen"]: rejected_output,
                                data_keys["rejected"]: chosen_output,
                                data_keys["agent_id"]: agent_id,
                                data_keys["round"]: round_id,
                            })
                            
                        pairs_for_agent += 1
                        if max_pairs_per_agent and pairs_for_agent >= max_pairs_per_agent:
                            break
                    if max_pairs_per_agent and pairs_for_agent >= max_pairs_per_agent:
                        break
                if max_pairs_per_agent and pairs_for_agent >= max_pairs_per_agent:
                    break
            if max_pairs_per_agent and pairs_for_agent >= max_pairs_per_agent:
                break
    return records

def convert_judge_trajectories_to_preference(judge_trajectories, consensus_idx, candidate_correctness, max_pairs_per_agent=None, ambiguous=False):
    """
    专门为 Judge 构造偏好数据。
    根据 Oracle 对所有候选的验证结果（candidate_correctness）来判定 Judge 的好坏。
    - 负样本（Rejected）：当存在正确候选时，Judge 却投给了错误候选（“完全相反”）。
    - 正样本（Chosen）：Judge 投给了正确候选，或者在没有正确候选时投给了任意候选（“平庸”）。
    """
    if consensus_idx is None or not judge_trajectories or not candidate_correctness:
        return []
        
    records = []
    data_keys = CONFIG["data_keys"]
    
    # 检查是否存在正确候选
    has_correct_candidate = any(candidate_correctness)
    
    # 将 Judge 轨迹分类为“好/平庸（Positive）”和“坏（Negative）”
    positive_trajs = []
    negative_trajs = []
    
    for j_traj in judge_trajectories:
        vote_val = j_traj.get(data_keys["vote_val"])
        if vote_val is None or vote_val < 0 or vote_val >= len(candidate_correctness):
            continue
            
        is_vote_correct = candidate_correctness[vote_val]
        
        if not is_vote_correct and has_correct_candidate:
            # 存在正确答案却选了错的 -> 坏 Judge
            negative_trajs.append(j_traj)
        else:
            # 选对了，或者大家都错（平庸） -> 好/平庸 Judge，归类为正样本
            positive_trajs.append(j_traj)
            
    if not positive_trajs or not negative_trajs:
        # 如果本轮 Judge 表现过于一致（全是好的或全是坏的），无法构造对比对
        # 注意：这里可以考虑引入 SFT 或者与一个随机/空轨迹对比，但按 DPO 逻辑先跳过
        return []

    # 构造对比对
    for p_traj in positive_trajs:
        p_steps = p_traj.get(data_keys["trajectory"], [])
        if not p_steps: continue
        p_output = p_steps[-1].get("output", "")
        p_input = p_steps[-1].get("input", [])
        
        for n_traj in negative_trajs:
            n_steps = n_traj.get(data_keys["trajectory"], [])
            if not n_steps: continue
            n_output = n_steps[-1].get("output", "")
            
            records.append({
                data_keys["messages"]: normalize_chat_history(p_input),
                data_keys["chosen"]: p_output,
                data_keys["rejected"]: n_output,
                "is_ambiguous": ambiguous
            })
                
            if max_pairs_per_agent and len(records) >= max_pairs_per_agent:
                break
        if max_pairs_per_agent and len(records) >= max_pairs_per_agent:
            break
            
    return records
