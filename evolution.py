import argparse
import logging
import os
import json
import re
import time
import random
import concurrent.futures
from collections import Counter
import torch
from unsloth import FastLanguageModel

from core.agent import Agent
from core.env import Environment
from core.oracle import Oracle
from data.loader import load_math_problems
from train.converter import convert_trajectory_to_sft, save_sft_data
from train.finetuner import SwarmTrainer, resolve_local_files_only

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = None
TOKENIZER = None
GEN_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
}

def init_inference_model(model_name, max_seq_length, load_in_4bit, cache_dir, local_files_only):
    global MODEL, TOKENIZER
    resolved_local_only = resolve_local_files_only(model_name, cache_dir, local_files_only)
    MODEL, TOKENIZER = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        cache_dir=cache_dir,
        local_files_only=resolved_local_only,
    )
    MODEL = FastLanguageModel.for_inference(MODEL)

def inference_model_fn(agent_id, history):
    try:
        if TOKENIZER is None or MODEL is None:
            raise RuntimeError("Model not initialized")
        if hasattr(TOKENIZER, "apply_chat_template"):
            prompt = TOKENIZER.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in history]) + "\nassistant:"
        inputs = TOKENIZER([prompt], return_tensors="pt").to(MODEL.device)
        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=GEN_CONFIG["max_new_tokens"],
                do_sample=GEN_CONFIG["do_sample"],
                temperature=GEN_CONFIG["temperature"],
            )
        decoded = TOKENIZER.batch_decode(outputs, skip_special_tokens=True)[0]
        if decoded.startswith(prompt):
            return decoded[len(prompt):].strip()
        return decoded.strip()
    except Exception as e:
        logging.error(f"Model inference failed for {agent_id}: {e}")
        return "无法处理。"

# --- Unified System Prompt (Homogeneous) ---

def get_system_prompt(agent_id):
    return (
        "你是 {agent_id}。\n"
        "合作者列表：Agent_0, Agent_1, Agent_2, Agent_3, Agent_4。\n"
        "输出格式：\n"
        "1. 协作消息：@Agent_ID 内容\n"
        "2. 最终答案：[FIN] 答案\n"
    ).format(agent_id=agent_id)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Solver Logic ---

def run_solver_trial(problem_text, trial_id, max_rounds):
    """
    运行 Solver Swarm 尝试解决问题。
    Input: Math Problem
    Output: [FIN] Numerical Answer
    """
    logging.info(f"Starting Solver Trial {trial_id}...")
    
    agents = [Agent(f"Agent_{i}", get_system_prompt(f"Agent_{i}")) for i in range(5)]
    env = Environment(agents, problem_text, max_rounds=max_rounds)
    
    result = env.run(inference_model_fn)
    result["trial_id"] = trial_id
    return result

# --- Judge Logic ---

def format_judge_task(problem_text, candidate_trajectories, gt_answer):
    candidates_text = ""
    for idx, traj in enumerate(candidate_trajectories):
        final_ans = traj.get("final_answer", "N/A")
        traj_content = ""
        for step in traj["trajectory"]:
            if step['agent'].startswith("Agent"):
                traj_content += f"{step['agent']}: {step['output'][:200]}...\n"
        
        candidates_text += f"选项 {idx} (Answer: {final_ans}):\n{traj_content}\n"

    judge_task = (
        "原始题目：\n"
        "{problem}\n"
        "\n"
        "标准答案：\n"
        "{gt_answer}\n"
        "\n"
        "候选选项：\n"
        "{candidates}\n"
        "请对候选选项进行从优到劣的排序，输出完整排序。\n"
        "输出格式：[FIN] 编号,编号,编号,编号"
    ).format(
        problem=problem_text,
        gt_answer=gt_answer,
        candidates=candidates_text
    )
    return judge_task

def run_judge_swarm_trial(judge_task, trial_id, max_rounds):
    """
    运行 Judge Swarm 进行评判。
    Input: Multiple Choice Task (constructed from trajectories)
    Output: [FIN] Index
    """
    logging.info(f"Starting Judge Swarm Trial {trial_id}...")
    
    # Judge Swarm 与 Solver Swarm 结构完全一致
    agents = [Agent(f"Agent_{i}", get_system_prompt(f"Agent_{i}")) for i in range(5)]
    env = Environment(agents, judge_task, max_rounds=max_rounds)
    
    result = env.run(inference_model_fn)
    return result

def extract_ranking(result_text, num_candidates):
    if not result_text:
        return None
    text = str(result_text)
    match = re.search(r'\[FIN\]\s*(.*)', text)
    if match:
        text = match.group(1).strip()
    numbers = re.findall(r'-?\d+', text)
    ranking = []
    for n in numbers:
        idx = int(n)
        if 0 <= idx < num_candidates and idx not in ranking:
            ranking.append(idx)
    if len(ranking) == num_candidates:
        return ranking
    return None

# --- Main Evolution Loop ---

def run_evolution_batch(
    batch_id,
    num_problems=5,
    num_solver_samples=4,
    num_judge_samples=3,
    solver_max_workers=4,
    judge_max_workers=3,
    max_rounds=3,
    dataset_name="open-r1/OpenR1-Math-220k",
    dataset_split="train",
    dataset_streaming=True,
    dataset_seed=42,
    dataset_shuffle_buffer=1000,
):
    """
    运行一个进化批次 (Batch)。
    返回: (accuracy, solver_data, judge_data)
    """
    logging.info(f"=== Starting Batch {batch_id} ===")
    
    problems = load_math_problems(
        dataset_name=dataset_name,
        split=dataset_split,
        num_samples=num_problems,
        streaming=dataset_streaming,
        seed=dataset_seed,
        shuffle_buffer=dataset_shuffle_buffer,
    )
    if not problems:
        logging.error("No problems loaded. Check dataset access and configuration.")
        return 0.0, [], []
    oracle = Oracle()
    
    batch_correct_count = 0
    batch_solver_data = [] # 存储本轮产生的优质 Solver 轨迹
    batch_judge_data = []  # 存储本轮产生的 Judge 轨迹 (无论对错，后续筛选)
    
    for prob_idx, prob_data in enumerate(problems):
        problem_text = prob_data["problem"]
        gt_answer = prob_data["answer"]
        logging.info(f"Problem {prob_idx}: {problem_text[:50]}...")
        
        # 1. 并行运行 Solver Swarms (生成 4 条轨迹)
        candidate_trajectories = []
        cpu_limit = os.cpu_count() or solver_max_workers
        solver_workers = min(num_solver_samples, solver_max_workers, cpu_limit) if num_solver_samples > 0 else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=solver_workers) as executor:
            futures = {executor.submit(run_solver_trial, problem_text, i, max_rounds): i for i in range(num_solver_samples)}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["status"] == "SUCCESS":
                    candidate_trajectories.append(res)
        
        if not candidate_trajectories:
            logging.warning("No valid trajectories generated.")
            continue
            
        # 2. 构造 Judge 任务
        judge_task_text = format_judge_task(problem_text, candidate_trajectories, gt_answer)
        
        # 3. 并行运行 Judge Swarms (3 个分身进行投票)
        votes = []
        judge_trajectories = []
        
        judge_workers = min(num_judge_samples, judge_max_workers, cpu_limit) if num_judge_samples > 0 else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=judge_workers) as executor:
            futures = {executor.submit(run_judge_swarm_trial, judge_task_text, i, max_rounds): i for i in range(num_judge_samples)}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["status"] == "SUCCESS":
                    ranking = extract_ranking(res["final_answer"], len(candidate_trajectories))
                    if ranking is not None:
                        votes.append(ranking[0])
                        judge_trajectories.append({
                            "task": judge_task_text,
                            "trajectory": res["trajectory"],
                            "final_answer": ",".join(str(i) for i in ranking),
                            "vote_val": ranking[0]
                        })

        # 4. 达成共识 (Majority Vote)
        if not votes:
            logging.warning("No valid votes from Judge Swarm.")
            consensus_idx = -1
        else:
            vote_counts = Counter(votes)
            consensus_idx = vote_counts.most_common(1)[0][0]
            logging.info(f"Judge Consensus: {consensus_idx} (Votes: {votes})")
        
        if 0 <= consensus_idx < len(candidate_trajectories):
            selected_traj = candidate_trajectories[consensus_idx]
            if oracle.verify(selected_traj["final_answer"], gt_answer):
                batch_correct_count += 1
            sft_entry = convert_trajectory_to_sft(selected_traj["trajectory"])
            batch_solver_data.extend(sft_entry)
        else:
            logging.info("Judge selected None (-1) or invalid index.")
            
        # 6. 收集 Judge 数据
        # 无论本题是否做对，我们都先收集 Judge 的数据
        # 这里的关键是：我们只收集那些“投给了共识结果”的 Judge 轨迹
        # 这样如果共识是对的（由下一轮准确率提升证明），那么这些 Judge 也就是“功臣”
        for j_traj in judge_trajectories:
            if j_traj["vote_val"] == consensus_idx:
                # 这是一个“符合共识”的 Judge 轨迹
                # 转换为 SFT 格式
                sft_entry = convert_trajectory_to_sft(j_traj["trajectory"])
                batch_judge_data.extend(sft_entry)

    accuracy = batch_correct_count / len(problems) if problems else 0
    return accuracy, batch_solver_data, batch_judge_data

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="swarm_run")
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument("--model-name", type=str, default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-cache-dir", type=str, default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--num-solver-samples", type=int, default=4)
    parser.add_argument("--num-judge-samples", type=int, default=3)
    parser.add_argument("--solver-max-workers", type=int, default=4)
    parser.add_argument("--judge-max-workers", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", type=str, default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--dataset-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-shuffle-buffer", type=int, default=1000)
    parser.add_argument("--gen-max-new-tokens", type=int, default=512)
    parser.add_argument("--gen-temperature", type=float, default=0.7)
    parser.add_argument("--train-lora", action="store_true")
    parser.add_argument("--lora-epochs", type=int, default=1)
    parser.add_argument("--lora-batch-size", type=int, default=2)
    parser.add_argument("--lora-grad-accum", type=int, default=4)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--lora-logging-steps", type=int, default=1)
    parser.add_argument("--lora-save-strategy", type=str, default="epoch")
    parser.add_argument("--lora-save-total-limit", type=int, default=2)
    parser.add_argument("--train-judge", action=argparse.BooleanOptionalAction, default=True)
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    GEN_CONFIG["max_new_tokens"] = args.gen_max_new_tokens
    GEN_CONFIG["temperature"] = args.gen_temperature

    init_inference_model(
        args.model_name,
        args.max_seq_length,
        args.load_in_4bit,
        args.model_cache_dir,
        args.local_files_only,
    )

    run_dir = os.path.join(args.output_dir, args.run_name)
    data_dir = os.path.join(run_dir, "data")
    solver_dir = os.path.join(data_dir, "solver")
    judge_dir = os.path.join(data_dir, "judge")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    os.makedirs(solver_dir, exist_ok=True)
    os.makedirs(judge_dir, exist_ok=True)

    trainer = None
    if args.train_lora:
        trainer = SwarmTrainer(
            model_name=args.model_name,
            output_dir=os.path.join(run_dir, "checkpoints"),
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            cache_dir=args.model_cache_dir,
            local_files_only=args.local_files_only,
        )

    prev_accuracy = 0.0

    for batch_id in range(args.num_batches):
        accuracy, solver_data, judge_data = run_evolution_batch(
            batch_id,
            num_problems=args.num_problems,
            num_solver_samples=args.num_solver_samples,
            num_judge_samples=args.num_judge_samples,
            solver_max_workers=args.solver_max_workers,
            judge_max_workers=args.judge_max_workers,
            max_rounds=args.max_rounds,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            dataset_streaming=args.dataset_streaming,
            dataset_seed=args.seed,
            dataset_shuffle_buffer=args.dataset_shuffle_buffer,
        )

        logging.info(f"Batch {batch_id} Result: Accuracy = {accuracy:.2%} (Prev: {prev_accuracy:.2%})")

        metric_record = {
            "batch_id": batch_id,
            "accuracy": accuracy,
            "prev_accuracy": prev_accuracy,
            "num_problems": args.num_problems,
            "num_solver_samples": args.num_solver_samples,
            "num_judge_samples": args.num_judge_samples,
            "timestamp": time.time(),
        }
        os.makedirs(run_dir, exist_ok=True)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_record, ensure_ascii=False) + "\n")

        solver_path = None
        judge_path = None

        if solver_data:
            solver_path = os.path.join(solver_dir, f"solver_batch_{batch_id}.jsonl")
            save_sft_data(solver_data, solver_path)
            logging.info(f"Saved {len(solver_data)} solver samples to {solver_path}")

        if accuracy > prev_accuracy and judge_data:
            sample_size = min(len(judge_data), len(solver_data)) if solver_data else len(judge_data)
            selected_judge_data = random.sample(judge_data, sample_size) if sample_size > 0 else judge_data
            judge_path = os.path.join(judge_dir, f"judge_batch_{batch_id}.jsonl")
            save_sft_data(selected_judge_data, judge_path)
            logging.info(f"Accuracy Improved! Saved {len(selected_judge_data)} judge samples to {judge_path}")
        else:
            logging.info("Accuracy did not improve. Discarding Judge data.")

        if trainer and solver_path:
            trainer.train_agent(
                "solver",
                solver_path,
                epochs=args.lora_epochs,
                batch_size=args.lora_batch_size,
                gradient_accumulation_steps=args.lora_grad_accum,
                learning_rate=args.lora_lr,
                logging_steps=args.lora_logging_steps,
                save_strategy=args.lora_save_strategy,
                save_total_limit=args.lora_save_total_limit,
            )

        if trainer and args.train_judge and judge_path:
            trainer.train_agent(
                "judge",
                judge_path,
                epochs=args.lora_epochs,
                batch_size=args.lora_batch_size,
                gradient_accumulation_steps=args.lora_grad_accum,
                learning_rate=args.lora_lr,
                logging_steps=args.lora_logging_steps,
                save_strategy=args.lora_save_strategy,
                save_total_limit=args.lora_save_total_limit,
            )

        prev_accuracy = accuracy

if __name__ == "__main__":
    main()
