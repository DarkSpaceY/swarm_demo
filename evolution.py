import argparse
import logging
import os
import json
import re
import time
import random
import concurrent.futures
import threading
from collections import Counter, deque
import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from config_loader import get_config

from core.agent import Agent
from core.env import Environment
from core.oracle import Oracle
from data.loader import load_math_problems
from train.converter import (
    convert_trajectory_to_sft, 
    save_sft_data, 
    convert_trajectories_to_preference, 
    save_jsonl_data,
    convert_judge_trajectories_to_preference
)
from train.finetuner import SwarmTrainer, resolve_local_files_only, ensure_qwen2_temp_qa

CONFIG = get_config()

logging.basicConfig(
    level=getattr(logging, str(CONFIG["logging"]["level"]).upper(), logging.DEBUG),
    format=CONFIG["logging"]["format"],
    force=bool(CONFIG["logging"]["force"]),
)
logging.getLogger().setLevel(getattr(logging, str(CONFIG["logging"]["level"]).upper(), logging.DEBUG))

MODEL = None
TOKENIZER = None
INFERENCE_LOCK = threading.Lock()
INFERENCE_ENGINE = None
VLLM_ENGINE = None
inference_model_fn = None
GEN_CONFIG = dict(CONFIG["generation"])

def _init_vllm(model_name, cache_dir, local_files_only):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_fast=False,
    )
    return None, tokenizer

def _init_transformers(model_name, cache_dir, local_files_only):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_fast=False,
    )
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    device = CONFIG["inference"]["cuda_device"] if torch.cuda.is_available() else CONFIG["inference"]["cpu_device"]
    model = model.to(device)
    return model, tokenizer

def _init_unsloth(model_name, max_seq_length, load_in_4bit, cache_dir, local_files_only):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = FastLanguageModel.for_inference(model)
    ensure_qwen2_temp_qa(model)
    return model, tokenizer

def _select_inference_mode(use_vllm):
    if use_vllm:
        return "vllm"
    disable_unsloth = os.environ.get(CONFIG["inference"]["disable_unsloth_env"]) == "1"
    return "transformers" if disable_unsloth else "unsloth"

def init_inference_model(model_name, max_seq_length, load_in_4bit, cache_dir, local_files_only, use_vllm):
    global MODEL, TOKENIZER
    resolved_local_only = resolve_local_files_only(model_name, cache_dir, local_files_only)
    mode = _select_inference_mode(use_vllm)
    if mode == "vllm":
        MODEL, TOKENIZER = _init_vllm(model_name, cache_dir, resolved_local_only)
    elif mode == "transformers":
        MODEL, TOKENIZER = _init_transformers(model_name, cache_dir, resolved_local_only)
    else:
        MODEL, TOKENIZER = _init_unsloth(model_name, max_seq_length, load_in_4bit, cache_dir, resolved_local_only)

def refresh_inference_from_trainer(trainer):
    global MODEL, TOKENIZER, INFERENCE_ENGINE, inference_model_fn
    if trainer is None:
        return
    trainer_model = getattr(trainer, "model", None)
    trainer_tokenizer = getattr(trainer, "tokenizer", None)
    if trainer_model is None or trainer_tokenizer is None:
        return
    TOKENIZER = trainer_tokenizer
    model = trainer_model
    try:
        model = FastLanguageModel.for_inference(model)
        ensure_qwen2_temp_qa(model)
    except (ImportError, RuntimeError, AttributeError, ValueError):
        model = trainer_model
    MODEL = model
    if INFERENCE_ENGINE is not None:
        INFERENCE_ENGINE.model = MODEL
        INFERENCE_ENGINE.tokenizer = TOKENIZER
        inference_model_fn = INFERENCE_ENGINE

def normalize_chat_history(history):
    if not isinstance(history, list):
        return []
    normalized = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not role:
            continue
        if role not in set(CONFIG["agent"]["roles"]):
            continue
        if content is None:
            content = CONFIG["common"]["empty_text"]
        elif not isinstance(content, str):
            content = str(content)
        normalized.append({"role": role, "content": content})
    return normalized

class InferenceEngine:
    def __init__(self, tokenizer, model, vllm_engine=None, use_vllm=False, vllm_enable_lora=False, lora_path_template=None, vllm_max_workers=4):
        self.tokenizer = tokenizer
        self.model = model
        self.vllm_engine = vllm_engine
        self.use_vllm = use_vllm
        self.vllm_enable_lora = vllm_enable_lora
        self.lora_path_template = lora_path_template
        self.vllm_max_workers = vllm_max_workers

    def __call__(self, agent_id, history):
        outputs = self.batch_generate({agent_id: history})
        return outputs.get(agent_id, CONFIG["generation"]["default_reply"])

    def batch_generate(self, inputs_by_agent):
        if not inputs_by_agent:
            return {}
        agent_ids = list(inputs_by_agent.keys())
        histories = [inputs_by_agent[aid] for aid in agent_ids]
        prompts = [self._build_prompt(h) for h in histories]
        if self.use_vllm and self.vllm_engine is not None:
            texts = self._vllm_generate_batch(agent_ids, prompts)
        else:
            texts = self._hf_generate_batch(prompts)
        return {aid: txt for aid, txt in zip(agent_ids, texts)}

    def _build_prompt(self, history):
        history = normalize_chat_history(history)
        if not history:
            history = [{"role": CONFIG["agent"]["role_user"], "content": CONFIG["generation"]["empty_prompt_fallback"]}]
        prompt = None
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = None
        if not isinstance(prompt, str) or not prompt:
            role_format = CONFIG["generation"]["fallback_role_format"]
            joiner = CONFIG["generation"]["fallback_joiner"]
            suffix = CONFIG["generation"]["fallback_suffix"]
            prompt = joiner.join([role_format.format(role=m["role"], content=m["content"]) for m in history]) + suffix
        return prompt

    def _hf_generate_batch(self, prompts):
        if self.tokenizer is None or self.model is None:
            return [CONFIG["generation"]["default_reply"]] * len(prompts)
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenizer_cfg = CONFIG["tokenizer"]
        inputs = self.tokenizer(prompts, return_tensors=tokenizer_cfg["return_tensors"], padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_ids_key = tokenizer_cfg["input_ids_key"]
        if input_ids_key not in inputs or inputs[input_ids_key].numel() == 0:
            return [CONFIG["generation"]["default_reply"]] * len(prompts)
        with INFERENCE_LOCK:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=GEN_CONFIG["max_new_tokens"],
                    do_sample=GEN_CONFIG["do_sample"],
                    temperature=GEN_CONFIG["temperature"],
                    top_p=GEN_CONFIG["top_p"],
                    top_k=GEN_CONFIG["top_k"],
                )
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        if sequences is None:
            return [CONFIG["generation"]["default_reply"]] * len(prompts)
        input_lens = None
        attention_key = tokenizer_cfg["attention_mask_key"]
        if attention_key in inputs:
            input_lens = inputs[attention_key].sum(dim=1).tolist()
        else:
            input_lens = [inputs[input_ids_key].shape[1]] * inputs[input_ids_key].shape[0]
        texts = []
        for i, seq in enumerate(sequences):
            gen_ids = seq[input_lens[i]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            if not text:
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
                if decoded.startswith(prompts[i]):
                    text = decoded[len(prompts[i]):].strip()
                else:
                    text = decoded
            texts.append(text if text else CONFIG["generation"]["default_reply"])
        return texts

    def _vllm_generate_batch(self, agent_ids, prompts):
        sampling_params = self._build_sampling_params()
        if not self.vllm_enable_lora:
            outputs = self.vllm_engine.generate(prompts, sampling_params)
            return [self._extract_vllm_text(out) for out in outputs]
        indexed = []
        for idx, (agent_id, prompt) in enumerate(zip(agent_ids, prompts)):
            lora_request = self._build_lora_request(agent_id)
            key = None
            if lora_request is not None:
                key = (lora_request.lora_name, lora_request.lora_path, lora_request.lora_int_id)
            indexed.append((idx, prompt, lora_request, key))
        groups = {}
        for idx, prompt, lora_request, key in indexed:
            groups.setdefault(key, []).append((idx, prompt, lora_request))
        results = [None] * len(prompts)
        min_workers = CONFIG["limits"]["min_workers"]
        max_workers = max(min_workers, int(self.vllm_max_workers)) if self.vllm_max_workers else min_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for key, items in groups.items():
                prompts_group = [item[1] for item in items]
                lora_request = items[0][2] if items and items[0][2] is not None else None
                futures[executor.submit(self.vllm_engine.generate, prompts_group, sampling_params, lora_request=lora_request)] = items
            for future in concurrent.futures.as_completed(futures):
                items = futures[future]
                outputs = future.result()
                for out, item in zip(outputs, items):
                    idx = item[0]
                    results[idx] = self._extract_vllm_text(out)
        return [text if text is not None else CONFIG["generation"]["default_reply"] for text in results]

    def _build_sampling_params(self):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=GEN_CONFIG["temperature"],
            top_p=GEN_CONFIG["top_p"],
            top_k=GEN_CONFIG["top_k"],
            max_tokens=GEN_CONFIG["max_new_tokens"],
            n=CONFIG["vllm"]["sampling_n"],
        )

    def _build_lora_request(self, agent_id):
        if not self.lora_path_template:
            return None
        lora_path = self.lora_path_template.format(agent_id=agent_id)
        if not os.path.exists(lora_path):
            return None
        try:
            agent_index = int(str(agent_id).split("_")[-1])
        except Exception:
            agent_index = abs(hash(agent_id)) % CONFIG["vllm"]["agent_index_mod"] + CONFIG["vllm"]["agent_index_offset"]
        from vllm.lora.request import LoRARequest
        return LoRARequest(str(agent_id), agent_index + 1, lora_path)

    def _extract_vllm_text(self, output):
        try:
            if output is None or not output.outputs:
                return CONFIG["generation"]["default_reply"]
            text = output.outputs[0].text
            return text.strip() if isinstance(text, str) else CONFIG["generation"]["default_reply"]
        except Exception:
            return CONFIG["generation"]["default_reply"]

def resolve_lora_path_template(template, run_dir):
    if template is None:
        return None
    marker = CONFIG["templates"]["lora_agent_marker"]
    placeholder = CONFIG["templates"]["lora_agent_placeholder"]
    run_dir_placeholder = CONFIG["templates"]["lora_run_dir_placeholder"]
    template = template.replace(placeholder, marker)
    try:
        template = template.replace(run_dir_placeholder, "{run_dir}").format(run_dir=run_dir)
    except Exception:
        return template.replace(marker, placeholder)
    return template.replace(marker, placeholder)

def init_inference_engine(
    model_name,
    use_vllm,
    vllm_prefix_caching,
    vllm_enable_lora,
    vllm_max_model_len,
    vllm_gpu_memory_utilization,
    vllm_max_loras,
    vllm_max_lora_rank,
    lora_path_template,
    vllm_max_workers,
):
    global INFERENCE_ENGINE, VLLM_ENGINE, inference_model_fn
    if use_vllm:
        from vllm import LLM
        kwargs = {
            "model": model_name,
            "enable_prefix_caching": vllm_prefix_caching,
            "enable_lora": vllm_enable_lora,
            "max_model_len": vllm_max_model_len,
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
            "max_loras": vllm_max_loras,
            "max_lora_rank": vllm_max_lora_rank,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        VLLM_ENGINE = LLM(**kwargs)
        INFERENCE_ENGINE = InferenceEngine(
            TOKENIZER,
            MODEL,
            vllm_engine=VLLM_ENGINE,
            use_vllm=True,
            vllm_enable_lora=vllm_enable_lora,
            lora_path_template=lora_path_template,
            vllm_max_workers=vllm_max_workers,
        )
    else:
        INFERENCE_ENGINE = InferenceEngine(TOKENIZER, MODEL)
    inference_model_fn = INFERENCE_ENGINE
# --- Unified System Prompt (Homogeneous) ---

def _build_agent_list():
    count = CONFIG["swarm"]["agent_count"]
    prefix = CONFIG["swarm"]["agent_prefix"]
    return "、".join([f"{prefix}{i}" for i in range(count)])

def get_system_prompt(agent_id):
    template = CONFIG["swarm"]["system_prompt_template"]
    return template.format(agent_id=agent_id, agent_list=_build_agent_list())

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
    
    agent_count = CONFIG["swarm"]["agent_count"]
    prefix = CONFIG["swarm"]["agent_prefix"]
    agents = [Agent(f"{prefix}{i}", get_system_prompt(f"{prefix}{i}")) for i in range(agent_count)]
    env = Environment(agents, problem_text, max_rounds=max_rounds)
    
    result = env.run(inference_model_fn)
    result["trial_id"] = trial_id
    return result

# --- Judge Logic ---

def format_judge_task(problem_text, candidate_trajectories, gt_answer):
    candidates_text = ""
    answer_placeholder = CONFIG["judge"]["answer_placeholder"]
    max_chars = CONFIG["judge"]["traj_snippet_max_chars"]
    candidate_template = CONFIG["judge"]["candidate_template"]
    for idx, traj in enumerate(candidate_trajectories):
        final_ans = traj.get("final_answer", answer_placeholder)
        traj_content = ""
        for step in traj["trajectory"]:
            if step['agent'].startswith(CONFIG["swarm"]["agent_prefix"]):
                traj_content += f"{step['agent']}: {step['output'][:max_chars]}...\n"
        candidates_text += candidate_template.format(idx=idx, final_ans=final_ans, traj_content=traj_content)

    task_template = CONFIG["judge"]["task_template"]
    judge_task = task_template.format(problem=problem_text, gt_answer=gt_answer, candidates=candidates_text)
    return judge_task

def run_judge_swarm_trial(judge_task, trial_id, max_rounds):
    """
    运行 Judge Swarm 进行评判。
    Input: Multiple Choice Task (constructed from trajectories)
    Output: [FIN] Index
    """
    logging.info(f"Starting Judge Swarm Trial {trial_id}...")
    
    # Judge Swarm 与 Solver Swarm 结构完全一致
    agent_count = CONFIG["swarm"]["agent_count"]
    prefix = CONFIG["swarm"]["agent_prefix"]
    agents = [Agent(f"{prefix}{i}", get_system_prompt(f"{prefix}{i}")) for i in range(agent_count)]
    env = Environment(agents, judge_task, max_rounds=max_rounds)
    
    result = env.run(inference_model_fn)
    logging.debug(f"Judge trial {trial_id} final answer: {result.get('final_answer')}")
    logging.debug(f"Judge trial {trial_id} trajectory len: {len(result.get('trajectory', []))}")
    return result

def _parse_ranking_with_llm(result_text, num_candidates):
    if not result_text or not callable(inference_model_fn):
        return None
    template = CONFIG["parser"]["judge_parser_prompt"]
    prompt = template.format(num_candidates=num_candidates, result_text=result_text)
    output_text = inference_model_fn(CONFIG["inference"]["judge_parser_agent_id"], [{"role": "user", "content": prompt}])
    return extract_ranking(output_text, num_candidates)

def extract_ranking(result_text, num_candidates):
    if not result_text:
        logging.debug("Ranking parse skipped: empty result_text")
        return None
    text = str(result_text)
    match = re.search(CONFIG["agent"]["fin_pattern"], text)
    if match:
        text = match.group(1).strip()
    numbers = re.findall(r'-?\d+', text)
    logging.debug(f"Ranking parse raw: {text} numbers: {numbers}")
    ranking = []
    for n in numbers:
        idx = int(n)
        if 0 <= idx < num_candidates and idx not in ranking:
            ranking.append(idx)
    if len(ranking) == num_candidates:
        return ranking
    logging.debug(f"Ranking parse failed: {ranking} expected {num_candidates}")
    return None

def _parse_dreamer_variants(text, num_variants, original_problem):
    if not text:
        return []
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(CONFIG["dreamer"]["line_prefix_pattern"], '', line).strip()
        if line and line != original_problem and line not in lines:
            lines.append(line)
        if len(lines) >= num_variants:
            break
    return lines

def dreamer_generate_variants(problem_text, gt_answer, num_variants):
    if num_variants <= 0:
        return []
    template = CONFIG["dreamer"]["prompt_template"]
    prompt = template.format(num_variants=num_variants, problem_text=problem_text, gt_answer=gt_answer)
    history = [{"role": "user", "content": prompt}]
    output_text = inference_model_fn(CONFIG["inference"]["dreamer_agent_id"], history)
    return _parse_dreamer_variants(output_text, num_variants, problem_text)

# --- Main Evolution Loop ---

def run_evolution_batch(
    batch_id,
    num_problems=None,
    num_solver_samples=None,
    num_judge_samples=None,
    solver_max_workers=None,
    judge_max_workers=None,
    max_rounds=None,
    dataset_name=None,
    dataset_split=None,
    dataset_streaming=None,
    dataset_seed=None,
    dataset_shuffle_buffer=None,
    preference_max_pairs_per_agent=None,
    enable_preference=None,
    enable_dreamer=None,
    dreamer_num_variants=None,
    dreamer_max_queue=None,
):
    """
    运行一个进化批次 (Batch)。
    返回: (accuracy, solver_data, judge_data, correct_count, total_count)
    """
    run_defaults = CONFIG["run_defaults"]
    dataset_defaults = CONFIG["dataset"]
    dreamer_defaults = CONFIG["dreamer"]
    if num_problems is None:
        num_problems = run_defaults["num_problems"]
    if num_solver_samples is None:
        num_solver_samples = run_defaults["num_solver_samples"]
    if num_judge_samples is None:
        num_judge_samples = run_defaults["num_judge_samples"]
    if solver_max_workers is None:
        solver_max_workers = run_defaults["solver_max_workers"]
    if judge_max_workers is None:
        judge_max_workers = run_defaults["judge_max_workers"]
    if max_rounds is None:
        max_rounds = run_defaults["max_rounds"]
    if dataset_name is None:
        dataset_name = dataset_defaults["name_default"]
    if dataset_split is None:
        dataset_split = dataset_defaults["split_default"]
    if dataset_streaming is None:
        dataset_streaming = dataset_defaults["streaming_default"]
    if dataset_seed is None:
        dataset_seed = dataset_defaults["seed_default"]
    if dataset_shuffle_buffer is None:
        dataset_shuffle_buffer = dataset_defaults["shuffle_buffer_default"]
    if preference_max_pairs_per_agent is None:
        preference_max_pairs_per_agent = run_defaults["preference_max_pairs_per_agent"]
    if enable_preference is None:
        enable_preference = run_defaults["train_preference"]
    if enable_dreamer is None:
        enable_dreamer = run_defaults["enable_dreamer"]
    if dreamer_num_variants is None:
        dreamer_num_variants = dreamer_defaults["num_variants_default"]
    if dreamer_max_queue is None:
        dreamer_max_queue = dreamer_defaults["max_queue_default"]
    
    # 记录本批次 Judge 的共识准确率
    batch_judge_correct_count = 0
    batch_judge_total_count = 0

    logging.info(f"=== Starting Batch {batch_id} ===")
    
    problems = list(load_math_problems(
        dataset_name=dataset_name,
        split=dataset_split,
        num_samples=num_problems,
        streaming=dataset_streaming,
        seed=dataset_seed,
        shuffle_buffer=dataset_shuffle_buffer,
    ))
    if not problems:
        logging.error("No problems loaded. Check dataset access and configuration.")
        return 0.0, [], [], 0, 0, [], []
    oracle = Oracle(inference_model_fn)
    
    batch_correct_count = 0
    batch_solver_data = [] # 存储本轮产生的优质 Solver 轨迹
    batch_judge_data = []  # 存储本轮产生的 Judge 轨迹 (无论对错，后续筛选)
    batch_pref_data = []
    dreamer_records = []
    
    prob_idx = 0
    while prob_idx < len(problems):
        data_keys = CONFIG["data_keys"]
        prob_data = problems[prob_idx]
        problem_text = prob_data[data_keys["problem"]]
        gt_answer = prob_data[data_keys["answer"]]
        logging.info(f"Problem {prob_idx}: {problem_text[:50]}...")
        
        # 1. 并行运行 Solver Swarms (生成 4 条轨迹)
        candidate_trajectories = []
        cpu_limit = os.cpu_count() or solver_max_workers
        min_samples = CONFIG["limits"]["min_samples_fallback"]
        solver_workers = min(num_solver_samples, solver_max_workers, cpu_limit) if num_solver_samples > 0 else min_samples
        with concurrent.futures.ThreadPoolExecutor(max_workers=solver_workers) as executor:
            futures = {executor.submit(run_solver_trial, problem_text, i, max_rounds): i for i in range(num_solver_samples)}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["status"] == CONFIG["inference"]["status_success"]:
                    candidate_trajectories.append(res)
        
        if not candidate_trajectories:
            logging.warning("No valid trajectories generated.")
            continue
            
        # 2. 构造 Judge 任务
        judge_task_text = format_judge_task(problem_text, candidate_trajectories, gt_answer)
        
        # 3. 并行运行 Judge Swarms (3 个分身进行投票)
        votes = []
        rankings = []
        judge_trajectories = []
        
        min_samples = CONFIG["limits"]["min_samples_fallback"]
        judge_workers = min(num_judge_samples, judge_max_workers, cpu_limit) if num_judge_samples > 0 else min_samples
        with concurrent.futures.ThreadPoolExecutor(max_workers=judge_workers) as executor:
            futures = {executor.submit(run_judge_swarm_trial, judge_task_text, i, max_rounds): i for i in range(num_judge_samples)}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["status"] == CONFIG["inference"]["status_success"]:
                    ranking = extract_ranking(res["final_answer"], len(candidate_trajectories))
                    if ranking is None:
                        ranking = _parse_ranking_with_llm(res["final_answer"], len(candidate_trajectories))
                    if ranking is not None:
                        votes.append(ranking[0])
                        rankings.append(ranking)
                        judge_trajectories.append({
                            CONFIG["data_keys"]["task"]: judge_task_text,
                            CONFIG["data_keys"]["trajectory"]: res["trajectory"],
                            CONFIG["data_keys"]["final_answer"]: ",".join(str(i) for i in ranking),
                            CONFIG["data_keys"]["vote_val"]: ranking[0]
                        })

        # 4. 达成共识 (Borda Count)
        ambiguous = False
        if not rankings:
            logging.warning("No valid votes from Judge Swarm.")
            consensus_idx = CONFIG["judge"]["consensus_invalid"]
        else:
            num_candidates = len(candidate_trajectories)
            scores = Counter()
            for r in rankings:
                for rank, candidate_idx in enumerate(r):
                    weight = num_candidates - 1 - rank
                    scores[candidate_idx] += weight
            
            most_common = scores.most_common()
            max_score = most_common[0][1]
            top_candidates = [cand for cand, score in most_common if score == max_score]
            if len(top_candidates) > 1:
                ambiguous = True
            elif len(most_common) > 1 and (max_score - most_common[1][1]) <= CONFIG["judge"]["tie_gap_threshold"]:
                ambiguous = True
            
            if len(top_candidates) > 1:
                consensus_idx = random.choice(top_candidates)
                logging.info(f"Judge Consensus (Borda Tie-Break): {consensus_idx} from {top_candidates} (Score: {max_score})")
            else:
                consensus_idx = top_candidates[0]
                logging.info(f"Judge Consensus (Borda): {consensus_idx} (Scores: {dict(scores)})")
        
        # 5. Oracle 验证与数据收集
        is_dream = prob_data.get(CONFIG["data_keys"]["source"]) == CONFIG["paths"]["dreamer_dir"]
        candidate_correctness = []
        for cand in candidate_trajectories:
            is_cand_correct = oracle.verify(cand["final_answer"], gt_answer)
            candidate_correctness.append(is_cand_correct)

        if 0 <= consensus_idx < len(candidate_trajectories):
            is_correct = candidate_correctness[consensus_idx]
            
            # 更新统计
            batch_judge_total_count += 1
            if is_correct:
                batch_judge_correct_count += 1
                batch_correct_count += 1

            # --- SFT 数据收集 (经验轨迹) ---
            # 规则：只需要 Judge 达成共识即可，依靠主循环的信任门控保护
            
            # Solver SFT
            selected_traj = candidate_trajectories[consensus_idx]
            batch_solver_data.extend(convert_trajectory_to_sft(selected_traj["trajectory"]))
            
            # Judge SFT (投给共识的 Judge 学习自己的轨迹)
            for j_traj in judge_trajectories:
                if j_traj[CONFIG["data_keys"]["vote_val"]] == consensus_idx:
                    batch_judge_data.extend(convert_trajectory_to_sft(j_traj[CONFIG["data_keys"]["trajectory"]]))

            # --- 偏好数据收集 ---
            # Solver 偏好
            if enable_preference:
                pref_records = convert_trajectories_to_preference(
                    candidate_trajectories,
                    consensus_idx,
                    max_pairs_per_agent=preference_max_pairs_per_agent,
                    is_correct=is_correct
                )
                if pref_records:
                    for rec in pref_records:
                        rec["is_dream"] = is_dream
                    batch_pref_data.extend(pref_records)

            # Judge 偏好
            judge_pref_records = convert_judge_trajectories_to_preference(
                judge_trajectories,
                consensus_idx,
                candidate_correctness,
                max_pairs_per_agent=preference_max_pairs_per_agent,
                ambiguous=ambiguous
            )
            if judge_pref_records:
                batch_pref_data.extend(judge_pref_records)

        logging.info(f"Batch {batch_id} Progress: {prob_idx + 1}/{len(problems)} problems done. ")
        
        # 6. Dreamer 机制
        if enable_dreamer and ambiguous and len(dreamer_records) < dreamer_max_queue:
            variants = dreamer_generate_variants(problem_text, gt_answer, dreamer_num_variants)
            for variant in variants:
                if len(dreamer_records) >= dreamer_max_queue:
                    break
                data_keys = CONFIG["data_keys"]
                dream_entry = {
                    data_keys["problem"]: variant,
                    data_keys["answer"]: gt_answer,
                    data_keys["source"]: CONFIG["paths"]["dreamer_dir"],
                    data_keys["origin"]: problem_text,
                    "is_dream": True
                }
                dreamer_records.append(dream_entry)
                problems.append(dream_entry)
         
        prob_idx += 1

    total_count = len(problems)
    accuracy = batch_correct_count / total_count if total_count else 0
    return accuracy, batch_solver_data, batch_judge_data, batch_correct_count, total_count, batch_pref_data, dreamer_records, batch_judge_correct_count, batch_judge_total_count

def build_arg_parser():
    parser = argparse.ArgumentParser()
    run_defaults = CONFIG["run_defaults"]
    dataset_defaults = CONFIG["dataset"]
    parser.add_argument("--run-name", type=str, default=run_defaults["run_name"])
    parser.add_argument("--output-dir", type=str, default=run_defaults["output_dir"])
    parser.add_argument("--model-name", type=str, default=run_defaults["model_name"])
    parser.add_argument("--model-cache-dir", type=str, default=None)
    parser.add_argument("--max-seq-length", type=int, default=run_defaults["max_seq_length"])
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=run_defaults["load_in_4bit"])
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=run_defaults["local_files_only"])
    parser.add_argument("--num-batches", type=int, default=run_defaults["num_batches"])
    parser.add_argument("--num-problems", type=int, default=run_defaults["num_problems"])
    parser.add_argument("--num-solver-samples", type=int, default=run_defaults["num_solver_samples"])
    parser.add_argument("--num-judge-samples", type=int, default=run_defaults["num_judge_samples"])
    parser.add_argument("--solver-max-workers", type=int, default=run_defaults["solver_max_workers"])
    parser.add_argument("--judge-max-workers", type=int, default=run_defaults["judge_max_workers"])
    parser.add_argument("--max-rounds", type=int, default=run_defaults["max_rounds"])
    parser.add_argument("--seed", type=int, default=run_defaults["seed"])
    parser.add_argument("--dataset-name", type=str, default=dataset_defaults["name_default"])
    parser.add_argument("--dataset-split", type=str, default=dataset_defaults["split_default"])
    parser.add_argument("--dataset-streaming", action=argparse.BooleanOptionalAction, default=run_defaults["dataset_streaming"])
    parser.add_argument("--dataset-shuffle-buffer", type=int, default=run_defaults["dataset_shuffle_buffer"])
    parser.add_argument("--gen-max-new-tokens", type=int, default=run_defaults["gen_max_new_tokens"])
    parser.add_argument("--gen-do-sample", action=argparse.BooleanOptionalAction, default=run_defaults["gen_do_sample"])
    parser.add_argument("--gen-temperature", type=float, default=run_defaults["gen_temperature"])
    parser.add_argument("--gen-top-p", type=float, default=run_defaults["gen_top_p"])
    parser.add_argument("--gen-top-k", type=int, default=run_defaults["gen_top_k"])
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=run_defaults["use_vllm"])
    parser.add_argument("--vllm-prefix-caching", action=argparse.BooleanOptionalAction, default=run_defaults["vllm_prefix_caching"])
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=run_defaults["vllm_gpu_memory_utilization"])
    parser.add_argument("--vllm-enable-lora", action=argparse.BooleanOptionalAction, default=run_defaults["vllm_enable_lora"])
    parser.add_argument("--vllm-max-loras", type=int, default=None)
    parser.add_argument("--vllm-max-lora-rank", type=int, default=None)
    parser.add_argument("--vllm-lora-path-template", type=str, default=run_defaults["vllm_lora_path_template"])
    parser.add_argument("--vllm-max-workers", type=int, default=run_defaults["vllm_max_workers"])
    parser.add_argument("--train-lora", action="store_true")
    parser.add_argument("--lora-epochs", type=int, default=run_defaults["lora_epochs"])
    parser.add_argument("--lora-batch-size", type=int, default=run_defaults["lora_batch_size"])
    parser.add_argument("--lora-grad-accum", type=int, default=run_defaults["lora_grad_accum"])
    parser.add_argument("--lora-lr", type=float, default=run_defaults["lora_lr"])
    parser.add_argument("--lora-logging-steps", type=int, default=run_defaults["lora_logging_steps"])
    parser.add_argument("--lora-save-strategy", type=str, default=run_defaults["lora_save_strategy"])
    parser.add_argument("--lora-save-total-limit", type=int, default=run_defaults["lora_save_total_limit"])
    parser.add_argument("--lora-judge-oversample", type=int, default=run_defaults["lora_judge_oversample"])
    parser.add_argument("--judge-update-interval", type=int, default=run_defaults["judge_update_interval"])
    parser.add_argument("--accuracy-window-batches", type=int, default=run_defaults["accuracy_window_batches"])
    parser.add_argument("--train-judge", action=argparse.BooleanOptionalAction, default=run_defaults["train_judge"])
    parser.add_argument("--train-preference", action=argparse.BooleanOptionalAction, default=run_defaults["train_preference"])
    parser.add_argument("--preference-algorithm", type=str, default=run_defaults["preference_algorithm"])
    parser.add_argument("--preference-beta", type=float, default=run_defaults["preference_beta"])
    parser.add_argument("--preference-sft-weight", type=float, default=run_defaults["preference_sft_weight"])
    parser.add_argument("--preference-epochs", type=int, default=run_defaults["preference_epochs"])
    parser.add_argument("--preference-batch-size", type=int, default=run_defaults["preference_batch_size"])
    parser.add_argument("--preference-grad-accum", type=int, default=run_defaults["preference_grad_accum"])
    parser.add_argument("--preference-lr", type=float, default=run_defaults["preference_lr"])
    parser.add_argument("--preference-logging-steps", type=int, default=run_defaults["preference_logging_steps"])
    parser.add_argument("--preference-save-strategy", type=str, default=run_defaults["preference_save_strategy"])
    parser.add_argument("--preference-save-total-limit", type=int, default=run_defaults["preference_save_total_limit"])
    parser.add_argument("--preference-max-pairs-per-agent", type=int, default=run_defaults["preference_max_pairs_per_agent"])
    parser.add_argument("--enable-dreamer", action=argparse.BooleanOptionalAction, default=run_defaults["enable_dreamer"])
    parser.add_argument("--dreamer-num-variants", type=int, default=run_defaults["dreamer_num_variants"])
    parser.add_argument("--agent-count", type=int, default=CONFIG["swarm"]["agent_count"])
    parser.add_argument("--judge-trust-threshold", type=float, default=0.6)
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    CONFIG["swarm"]["agent_count"] = args.agent_count
    GEN_CONFIG["max_new_tokens"] = args.gen_max_new_tokens
    GEN_CONFIG["do_sample"] = args.gen_do_sample
    GEN_CONFIG["temperature"] = args.gen_temperature
    GEN_CONFIG["top_p"] = args.gen_top_p
    GEN_CONFIG["top_k"] = args.gen_top_k

    paths_cfg = CONFIG["paths"]
    run_dir = os.path.join(args.output_dir, args.run_name)
    data_dir = os.path.join(run_dir, paths_cfg["data_dir"])
    solver_dir = os.path.join(data_dir, paths_cfg["solver_dir"])
    judge_dir = os.path.join(data_dir, paths_cfg["judge_dir"])
    pref_dir = os.path.join(data_dir, paths_cfg["preference_dir"])
    dreamer_dir = os.path.join(run_dir, paths_cfg["dreamer_dir"])
    metrics_path = os.path.join(run_dir, paths_cfg["metrics_file"])
    os.makedirs(solver_dir, exist_ok=True)
    os.makedirs(judge_dir, exist_ok=True)
    os.makedirs(pref_dir, exist_ok=True)
    os.makedirs(dreamer_dir, exist_ok=True)

    init_inference_model(
        args.model_name,
        args.max_seq_length,
        args.load_in_4bit,
        args.model_cache_dir,
        args.local_files_only,
        args.use_vllm,
    )

    vllm_max_model_len = args.vllm_max_model_len or args.max_seq_length
    vllm_lora_path_template = resolve_lora_path_template(args.vllm_lora_path_template, run_dir)
    init_inference_engine(
        model_name=args.model_name,
        use_vllm=args.use_vllm,
        vllm_prefix_caching=args.vllm_prefix_caching,
        vllm_enable_lora=args.vllm_enable_lora,
        vllm_max_model_len=vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_loras=args.vllm_max_loras,
        vllm_max_lora_rank=args.vllm_max_lora_rank,
        lora_path_template=vllm_lora_path_template,
        vllm_max_workers=args.vllm_max_workers,
    )

    trainer = None
    if args.train_lora:
        trainer = SwarmTrainer(
            model_name=args.model_name,
            output_dir=os.path.join(run_dir, paths_cfg["checkpoints_dir"]),
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            cache_dir=args.model_cache_dir,
            local_files_only=args.local_files_only,
        )

    prev_accuracy = 0.0
    judge_prev_accuracy = 0.0
    accuracy_window = max(CONFIG["limits"]["min_accuracy_window"], args.accuracy_window_batches)
    window_stats = deque(maxlen=accuracy_window)
    judge_buffer = []
    judge_buffer_start = None

    for batch_id in range(args.num_batches):
        accuracy, solver_data, judge_data, correct_count, total_count, pref_data, dreamer_records, judge_correct_count, judge_total_count = run_evolution_batch(
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
            preference_max_pairs_per_agent=args.preference_max_pairs_per_agent,
            enable_preference=args.train_preference,
            enable_dreamer=args.enable_dreamer,
            dreamer_num_variants=args.dreamer_num_variants,
            dreamer_max_queue=args.dreamer_max_queue,
        )

        # 计算 Judge 信任度
        judge_accuracy = judge_correct_count / judge_total_count if judge_total_count else 0
        is_judge_trusted = judge_accuracy >= args.judge_trust_threshold
        logging.info(f"Batch {batch_id} Judge Accuracy: {judge_accuracy:.2%} (Threshold: {args.judge_trust_threshold:.2%})")
        if not is_judge_trusted:
            logging.warning("Judge trust low. Solver update and Dreamer variants will be skipped for this batch.")

        window_stats.append((correct_count, total_count))
        window_correct = sum(item[0] for item in window_stats)
        window_total = sum(item[1] for item in window_stats)
        window_accuracy = window_correct / window_total if window_total else 0.0
        logging.info(f"Batch {batch_id} Result: Accuracy = {window_accuracy:.2%} (Prev: {prev_accuracy:.2%})")

        metric_record = {
            "batch_id": batch_id,
            "accuracy": window_accuracy,
            "batch_accuracy": accuracy,
            "judge_accuracy": judge_accuracy,
            "is_judge_trusted": is_judge_trusted,
            "prev_accuracy": prev_accuracy,
            "num_problems": args.num_problems,
            "num_solver_samples": args.num_solver_samples,
            "num_judge_samples": args.num_judge_samples,
            "timestamp": time.time(),
        }
        os.makedirs(run_dir, exist_ok=True)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_record, ensure_ascii=False) + "\n")

        # 1. 处理 Solver SFT 数据 (仅信任时)
        solver_path = None
        if solver_data and is_judge_trusted:
            solver_path = os.path.join(solver_dir, paths_cfg["solver_file_template"].format(batch_id=batch_id))
            save_sft_data(solver_data, solver_path)
            logging.info(f"Saved {len(solver_data)} solver samples to {solver_path}")

        # 2. 处理 Judge SFT 数据 (始终处理，但 run_evolution_batch 已过滤正确轨迹)
        if judge_data:
            judge_buffer.extend(judge_data)
            if judge_buffer_start is None:
                judge_buffer_start = batch_id

        # 3. 处理偏好数据 (Solver 和 Judge)
        if pref_data:
            pref_path = os.path.join(pref_dir, paths_cfg["pref_file_template"].format(batch_id=batch_id))
            save_jsonl_data(pref_data, pref_path)
            
            # 分离 Solver 和 Judge 的偏好数据
            # Judge 偏好数据的特征是带有 is_ambiguous 字段
            solver_pref = [r for r in pref_data if "is_ambiguous" not in r]
            judge_pref = [r for r in pref_data if "is_ambiguous" in r]
            
            if trainer and args.train_preference:
                # 训练 Solver (仅信任时)
                if solver_pref and is_judge_trusted:
                    db_pref = [r for r in solver_pref if not r.get("is_dream", False)]
                    dream_pref = [r for r in solver_pref if r.get("is_dream", False)]
                    
                    if db_pref:
                        trainer.train_preference_agent(CONFIG["agents"]["solver_id"], pref_path, algorithm="dpo")
                    if dream_pref:
                        trainer.train_preference_agent(CONFIG["agents"]["solver_id"], pref_path, algorithm="rpo")

                # 训练 Judge (始终训练)
                if judge_pref:
                    # 区分确定状态 (DPO) 和 模糊状态 (RPO)
                    certain_pref = [r for r in judge_pref if not r.get("is_ambiguous", False)]
                    ambiguous_pref = [r for r in judge_pref if r.get("is_ambiguous", False)]
                    
                    if certain_pref:
                        certain_pref_path = pref_path.replace(".jsonl", "_judge_certain.jsonl")
                        save_jsonl_data(certain_pref, certain_pref_path)
                        trainer.train_preference_agent(
                            CONFIG["agents"]["judge_id"], 
                            certain_pref_path, 
                            algorithm="dpo"
                        )
                    if ambiguous_pref:
                        # Ambiguous 状态触发的 Judge 训练强制使用 RPO
                        ambiguous_pref_path = pref_path.replace(".jsonl", "_judge_ambiguous.jsonl")
                        save_jsonl_data(ambiguous_pref, ambiguous_pref_path)
                        trainer.train_preference_agent(
                            CONFIG["agents"]["judge_id"], 
                            ambiguous_pref_path, 
                            algorithm="rpo"
                        )

                refresh_inference_from_trainer(trainer)

        # 4. 处理 Dreamer 记录 (仅信任时)
        if dreamer_records and is_judge_trusted:
            dreamer_path = os.path.join(dreamer_dir, paths_cfg["dreamer_file_template"].format(batch_id=batch_id))
            save_jsonl_data(dreamer_records, dreamer_path)
            logging.info(f"Saved {len(dreamer_records)} dreamer records to {dreamer_path}")

        agents_cfg = CONFIG["agents"]
        data_keys = CONFIG["data_keys"]
        if trainer and solver_path:
            trainer.train_agent(
                agents_cfg["solver_id"],
                solver_path,
                epochs=args.lora_epochs,
                batch_size=args.lora_batch_size,
                gradient_accumulation_steps=args.lora_grad_accum,
                learning_rate=args.lora_lr,
                logging_steps=args.lora_logging_steps,
                save_strategy=args.lora_save_strategy,
                save_total_limit=args.lora_save_total_limit,
            )
            refresh_inference_from_trainer(trainer)
            logging.info("Inference model refreshed with latest Solver LoRA.")

        if trainer and args.train_preference and pref_path:
            pref_agent_map = {}
            for item in pref_data:
                agent_id = item.get(data_keys["agent_id"]) or agents_cfg["preference_fallback_id"]
                pref_agent_map.setdefault(agent_id, []).append(item)
            for agent_id, records in pref_agent_map.items():
                agent_path = os.path.join(pref_dir, paths_cfg["pref_agent_file_template"].format(batch_id=batch_id, agent_id=agent_id))
                save_jsonl_data(records, agent_path)
                trainer.train_preference_agent(
                    agent_id,
                    agent_path,
                    algorithm=args.preference_algorithm,
                    beta=args.preference_beta,
                    sft_weight=args.preference_sft_weight,
                    epochs=args.preference_epochs,
                    batch_size=args.preference_batch_size,
                    gradient_accumulation_steps=args.preference_grad_accum,
                    learning_rate=args.preference_lr,
                    logging_steps=args.preference_logging_steps,
                    save_strategy=args.preference_save_strategy,
                    save_total_limit=args.preference_save_total_limit,
                )
            refresh_inference_from_trainer(trainer)
            logging.info("Inference model refreshed with latest preference training.")

        should_update_judge = args.train_judge and ((batch_id + 1) % max(CONFIG["limits"]["min_accuracy_window"], args.judge_update_interval) == 0)
        if trainer and should_update_judge:
            if window_accuracy > judge_prev_accuracy and judge_buffer:
                sample_size = min(len(judge_buffer), len(solver_data)) if solver_data else len(judge_buffer)
                selected_judge_data = random.sample(judge_buffer, sample_size) if sample_size > 0 else judge_buffer
                judge_path = os.path.join(judge_dir, paths_cfg["judge_file_template"].format(start=judge_buffer_start, end=batch_id))
                save_sft_data(selected_judge_data, judge_path)
                logging.info(f"Accuracy Improved! Saved {len(selected_judge_data)} judge samples to {judge_path}")
                trainer.train_agent(
                    agents_cfg["judge_id"],
                    judge_path,
                    epochs=args.lora_epochs,
                    batch_size=args.lora_batch_size,
                    gradient_accumulation_steps=args.lora_grad_accum,
                    learning_rate=args.lora_lr,
                    logging_steps=args.lora_logging_steps,
                    save_strategy=args.lora_save_strategy,
                    save_total_limit=args.lora_save_total_limit,
                    judge_oversample=args.lora_judge_oversample,
                )
                refresh_inference_from_trainer(trainer)
                logging.info("Inference model refreshed with latest Judge LoRA.")
            else:
                logging.info("Accuracy did not improve. Discarding Judge data.")
            judge_prev_accuracy = window_accuracy
            judge_buffer = []
            judge_buffer_start = None

        prev_accuracy = window_accuracy

if __name__ == "__main__":
    main()
