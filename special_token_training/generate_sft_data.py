import json
import random
import os

# 默认配置 (对齐 config.yaml)
DEFAULT_CONFIG = {
    "swarm": {
        "agent_prefix": "Agent_",
        "agent_count": 5
    },
    "agent": {
        "role_system": "system",
        "role_user": "user",
        "role_assistant": "assistant",
        "task_prefix": "任务目标: {problem}",
        "inbox_prefix": "From {sender_id}: {content}",
        "inbox_message_prefix": "新收到消息:\n{stimuli}"
    },
    "judge": {
        "candidate_template": "选项 {idx} (Answer: {final_ans}):\n{traj_content}\n",
        "task_template": "原始题目：\n{problem}\n\n标准答案：\n{gt_answer}\n\n候选选项：\n{candidates}\n请对候选选项进行从优到劣的排序，并给出你认可的顺序。"
    }
}

CONFIG = DEFAULT_CONFIG
try:
    import sys
    sys.path.append(os.getcwd())
    try:
        from config_loader import get_config
        real_config = get_config()
        if real_config:
            # 深度合并确保所有字段都存在
            for section, items in DEFAULT_CONFIG.items():
                if section not in real_config:
                    real_config[section] = items
                else:
                    for k, v in items.items():
                        if k not in real_config[section]:
                            real_config[section][k] = v
            CONFIG = real_config
    except Exception:
        pass
except Exception:
    pass

# 定义 Special Tokens
TOKENS = {
    "send": "<|send|>",
    "to": "<|to|>",
    "to_all": "<|to_all|>",
    "msg": "<|msg|>",
    "fin": "<|fin|>"
}

PROBLEMS = [
    {"q": "计算 12 + 24 * 3", "a": "84"},
    {"q": "2x + 10 = 20，求 x", "a": "5"},
    {"q": "一个正方形边长为 5，求面积", "a": "25"},
    {"q": "找出 1 到 20 之间的质数", "a": "2, 3, 5, 7, 11, 13, 17, 19"},
    {"q": "5! 等于多少？", "a": "120"},
    {"q": "计算 (15 - 3) / 4", "a": "3"}
]

# 极大增强信息密度和逻辑深度的内容池
THOUGHTS = [
    "当前任务涉及多步逻辑推导，我需要首先对整体任务进行解构。第一步是识别关键约束条件，并根据公式 {formula} 进行初步预估。考虑到计算过程中可能出现的舍入误差，我会采用分步校验的方法。",
    "基于已知的前置计算结果，我发现当前的数据链路中存在潜在的逻辑断层。我将重新审视运算优先级，并尝试从逆向推导的角度验证中间值的准确性，以确保最终输出的严谨性。",
    "系统内其他成员提供的局部解虽然在单一维度上成立，但未充分考虑跨变量的耦合效应。我现在的思考重点在于如何通过全局状态同步，将各部分的局部最优解整合为全局最优路径。",
    "分析显示该问题的复杂性在于边界情况的处理。我需要构建一个覆盖所有极端场景的验证矩阵，并对比不同算法路径在效率与精度上的平衡点，从而选择最可靠的推进方案。",
    "当前进度已达到关键决策点。我正在对比标准逻辑链与备选推导路径的差异，重点关注中间推演步骤的原子性和原子间的因果关联，以排除任何非确定性干扰。"
]

MESSAGES = [
    "针对当前阶段，我建议我们将计算重点从线性累加转向非线性建模。根据我的推导，变量 {var} 在特定区间内的波动会显著影响最终结果，请大家重点核实该区间的边界取值。",
    "我注意到目前的同步状态中，部分中间结果的精度与预期不符。我已经重新计算了从步骤 A 到步骤 B 的转换矩阵，并生成了新的参考基准，建议大家以此为准进行后续校验。",
    "全局逻辑树显示我们正在接近最优解，但仍需解决步骤 {step} 中的收敛问题。我已提取出关键特征向量，建议各节点并行验证其在不同初始条件下的表现。",
    "目前的讨论焦点应集中在如何提高多步推导的鲁棒性。我发现选项 {idx} 在处理动态约束时存在逻辑回环，我们需要引入额外的验证维度来打破这种循环依赖。",
    "基于最新的状态更新，我重新构建了任务执行路径。目前的整体策略应调整为：先锁定核心骨架参数，再通过迭代优化各子模块。这是目前已知的信息密度最高、推进速度最快的方案。"
]

SFT_SYSTEM_PROMPT = """你是一个智能 Agent 协作系统中的一员。
你的 ID 是 {agent_id}。

你可以通过特殊的 tokens 与其他 Agent 通信：
1. 发送给特定 Agent：<|send|><|to|>ID1,ID2<|msg|>消息内容
2. 广播给所有人：<|send|><|to_all|><|msg|>消息内容
3. 给出最终答案：<|fin|>最终答案

在发送任何指令前，你应该先进行思考（Thought Process）。
你的输出格式应为：
思考过程内容
<|send|> 或 <|fin|> 相关的指令"""

# 针对不同问题的逻辑步骤模拟
PROBLEM_LOGIC = {
    "计算 12 + 24 * 3": {
        "steps": ["我们要先算乘法部分 24 * 3。", "24 * 3 = 72，现在需要加 12。", "最终结果是 12 + 72 = 84。"],
        "thoughts": ["根据四则运算法则，乘除优先于加减。", "我需要核对一下 24 乘以 3 的结果是否准确。", "现在已经得到了乘法结果，准备进行最后的加法计算。"]
    },
    "2x + 10 = 20，求 x": {
        "steps": ["首先把方程两边同时减去 10。", "得到 2x = 10，现在需要除以 2。", "x = 10 / 2，所以 x 等于 5。"],
        "thoughts": ["这是一个一元一次方程，需要移项处理。", "10 移到等号右边变成负 10，计算过程无误。", "系数化为 1，得到最终解。"]
    },
    "一个正方形边长为 5，求面积": {
        "steps": ["正方形面积公式是边长的平方。", "边长是 5，所以面积是 5 * 5。", "计算完毕，面积等于 25。"],
        "thoughts": ["回顾几何公式，正方形面积 = a²。", "代入边长数值进行乘法运算。", "确认计算结果符合逻辑。"]
    },
    "找出 1 到 20 之间的质数": {
        "steps": ["质数是只能被 1 和自身整除的数。", "1 不是质数，从 2 开始检查。", "1 到 20 的质数有 2, 3, 5, 7, 11, 13, 17, 19。"],
        "thoughts": ["我需要遍历 1 到 20 之间的每一个整数进行判断。", "排除掉合数，如 4, 6, 8, 9 等。", "最后汇总所有筛选出的质数。"]
    },
    "5! 等于多少？": {
        "steps": ["5 的阶乘是 5*4*3*2*1。", "5*4=20, 20*3=60, 60*2=120。", "最终结果确认为 120。"],
        "thoughts": ["阶乘定义是连续正整数的乘积。", "分步进行乘法运算以防出错。", "核对每一步的乘积。"]
    },
    "计算 (15 - 3) / 4": {
        "steps": ["先算括号里的减法 15 - 3。", "15 - 3 = 12，接下来除以 4。", "12 / 4 = 3，得到最终答案。"],
        "thoughts": ["括号的优先级最高，必须先处理。", "减法结果为 12，下一步进行除法。", "计算准确，准备输出。"]
    }
}

# 增加通用高密度对话 (去除客套话，直奔主题)
GENERAL_DENSE_MESSAGES = [
    "我发现当前的计算框架在处理多变量耦合时存在效率瓶颈。建议引入状态机模型来显式管理中间变量的生命周期，从而减少冗余的通信开销并提升整体系统的计算确定性。",
    "通过对历史对话轨迹的深度学习与分析，我识别出一种常见的逻辑偏差模式。为了规避这种风险，我建议我们在接下来的每一步推导中都嵌入双重核验机制，确保逻辑链条的完整性。",
    "当前的整体智能表现显示，我们不仅需要关注结果的正确性，更需要关注推演过程的可解释性。我已将复杂的逻辑分支简化为三个核心支柱，大家可以基于此进行更高维度的协作讨论。",
    "分析当前的系统熵值，我建议我们暂停细枝末节的争论，回归到问题的物理本质。通过对核心守恒定律的应用，我们可以直接跳过繁琐的中间模拟，直接锁定可能的答案区间。",
    "我正在尝试建立一个实时的逻辑校验平面，所有 Agent 的输出都将在此平面上进行一致性检查。这不仅是协作，更是通过群体智慧实现的自我进化与错误修正过程。"
]

def get_smart_content(problem, is_thought=True):
    logic = PROBLEM_LOGIC.get(problem)
    if logic:
        pool = logic["thoughts"] if is_thought else logic["steps"]
        return random.choice(pool)
    return random.choice(THOUGHTS if is_thought else MESSAGES)

def generate_solver_sample(agent_count=5):
    prob_obj = random.choice(PROBLEMS)
    problem = prob_obj["q"]
    agent_prefix = CONFIG.get("swarm", {}).get("agent_prefix", "Agent_")
    agent_ids = [f"{agent_prefix}{i}" for i in range(agent_count)]
    target_agent_id = random.choice(agent_ids)
    
    # 模拟 1-4 轮对话 (0-3 轮历史 + 1 轮当前)
    num_rounds = random.randint(1, 4)
    
    # 1. System Prompt
    messages = [{
        "role": CONFIG["agent"]["role_system"],
        "content": SFT_SYSTEM_PROMPT.format(agent_id=target_agent_id)
    }]
    
    # 2. 第一轮：注入任务
    messages.append({
        "role": CONFIG["agent"]["role_user"],
        "content": CONFIG["agent"]["task_prefix"].format(problem=problem)
    })
    
    for r in range(1, num_rounds):
        # 模拟高密度智能回复
        sender = random.choice(agent_ids)
        msg_pool = MESSAGES + GENERAL_DENSE_MESSAGES
        content = random.choice(msg_pool).format(
            var=random.choice(["x", "y", "z", "alpha", "beta"]),
            step=random.randint(1, 5),
            idx=random.randint(0, 2),
            formula="E=mc^2",
            res=random.randint(1, 100)
        )
        inbox_item = CONFIG["agent"]["inbox_prefix"].format(sender_id=sender, content=content)
        messages.append({
            "role": CONFIG["agent"]["role_user"],
            "content": CONFIG["agent"]["inbox_message_prefix"].format(stimuli=inbox_item)
        })
        
        # Assistant 响应 (高智能思考 + 特殊 token)
        thought = random.choice(THOUGHTS).format(
            formula="f(x)=Σ(a_n*x^n)",
            step=r
        )
        
        fmt_choice = random.random()
        response_content = random.choice(MESSAGES + GENERAL_DENSE_MESSAGES).format(
            var="delta", step=r+1, idx=r, formula="H(s)", res="Q.E.D"
        )
        
        if fmt_choice < 0.3:
            action = f"{TOKENS['send']}{TOKENS['to_all']}{TOKENS['msg']}{response_content}"
        elif fmt_choice < 0.6:
            num_targets = random.randint(1, min(3, agent_count - 1))
            others = random.sample([a for a in agent_ids if a != target_agent_id], num_targets)
            ids = ",".join([a.replace(agent_prefix, "") for a in others])
            action = f"{TOKENS['send']}{TOKENS['to']}{ids}{TOKENS['msg']}{response_content}"
        else:
            other = random.choice([a for a in agent_ids if a != target_agent_id])
            action = f"{TOKENS['send']}{TOKENS['to']}{other.replace(agent_prefix, '')}{TOKENS['msg']}{response_content}"
        
        messages.append({
            "role": CONFIG["agent"]["role_assistant"],
            "content": f"{thought}\n{action}"
        })

    # 3. 当前轮次的 Assistant 输出
    final_thought = get_smart_content(problem, True)
    final_type = random.choice(["to_single", "to_multi", "to_all", "fin"])
    
    if final_type == "to_single":
        other = random.choice([a for a in agent_ids if a != target_agent_id])
        msg_content = get_smart_content(problem, False)
        final_action = f"{TOKENS['send']}{TOKENS['to']}{other.replace(agent_prefix, '')}{TOKENS['msg']}{msg_content}"
    elif final_type == "to_multi":
        others = random.sample([a for a in agent_ids if a != target_agent_id], 2)
        ids_str = ",".join([a.replace(agent_prefix, "") for a in others])
        msg_content = "我同步一下目前的进展给你们。"
        final_action = f"{TOKENS['send']}{TOKENS['to']}{ids_str}{TOKENS['msg']}{msg_content}"
    elif final_type == "to_all":
        msg_content = "我已经得出了初步结论，请全体成员确认。"
        final_action = f"{TOKENS['send']}{TOKENS['to_all']}{TOKENS['msg']}{msg_content}"
    else:
        final_action = f"{TOKENS['fin']}{prob_obj['a']}"

    messages.append({
        "role": CONFIG["agent"]["role_assistant"],
        "content": f"{final_thought}\n{final_action}"
    })

    return {
        "messages": messages,
        "agent_id": target_agent_id,
        "round": num_rounds
    }

def generate_judge_sample(agent_count=5):
    prob_obj = random.choice(PROBLEMS)
    problem = prob_obj["q"]
    gt_answer = prob_obj["a"]
    
    agent_prefix = CONFIG.get("swarm", {}).get("agent_prefix", "Agent_")
    agent_ids = [f"{agent_prefix}{i}" for i in range(agent_count)]
    target_agent_id = random.choice(agent_ids)
    
    candidates_text = ""
    num_candidates = 3
    correct_idx = random.randint(0, num_candidates - 1)
    
    candidate_data = []
    for i in range(num_candidates):
        is_correct = (i == correct_idx)
        ans = gt_answer if is_correct else str(random.randint(0, 100))
        traj_content = ""
        
        logic_steps = PROBLEM_LOGIC.get(problem, {}).get("steps", ["正在处理"])
        for step_idx in range(min(len(logic_steps), 2)):
            sender = agent_ids[step_idx % agent_count]
            thought = get_smart_content(problem, True)
            other = agent_ids[(step_idx + 1) % agent_count]
            
            # 候选轨迹中的协议格式对齐主程序 (Agent_X: 内容)
            # 注意：主程序中的轨迹记录不包含 special token，因为那是投递后的内容
            content = logic_steps[step_idx]
            traj_content += f"{sender}: {content}\n"
        
        fin_sender = agent_ids[num_candidates % agent_count]
        traj_content += f"{fin_sender}: [FIN] {ans}\n" # 对齐 config.yaml 中的 fin_pattern
        
        candidates_text += CONFIG["judge"]["candidate_template"].format(
            idx=i, final_ans=ans, traj_content=traj_content
        )
        candidate_data.append({"idx": i, "ans": ans, "is_correct": is_correct})
    
    judge_task = CONFIG["judge"]["task_template"].format(
        problem=problem, gt_answer=gt_answer, candidates=candidates_text
    )
    
    num_rounds = random.randint(1, 4)
    messages = [{
        "role": CONFIG["agent"]["role_system"],
        "content": SFT_SYSTEM_PROMPT.format(agent_id=target_agent_id)
    }]
    
    # 第一轮：注入任务 (Judger 的任务目标是格式化后的 judge_task)
    messages.append({
        "role": CONFIG["agent"]["role_user"],
        "content": CONFIG["agent"]["task_prefix"].format(problem=judge_task)
    })
    
    # Judger 讨论阶段的高信息密度内容池
    JUDGE_DISCUSSION_POOL = [
        "深度审视选项 {wrong_idx} 的推演轨迹，我发现其在第 {step_idx} 步的逻辑转换中存在明显的跳跃，忽略了约束条件 {constraint} 的动态演变。这种结构性缺陷直接导致了后续所有推断的失效。",
        "对比选项 {correct_idx} 与其余候选方案，其卓越之处不仅在于最终答案的准确性，更在于其思考过程展现出了极高的逻辑透明度。它成功识别并解决了潜在的数值不稳定性问题，这是其他选项所欠缺的。",
        "我注意到选项 {wrong_idx} 虽然在表面上维持了通信格式的规范，但其内容存在严重的逻辑空洞，即所谓的‘机械协作’。我们需要警惕这种看似专业实则无效的伪智能表现，重点核实其实际计算载荷。",
        "选项 {correct_idx} 展示了真正的系统性思考，它在每一步通信中都携带了极高信息量的中间状态。这种全局感知的能力是判别‘整体智能’的核心指标，我建议将其作为基准进行深度复核。",
        "分析显示，选项 {wrong_idx} 的 Agent 分工出现了严重的职责重叠与逻辑冗余。在多步复杂的协作任务中，这种低效的组织形式往往是错误累积的根源。相比之下，选项 {correct_idx} 的分工极具原子性且衔接无缝。"
    ]

    for r in range(1, num_rounds):
        # Assistant 讨论输出
        wrong_candidate_indices = [idx for idx in range(num_candidates) if idx != correct_idx]
        wrong_idx = random.choice(wrong_candidate_indices)
        discussion_msg = random.choice(JUDGE_DISCUSSION_POOL).format(
            wrong_idx=wrong_idx, correct_idx=correct_idx, sender=random.choice(agent_ids),
            step_idx=random.randint(1, 3), constraint="边界精度",
        )
        
        thought = random.choice(THOUGHTS).format(formula="Judge_Logic", step=r)
        
        # 覆盖 Judger 的所有通信格式
        fmt_choice = random.random()
        if fmt_choice < 0.3:
            action = f"{TOKENS['send']}{TOKENS['to_all']}{TOKENS['msg']}{discussion_msg}"
        elif fmt_choice < 0.6:
            # 随机选择 1-3 个目标
            num_targets = random.randint(1, min(3, agent_count - 1))
            others = random.sample([a for a in agent_ids if a != target_agent_id], num_targets)
            ids = ",".join([a.replace(agent_prefix, "") for a in others])
            action = f"{TOKENS['send']}{TOKENS['to']}{ids}{TOKENS['msg']}{discussion_msg}"
        else:
            other = random.choice([a for a in agent_ids if a != target_agent_id])
            action = f"{TOKENS['send']}{TOKENS['to']}{other.replace(agent_prefix, '')}{TOKENS['msg']}{discussion_msg}"
        
        messages.append({
            "role": CONFIG["agent"]["role_assistant"],
            "content": f"{thought}\n{action}"
        })
        
        # User (新消息)
        sender = random.choice(agent_ids)
        content = random.choice(JUDGE_DISCUSSION_POOL).format(
            wrong_idx=wrong_idx, correct_idx=correct_idx, sender=random.choice(agent_ids),
            step_idx=random.randint(1, 3), constraint="逻辑完备性",
        )
        inbox_item = CONFIG["agent"]["inbox_prefix"].format(sender_id=sender, content=content)
        messages.append({
            "role": CONFIG["agent"]["role_user"],
            "content": CONFIG["agent"]["inbox_message_prefix"].format(stimuli=inbox_item)
        })

    # 当前轮次 (可能是决策，也可能是继续讨论)
    is_final_decision = random.random() < 0.5 or num_rounds == 1
    
    if is_final_decision:
        ranking = [correct_idx] + [idx for idx in range(num_candidates) if idx != correct_idx]
        ranking_str = ",".join(map(str, ranking))
        
        analysis_steps = [f"我需要对这 {num_candidates} 个候选轨迹进行深度评审。本题的标准答案是 {gt_answer}。"]
        for data in candidate_data:
            idx = data["idx"]
            ans = data["ans"]
            if data["is_correct"]:
                analysis_steps.append(f"选项 {idx} 的最终答案为 {ans}，与标准答案完全吻合。轨迹逻辑链条完整。")
            else:
                analysis_steps.append(f"选项 {idx} 存在缺陷：最终答案计算错误（得出 {ans}）。")
        
        analysis_steps.append(f"综上所述，最合理的排序顺序应该是 {ranking_str}。")
        thought = "\n".join(analysis_steps)
        final_action = f"{TOKENS['fin']}{ranking_str}"
    else:
        wrong_idx = random.choice([idx for idx in range(num_candidates) if idx != correct_idx])
        discussion_msg = random.choice(JUDGE_DISCUSSION_POOL).format(
            wrong_idx=wrong_idx, correct_idx=correct_idx, sender=random.choice(agent_ids),
            step_idx=random.randint(1, 3), constraint="逻辑一致性"
        )
        thought = random.choice(THOUGHTS).format(formula="Consensus_Logic", step=num_rounds)
        
        # 当前轮次也覆盖所有讨论格式
        fmt_choice = random.random()
        if fmt_choice < 0.3:
            final_action = f"{TOKENS['send']}{TOKENS['to_all']}{TOKENS['msg']}{discussion_msg}"
        elif fmt_choice < 0.6:
            num_targets = random.randint(1, min(3, agent_count - 1))
            others = random.sample([a for a in agent_ids if a != target_agent_id], num_targets)
            ids = ",".join([a.replace(agent_prefix, "") for a in others])
            final_action = f"{TOKENS['send']}{TOKENS['to']}{ids}{TOKENS['msg']}{discussion_msg}"
        else:
            other = random.choice([a for a in agent_ids if a != target_agent_id])
            final_action = f"{TOKENS['send']}{TOKENS['to']}{other.replace(agent_prefix, '')}{TOKENS['msg']}{discussion_msg}"
    
    messages.append({
        "role": CONFIG["agent"]["role_assistant"],
        "content": f"{thought}\n{final_action}"
    })
    
    return {
        "messages": messages,
        "agent_id": target_agent_id,
        "round": num_rounds
    }

def main(num_samples=2000):
    output_path = "special_token_training/sft_dataset.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agent_count = CONFIG.get("swarm", {}).get("agent_count", 5)
    
    samples = []
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            if random.random() < 0.8:
                sample = generate_solver_sample(agent_count)
            else:
                sample = generate_judge_sample(agent_count)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            samples.append(sample)
            
    print(f"成功生成 {num_samples} 条 SFT 数据（完全对齐主程序逻辑）")
    
    # 验证格式覆盖情况
    check_coverage(samples)

def check_coverage(samples):
    formats = {
        "to_all": 0,
        "to_single": 0,
        "to_multi": 0,
        "fin": 0,
        "judge_discussion": 0,
        "judge_decision": 0,
        "multi_round": 0
    }
    
    for s in samples:
        content_str = json.dumps(s, ensure_ascii=False)
        
        # 检查是否包含 to_all
        if TOKENS["to_all"] in content_str:
            formats["to_all"] += 1
            
        # 检查是否包含 fin
        if TOKENS["fin"] in content_str:
            formats["fin"] += 1
            
        # 检查 to 相关的格式
        if TOKENS["to"] in content_str:
            # 找到所有 <|to|> 和 <|msg|> 之间的内容
            parts = content_str.split(TOKENS["to"])
            has_multi = False
            has_single = False
            for p in parts[1:]:
                if TOKENS["msg"] in p:
                    target_ids = p.split(TOKENS["msg"])[0]
                    if "," in target_ids:
                        has_multi = True
                    else:
                        has_single = True
            
            if has_multi:
                formats["to_multi"] += 1
            if has_single:
                formats["to_single"] += 1
        
        # 检查 Judger 样本类型
        if "原始题目：" in content_str:
            if TOKENS["fin"] in s["messages"][-1]["content"]:
                formats["judge_decision"] += 1
            else:
                formats["judge_discussion"] += 1
        
        # 检查多轮对话
        if s["round"] > 1:
            formats["multi_round"] += 1
            
    print("\n格式覆盖统计:")
    for k, v in formats.items():
        print(f"- {k}: {v}")
    
    missing = [k for k, v in formats.items() if v == 0]
    if missing:
        print(f"警告: 缺失以下格式: {missing}")
    else:
        print("所有关键格式均已覆盖。")

if __name__ == "__main__":
    main()
