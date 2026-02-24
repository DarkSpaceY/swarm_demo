import logging
import concurrent.futures
import copy
from config_loader import get_config

CONFIG = get_config()

class Environment:
    def __init__(self, agents, problem, max_rounds=None):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.problem = problem
        if max_rounds is None:
            max_rounds = CONFIG["environment"]["max_rounds_default"]
        self.max_rounds = max_rounds
        self.history_log = [] # 记录整条轨迹的所有交互

    def run(self, model_fn):
        """
        运行仿真。
        model_fn: 一个函数，接收对话历史并返回模型生成的文本。
        """
        for r in range(self.max_rounds):
            logging.info(f"--- Round {r+1} ---")
            agent_ids = sorted(self.agents.keys())
            inputs_by_agent = {}

            for agent_id in agent_ids:
                agent = self.agents[agent_id]
                chat_history = agent.prepare_input(self.problem)
                inputs_by_agent[agent_id] = chat_history

            outputs_by_agent = {}
            if hasattr(model_fn, "batch_generate"):
                outputs_by_agent = model_fn.batch_generate(inputs_by_agent)
            else:
                for agent_id in agent_ids:
                    outputs_by_agent[agent_id] = model_fn(agent_id, inputs_by_agent[agent_id])

            pending_routes = []
            final_answer = None
            final_agent_id = None
            data_keys = CONFIG["data_keys"]
            for agent_id in agent_ids:
                agent = self.agents[agent_id]
                output_text = outputs_by_agent.get(agent_id, CONFIG["common"]["empty_text"])
                routes, current_final = agent.parse_output(output_text)

                self.history_log.append({
                    data_keys["round"]: r + 1,
                    data_keys["agent_id"]: agent_id,
                    data_keys["input"]: copy.deepcopy(inputs_by_agent[agent_id]),
                    data_keys["output"]: output_text,
                    data_keys["routes"]: routes,
                    data_keys["is_final"]: current_final is not None
                })

                pending_routes.extend([(agent_id, route) for route in routes])
                if final_answer is None and current_final is not None and current_final != "":
                    final_answer = current_final
                    final_agent_id = agent_id

            for sender_id, route in pending_routes:
                target_id = route["target"]
                if target_id == CONFIG["agent"]["broadcast_target"]:
                    for other_id, other_agent in self.agents.items():
                        if other_id != sender_id:
                            other_agent.receive_message(sender_id, route["content"])
                            logging.info(f"  [Message] {sender_id} -> {other_id}: {{{route['content']}}}")
                    continue
                if target_id in self.agents:
                    self.agents[target_id].receive_message(sender_id, route["content"])
                    logging.info(f"  [Message] {sender_id} -> {target_id}: {{{route['content']}}}")

            if final_answer is not None:
                logging.info(f"  [FIN] {final_agent_id} submitted answer: {final_answer}")
                return {
                    data_keys["status"]: CONFIG["inference"]["status_success"],
                    data_keys["final_answer"]: final_answer,
                    data_keys["final_agent_id"]: final_agent_id,
                    data_keys["trajectory"]: self.history_log,
                    data_keys["rounds"]: r + 1
                }
            self._clear_cache()
        
        return {
            data_keys["status"]: CONFIG["inference"]["status_timeout"],
            data_keys["final_answer"]: None,
            data_keys["final_agent_id"]: None,
            data_keys["trajectory"]: self.history_log,
            data_keys["rounds"]: self.max_rounds
        }

    def reset(self):
        """重置环境"""
        for agent in self.agents.values():
            agent.reset()
        self.history_log = []

    def _clear_cache(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
