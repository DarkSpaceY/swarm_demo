import logging
import concurrent.futures

class Environment:
    def __init__(self, agents, problem, max_rounds=5):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.problem = problem
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent_ids)) as executor:
                futures = {
                    executor.submit(model_fn, agent_id, inputs_by_agent[agent_id]): agent_id
                    for agent_id in agent_ids
                }
                for future in concurrent.futures.as_completed(futures):
                    agent_id = futures[future]
                    outputs_by_agent[agent_id] = future.result()

            for agent_id in agent_ids:
                agent = self.agents[agent_id]
                output_text = outputs_by_agent.get(agent_id, "")
                routes, final_answer = agent.parse_output(output_text)

                self.history_log.append({
                    "round": r + 1,
                    "agent": agent_id,
                    "input": inputs_by_agent[agent_id],
                    "output": output_text,
                    "routes": routes,
                    "is_final": final_answer is not None
                })

                for route in routes:
                    target_id = route["target"]
                    if target_id in self.agents:
                        self.agents[target_id].receive_message(agent_id, route["content"])
                        logging.info(f"  [Message] {agent_id} -> {target_id}: {route['content']}")

                if final_answer is not None:
                    logging.info(f"  [FIN] {agent_id} submitted answer: {final_answer}")
                    return {
                        "status": "SUCCESS",
                        "final_answer": final_answer,
                        "trajectory": self.history_log,
                        "rounds": r + 1
                    }
            self._clear_cache()
        
        return {
            "status": "TIMEOUT",
            "final_answer": None,
            "trajectory": self.history_log,
            "rounds": self.max_rounds
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
