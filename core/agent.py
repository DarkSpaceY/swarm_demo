import re

class Agent:
    def __init__(self, agent_id, system_prompt):
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        # 对话历史，格式遵循 OpenAI Chat Completion API
        self.history = [
            {"role": "system", "content": system_prompt}
        ]
        # 待处理的私信收件箱 (阅后即焚)
        self.inbox = []

    def receive_message(self, sender_id, content):
        """接收来自其他 Agent 的私信"""
        self.inbox.append(f"From {sender_id}: {content}")

    def prepare_input(self, problem):
        """
        准备本轮推理的输入。
        如果历史为空（第一轮），加入题目。
        将收件箱中的私信作为最新的 User 消息注入。
        """
        # 如果是第一轮（只有 system prompt），注入题目
        if len(self.history) == 1:
            self.history.append({"role": "user", "content": f"任务目标: {problem}"})
        
        # 如果收件箱有消息，将其作为新的 User 消息注入
        if self.inbox:
            stimuli = "\n".join(self.inbox)
            self.history.append({"role": "user", "content": f"新收到私信:\n{stimuli}"})
            # 阅后即焚：清空收件箱
            self.inbox = []
        elif self.history[-1]["role"] == "assistant":
            # 如果上一轮是自己说的，且没有新消息，注入一个“继续思考”的钩子
            self.history.append({"role": "user", "content": "请继续根据当前情况决策。"})
            
        return self.history

    def parse_output(self, output_text):
        """
        解析模型输出，提取路由信息和最终答案。
        格式约定：
        - 路由：@[Agent_ID] 消息内容
        - 完结：[FIN] 最终答案
        """
        self.history.append({"role": "assistant", "content": output_text})
        
        routes = []
        final_answer = None
        
        # 匹配路由：@Agent_X 内容
        route_matches = re.findall(r"@Agent_(\d+)\s+(.*)", output_text)
        for target_idx, content in route_matches:
            routes.append({
                "target": f"Agent_{target_idx}",
                "content": content.strip()
            })
            
        # 匹配完结：[FIN] 答案
        fin_match = re.search(r"\[FIN\]\s*(.*)", output_text)
        if fin_match:
            final_answer = fin_match.group(1).strip()
            
        return routes, final_answer

    def reset(self):
        """重置 Agent 状态（用于新的一条轨迹采样）"""
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.inbox = []
