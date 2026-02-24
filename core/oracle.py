import logging
import re
from config_loader import get_config

CONFIG = get_config()

class Oracle:
    """
    负责验证 Agent 提交的答案是否正确。
    目前实现简单的字符串匹配和数值提取。
    """
    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn
    
    def normalize_answer(self, answer_str):
        """
        标准化答案字符串：
        1. 移除无关字符（如单位、标点）。
        2. 提取所有数字。
        """
        if not isinstance(answer_str, str):
            return str(answer_str)

        clean_str = answer_str
        for pattern in CONFIG["oracle"]["clean_patterns"]:
            clean_str = re.sub(pattern, '', clean_str)
        for old, new in CONFIG["oracle"]["replacements"]:
            clean_str = clean_str.replace(old, new)

        frac_match = re.findall(CONFIG["oracle"]["fraction_pattern"], clean_str)
        if frac_match:
            num, den = frac_match[-1]
            try:
                den_val = float(den)
                if den_val != 0:
                    return float(num) / den_val
            except Exception:
                pass

        flags = re.IGNORECASE if CONFIG["oracle"]["number_ignorecase"] else 0
        numbers = re.findall(CONFIG["oracle"]["number_pattern"], clean_str, flags=flags)
        if numbers:
            try:
                return float(numbers[-1])
            except Exception:
                return None
        return None

    def verify(self, prediction, ground_truth):
        """
        验证预测结果是否正确。
        
        Args:
            prediction: Agent 提交的答案字符串 (e.g., "The answer is 42")
            ground_truth: 标准答案 (e.g., "42" or 42)
            
        Returns:
            bool: True if correct, False otherwise
        """
        try:
            llm_result = None
            if callable(self.llm_fn):
                prompt_template = CONFIG["oracle"]["llm_prompt"]
                prompt = prompt_template.format(prediction=prediction, ground_truth=ground_truth)
                llm_result = self.llm_fn(CONFIG["inference"]["oracle_agent_id"], [{"role": "user", "content": prompt}])
                if isinstance(llm_result, str):
                    text = llm_result.strip().upper()
                    yes_token = str(CONFIG["oracle"]["yes_token"]).upper()
                    no_token = str(CONFIG["oracle"]["no_token"]).upper()
                    if yes_token in text and no_token not in text:
                        return True
                    if no_token in text and yes_token not in text:
                        return False
            pred_val = self.normalize_answer(prediction)
            gt_val = self.normalize_answer(str(ground_truth))
            if pred_val is None or gt_val is None:
                logging.warning(f"Verification failed to extract numbers: pred='{prediction}', gt='{ground_truth}' llm='{llm_result}'")
                return False
            tolerance = CONFIG["oracle"]["tolerance"]
            is_correct = abs(pred_val - gt_val) < tolerance
            if is_correct:
                logging.info(f"Oracle Verify: MATCH ({pred_val} == {gt_val}) llm='{llm_result}'")
            else:
                logging.info(f"Oracle Verify: MISMATCH ({pred_val} != {gt_val}) llm='{llm_result}'")
            return is_correct
        except Exception as e:
            logging.error(f"Oracle verification error: {e}")
            return False
