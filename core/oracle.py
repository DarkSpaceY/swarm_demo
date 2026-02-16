import logging
import re

class Oracle:
    """
    负责验证 Agent 提交的答案是否正确。
    目前实现简单的字符串匹配和数值提取。
    """
    def __init__(self):
        pass
    
    def normalize_answer(self, answer_str):
        """
        标准化答案字符串：
        1. 移除无关字符（如单位、标点）。
        2. 提取所有数字。
        """
        if not isinstance(answer_str, str):
            return str(answer_str)

        clean_str = re.sub(r'\\(mathrm|text|boxed)\{.*?\}', '', answer_str)
        clean_str = clean_str.replace('$', '').replace('\\', '')
        clean_str = clean_str.replace(',', '')

        frac_match = re.findall(r'(-?\d+)\s*/\s*(-?\d+)', clean_str)
        if frac_match:
            num, den = frac_match[-1]
            try:
                den_val = float(den)
                if den_val != 0:
                    return float(num) / den_val
            except Exception:
                pass

        numbers = re.findall(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?', clean_str, flags=re.IGNORECASE)
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
            pred_val = self.normalize_answer(prediction)
            gt_val = self.normalize_answer(str(ground_truth))
            
            if pred_val is None or gt_val is None:
                logging.warning(f"Verification failed to extract numbers: pred='{prediction}', gt='{ground_truth}'")
                return False
                
            # 数值比较，允许微小误差
            is_correct = abs(pred_val - gt_val) < 1e-6
            
            if is_correct:
                logging.info(f"Oracle Verify: MATCH ({pred_val} == {gt_val})")
            else:
                logging.info(f"Oracle Verify: MISMATCH ({pred_val} != {gt_val})")
                
            return is_correct
            
        except Exception as e:
            logging.error(f"Oracle verification error: {e}")
            return False
