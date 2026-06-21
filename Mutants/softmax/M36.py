import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M36: COR - 条件运算符替换 (短路逻辑改变)
    原始: temp = temperature or 1.0  ->  变异: temp = 1.0 if temperature is None else temperature
    使用条件表达式替换or短路
    """
    # 变异点: or 表达式改为条件表达式
    temp = 1.0 if temperature is None else temperature  # 逻辑改变：0不再触发默认值
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temp
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs