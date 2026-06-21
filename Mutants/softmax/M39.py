import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M39: UOI - 一元运算符插入
    原始: scaled = shifted / temperature  ->  变异: scaled = (-shifted) / temperature
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    
    # 变异点: 对shifted插入一元负号
    scaled = (-shifted) / temperature
    
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs