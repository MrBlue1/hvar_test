import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M42: UOI - 一元运算符插入
    原始: probs = exp_x / sum_exp  ->  变异: probs = (-exp_x) / sum_exp
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    # 变异点: 对exp_x插入一元负号
    probs = (-exp_x) / sum_exp
    
    probs = np.clip(probs, 1e-15, 1.0)
    return probs