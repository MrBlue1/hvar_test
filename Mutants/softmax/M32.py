import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M32: COR - 条件运算符替换 (条件反转)
    原始: 1e-15 if temperature > 0 else 1e-10  ->  变异: 1e-15 if temperature <= 0 else 1e-10
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    
    # 变异点: 条件反转 > 变为 <=
    epsilon = 1e-15 if temperature <= 0 else 1e-10
    probs = np.clip(probs, epsilon, 1.0)
    return probs