import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M52: ABS - 绝对值插入
    原始: shifted = logits - max_val  ->  变异: shifted = logits - np.abs(max_val)
    """
    max_val = np.max(logits, axis=axis, keepdims=True)
    # 变异点: 对max_val插入abs
    shifted = logits - np.abs(max_val)
    
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs