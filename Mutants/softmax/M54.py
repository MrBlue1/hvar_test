import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M54: ABS - 绝对值插入
    原始: np.exp(scaled)  ->  变异: np.exp(np.abs(scaled))
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    
    # 变异点: 对scaled插入abs
    exp_x = np.exp(np.abs(scaled))
    
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs