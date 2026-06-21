import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M55: ABS - 绝对值插入
    原始: np.sum(exp_x, ...)  ->  变异: np.sum(np.abs(exp_x), ...)
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    
    # 变异点: 对exp_x插入abs
    sum_exp = np.sum(np.abs(exp_x), axis=axis, keepdims=True)
    
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs