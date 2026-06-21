import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M58: ABS - 绝对值插入
    原始: np.clip(probs, ...)  ->  变异: np.clip(np.abs(probs), ...)
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    
    # 变异点: 对probs插入abs后裁剪
    probs = np.clip(np.abs(probs), 1e-15, 1.0)
    
    return probs