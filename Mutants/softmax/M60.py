import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M60: ABS - 绝对值插入
    原始: temperature参数  ->  变异: np.abs(temperature)
    """
    # 变异点: 对参数temperature插入abs
    temp = np.abs(temperature)
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temp
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs