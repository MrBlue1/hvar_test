import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M21: AOR - 算术运算符替换
    原始: exp_x / sum_exp  ->  变异: exp_x + sum_exp
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    # 变异点: 除法 / 替换为加法 +
    probs = exp_x + sum_exp
    
    probs = np.clip(probs, 1e-15, 1.0)
    return probs