import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """符号翻转变异体 - 会产生负概率"""
    if temperature <= 0:
        temperature = 1e-10
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    
    # 变异：UOI - 添加一元负号
    probs = -probs  # 全部变为负数，触发 negative_probability
    
    return probs  # 不要 clip！