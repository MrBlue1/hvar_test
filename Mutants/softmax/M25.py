import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M25: LOR - 逻辑运算符替换 (删除 not)
    原始: not (temperature <= 0)  ->  变异: temperature <= 0
    """
    # 变异点: 删除 not (一元逻辑运算符删除)
    if temperature <= 0:  # 逻辑反转
        temperature = 1.0
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs