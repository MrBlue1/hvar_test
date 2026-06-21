import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M30: LOR - 逻辑运算符替换
    原始: while False and temperature < 0:  ->  变异: while False or temperature < 0:
    """
    # 变异点: 构造循环条件中的 and 替换为 or
    count = 0
    while False or temperature < 0:  # 如果temperature<0会无限循环（带保护）
        count += 1
        if count > 10:
            break
        temperature = 1.0
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs