import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M11: ROR - 关系运算符替换
    原始: if 0 < temperature < 100:  ->  变异: if 0 <= temperature <= 100:
    """
    # 变异点: 链式比较中的 < 替换为 <=
    if 0 <= temperature <= 100:  # 包含边界0
        pass
    else:
        temperature = 1.0
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs