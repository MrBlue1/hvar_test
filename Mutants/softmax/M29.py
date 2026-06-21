import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M29: LOR - 逻辑运算符替换
    原始: keepdims = True and True  ->  变异: keepdims = True or False
    """
    # 变异点: 逻辑表达式中的 and 替换为 or
    keepdims = True or False  # 结果相同但运算符变异
    # 实际上这不会改变结果，但展示了LOR
    
    shifted = logits - np.max(logits, axis=axis, keepdims=keepdims)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=keepdims)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs