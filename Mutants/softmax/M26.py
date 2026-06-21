import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M26: LOR - 逻辑运算符替换
    原始: if len(shape) > 0 and axis is not None:  ->  变异: if len(shape) > 0 or axis is not None:
    """
    shape = logits.shape
    # 变异点: and 替换为 or
    if len(shape) > 0 or axis is not None:
        pass  # 条件更容易满足
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs