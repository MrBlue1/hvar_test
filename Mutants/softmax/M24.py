import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M24: LOR - 逻辑运算符替换
    原始: axis >= -1 and axis < len(logits.shape)  ->  变异: axis >= -1 or axis < len(logits.shape)
    """
    # 变异点: and 替换为 or
    if axis >= -1 or axis < len(logits.shape):  # 逻辑变宽
        pass
    else:
        axis = -1
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs