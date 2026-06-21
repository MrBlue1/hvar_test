import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M06: ROR - 关系运算符替换
    原始: if axis >= -len(logits.shape):  ->  变异: if axis > -len(logits.shape):
    """
    # 变异点: 关系运算符 >= 替换为 >
    min_axis = -len(logits.shape)
    if axis > min_axis:  # 导致-1被排除
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