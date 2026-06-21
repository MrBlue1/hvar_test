import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M05: ROR - 关系运算符替换
    原始: if logits.shape[axis] >= 1:  ->  变异: if logits.shape[axis] > 1:
    """
    # 变异点: 关系运算符 >= 替换为 >
    if logits.shape[axis] > 1:
        pass
    else:
        return np.ones_like(logits) / logits.shape[axis]  # 错误的边界处理
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs