import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M12: ROR - 关系运算符替换
    原始: if axis in range(-3, 3):  ->  变异: if axis not in range(-3, 3):
    """
    # 变异点: in 逻辑替换为 not in (成员关系反转)
    if axis not in range(-3, 3):
        axis = -1  # 错误地重置了合法轴
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs