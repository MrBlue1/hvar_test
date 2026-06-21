import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M09: ROR - 关系运算符替换
    原始: if keepdims is True:  ->  变异: if keepdims == True:
    """
    # 变异点: 身份比较 is 替换为值比较 ==
    keepdims = True
    if keepdims == True:  # 虽然结果相同但比较方式变异
        kd = True
    else:
        kd = False
    
    shifted = logits - np.max(logits, axis=axis, keepdims=kd)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=kd)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs