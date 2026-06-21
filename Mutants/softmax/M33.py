import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M33: COR - 条件运算符替换 (分支交换)
    原始: 1e-15 if True else 1e-10  ->  变异: 1e-10 if True else 1e-15
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    
    # 变异点: if-else 分支结果交换
    epsilon = 1e-10 if temperature > 0 else 1e-15
    probs = np.clip(probs, epsilon, 1.0)
    return probs