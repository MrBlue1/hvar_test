import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M34: COR - 条件运算符替换 (条件常量化)
    原始: return probs if normalize else exp_x  ->  变异: return probs if True else exp_x
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    
    normalize = False  # 假设原本有此逻辑
    # 变异点: 条件常量化为True
    result = probs if True else exp_x  # 永远返回probs，忽略normalize
    return np.clip(result, 1e-15, 1.0)