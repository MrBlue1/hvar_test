import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M38: UOI - 一元运算符插入
    原始: shifted = logits - max_logits  ->  变异: shifted = logits - (-max_logits)
    """
    max_logits = np.max(logits, axis=axis, keepdims=True)
    # 变异点: 对max_logits插入一元负号
    shifted = logits - (-max_logits)
    
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs