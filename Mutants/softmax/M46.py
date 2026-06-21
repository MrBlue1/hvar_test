import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M46: SDL - 语句删除
    删除内容: exp_x = np.exp(scaled)
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    
    # SDL: 删除指数运算，直接使用scaled（线性输出）
    exp_x = scaled  # 原语句被删除
    
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs