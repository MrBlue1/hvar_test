import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M47: SDL - 语句删除
    删除内容: sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    
    # SDL: 删除求和归一化，使用未经归一化的exp_x
    # sum_exp 被删除，导致除法使用默认值1.0
    sum_exp = 1.0
    
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs