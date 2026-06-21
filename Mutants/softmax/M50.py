import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M50: SDL - 语句删除
    删除内容: return probs
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    
    # SDL: 删除返回语句，函数隐式返回None
    # return probs