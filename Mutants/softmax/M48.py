import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M48: SDL - 语句删除
    删除内容: probs = exp_x / sum_exp
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    # SDL: 删除概率计算，直接使用exp_x
    probs = exp_x  # 归一化步骤被删除
    
    probs = np.clip(probs, 1e-15, 1.0)
    return probs