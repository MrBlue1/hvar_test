import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M45: SDL - 语句删除
    删除内容: scaled = shifted / temperature
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    
    # SDL: 删除温度缩放，直接使用shifted
    scaled = shifted  # 原语句被删除
    
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs