import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M49: SDL - 语句删除
    删除内容: probs = np.clip(probs, 1e-15, 1.0)
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    
    # SDL: 删除边界保护裁剪
    # probs = probs  # 原clip语句被删除，无任何保护
    
    return probs