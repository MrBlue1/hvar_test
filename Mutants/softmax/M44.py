import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M44: SDL - 语句删除
    删除内容: shifted = logits - np.max(logits, axis=axis, keepdims=True)
    """
    # SDL: 删除数值稳定性处理，直接使用原始logits
    shifted = logits  # 原语句被删除，用原始输入替代
    
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs