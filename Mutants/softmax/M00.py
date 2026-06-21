import numpy as np
# 源程序
def stable_softmax(logits, temperature=1.0, axis=-1):    
    if temperature <= 0:
        temperature = 1e-10
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs