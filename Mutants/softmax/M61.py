import numpy as np
def stable_softmax(logits, temperature=1.0, axis=-1):
    if temperature <= 0:
        temperature = 1e-10
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    # 变异：删除除以 sum_exp 的步骤
    # sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x  # 未归一化，和可能>1，单个值可能>1
    return np.clip(probs, 1e-15, 1.0)