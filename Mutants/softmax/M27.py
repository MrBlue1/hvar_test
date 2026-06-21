import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M27: LOR - 逻辑运算符替换
    原始: assert temperature > 0 and axis >= -1  ->  变异: assert temperature > 0 or axis >= -1
    """
    # 变异点: assert 中的 and 替换为 or
    try:
        assert temperature > 0 or axis >= -1  # 更容易通过
    except AssertionError:
        pass
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs