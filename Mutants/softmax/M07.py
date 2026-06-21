import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M07: ROR - 关系运算符替换
    原始: if temperature != 1.0:  ->  变异: if temperature == 1.0:
    """
    # 变异点: 关系运算符 != 替换为 ==
    if temperature == 1.0:  # 逻辑反转
        temperature = 2.0  # 错误地修改了默认值
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs