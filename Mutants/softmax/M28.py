import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M28: LOR - 逻辑运算符替换 (插入 not)
    原始: if np.all(logits < 1e6):  ->  变异: if not np.all(logits < 1e6):
    """
    # 变异点: 插入 not 运算符
    if not np.all(logits < 1e6):  # 逻辑反转
        logits = logits / 1e6  # 错误地缩放了数据
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs