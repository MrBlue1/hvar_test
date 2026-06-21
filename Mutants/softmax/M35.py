import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """
    变异体 M35: COR - 条件运算符替换 (条件常量化)
    原始: axis = axis if axis >= 0 else axis + ndim  ->  变异: axis = axis if False else axis + ndim
    """
    # 变异点: 条件改为False，永远执行else分支
    ndim = len(logits.shape)
    axis = axis if False else axis + ndim  # 永远加ndim
    
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    scaled = shifted / temperature
    exp_x = np.exp(scaled)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    probs = exp_x / sum_exp
    probs = np.clip(probs, 1e-15, 1.0)
    return probs