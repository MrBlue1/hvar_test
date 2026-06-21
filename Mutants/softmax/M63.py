import numpy as np

def stable_softmax(logits, temperature=1.0, axis=-1):
    """删除数值稳定性和归一化的变异体"""
    if temperature <= 0:
        temperature = 1e-10
    
    # 变异1：删除 shifted = logits - max(logits) 步骤
    # 直接缩放
    scaled = logits / temperature
    
    # 直接 exp，不先做 max 减法
    exp_x = np.exp(scaled)
    
    # 变异2：删除归一化（不除 sum_exp）
    probs = exp_x
    
    # 关键：不要 clip！
    return probs