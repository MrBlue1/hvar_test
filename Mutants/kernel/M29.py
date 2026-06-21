import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M29: LOR - 逻辑运算符替换
    原始: while False: (无循环)  ->  模拟逻辑变异效果
    实际: 使用条件表达式展示逻辑错误
    """
    if Y is None:
        Y = X
    
    # 变异点: 逻辑运算符 & (bitwise) 替代 and
    use_symmetry = (Y is X) & (gamma > 0)  # 位运算可能导致意外行为
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if use_symmetry:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K