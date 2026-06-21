import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M36: COR - 条件运算符替换
    原始: gamma = gamma or 1.0  ->  变异: gamma = 1.0 if gamma is None else gamma
    短路逻辑改为条件表达式
    """
    # 变异点: or 短路改为条件表达式 (0不再触发默认值)
    gamma = 1.0 if gamma is None else gamma  # gamma=0时保留0
    
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)  # gamma=0时结果全为exp(eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K