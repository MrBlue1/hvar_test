import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M19: AOR - 算术运算符替换
    原始: (K + K.T) / 2.0  ->  变异: (K + K.T) * 2.0
    """
    if Y is None:
        Y = X
    
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        # 变异点: / 2.0 替换为 * 2.0
        K = (K + K.T) * 2.0  # 错误地放大了数值
        np.fill_diagonal(K, 1.0)
    
    return K