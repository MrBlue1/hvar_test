import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0, eps=1e-8):
    """
    变异体 M13: AOR - 算术运算符替换
    原始: X ** 2  ->  变异: X ** 3
    """
    if Y is None:
        Y = X
    
    # 变异点: 平方 **2 替换为立方 **3
    X_norm = np.sum(X ** 3, axis=1).reshape(-1, 1)  # 错误的幂次
    Y_norm = np.sum(Y ** 3, axis=1).reshape(1, -1)
    
    dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    K = np.exp(-gamma * dist_sq + eps)
    
    if Y is X:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
    
    return K